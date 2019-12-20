import pickle
import os
from shutil import rmtree
import sys
from time import sleep

import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')
from batcher import SkipGramBatchLoader
from compute_sections import enumerate_section_ids
from model_utils import get_git_revision_hash, render_args, restore_model, save_checkpoint
from vae import VAE
from vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('--data_dir', default='../preprocess/data/')
    parser.add_argument('--experiment', default='default', help='Save path in weights/ for experiment.')
    parser.add_argument('--restore_experiment', default=None, help='Experiment name from which to restore.')

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('-combine_phrases', default=False, action='store_true')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--window', default=5, type=int)
    parser.add_argument('--ns', type=int, default=1)

    # Model Hyperparameters
    parser.add_argument('--encoder_hidden_dim', default=64, type=int, help='hidden dimension for encoder')
    parser.add_argument('--encoder_input_dim', default=64, type=int, help='embedding dimemsions for encoder')
    parser.add_argument('--hinge_loss_margin', default=1.0, type=float, help='reconstruction margin')
    parser.add_argument('--latent_dim', default=100, type=int, help='z dimension')

    args = parser.parse_args()
    args.git_hash = get_git_revision_hash()
    render_args(args)

    # Load Data
    debug_str = '_mini' if args.debug else ''
    phrase_str = '_phrase' if args.combine_phrases else ''

    ids_infile = os.path.join(args.data_dir, 'ids{}{}.npy'.format(debug_str, phrase_str))
    print('Loading data from {}...'.format(ids_infile))
    with open(ids_infile, 'rb') as fd:
        ids = np.load(fd)

    # Load Vocabulary
    vocab_infile = '../preprocess/data/vocab{}{}.pk'.format(debug_str, phrase_str)
    print('Loading vocabulary from {}...'.format(vocab_infile))
    with open(vocab_infile, 'rb') as fd:
        token_vocab = pickle.load(fd)
    print('Loaded vocabulary of size={}...'.format(token_vocab.separator_start_vocab_id))

    print('Collecting document information...')
    section_pos_idxs = np.where(ids <= 0)[0]
    section_id_range = np.arange(token_vocab.separator_start_vocab_id, token_vocab.size())
    section_vocab = Vocab()
    for section_id in section_id_range:
        section_vocab.add_token(token_vocab.get_token(section_id))
    full_section_ids = enumerate_section_ids(ids, section_pos_idxs, token_vocab, section_vocab)

    token_vocab.truncate()

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Training on {}...'.format(device_str))

    batcher = SkipGramBatchLoader(len(ids), section_pos_idxs, batch_size=args.batch_size)

    model = VAE(args, token_vocab.size(), section_vocab.size()).to(args.device)
    if args.restore_experiment is not None:
        prev_args, model, vocab, optimizer_state = restore_model(args.restore_experiment)

    # Instantiate Adam optimizer
    trainable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    if args.restore_experiment is not None:
        optimizer.load_state_dict(optimizer_state)

    # Create model experiments directory or clear if it already exists
    weights_dir = os.path.join('weights', args.experiment)
    if os.path.exists(weights_dir):
        print('Clearing out previous weights in {}'.format(weights_dir))
        rmtree(weights_dir)
    os.mkdir(weights_dir)

    # Make sure it's calculating gradients
    model.train()  # just sets .requires_grad = True
    for epoch in range(1, args.epochs + 1):
        sleep(0.1)  # Make sure logging is synchronous with tqdm progress bar
        print('Starting Epoch={}'.format(epoch))
        batcher.reset()
        num_batches = batcher.num_batches()
        epoch_loss = 0.0
        for _ in tqdm(range(num_batches)):
            # Reset gradients
            optimizer.zero_grad()

            center_ids, context_ids, section_ids, num_contexts = batcher.next(ids, full_section_ids, args.window)
            center_ids_tens = torch.LongTensor(center_ids).to(args.device)
            context_ids_tens = torch.LongTensor(context_ids).to(args.device)
            section_ids_tens = torch.LongTensor(section_ids).to(args.device)

            neg_id_shape = (context_ids.shape[0], context_ids.shape[1])
            neg_ids = token_vocab.neg_sample(size=neg_id_shape)
            neg_ids_tens = torch.LongTensor(neg_ids).to(args.device)

            loss = model(center_ids_tens, section_ids_tens, context_ids_tens, neg_ids_tens, num_contexts)
            loss.backward()  # backpropagate loss

            epoch_loss += loss.item()
            optimizer.step()
        epoch_loss /= float(batcher.num_batches())
        sleep(0.1)
        print('Epoch={}. Loss={}.'.format(epoch, epoch_loss))
        assert not batcher.has_next()

        # Serializing everything from model weights and optimizer state, to to loss function and arguments
        losses_dict = {'losses': {'kl_loss': epoch_loss}}
        checkpoint_fp = os.path.join(weights_dir, 'checkpoint_{}.pth'.format(epoch))
        save_checkpoint(args, model, optimizer, token_vocab, section_vocab, losses_dict, checkpoint_fp=checkpoint_fp)
