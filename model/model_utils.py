import os

import argparse
import torch

from vae import VAE


def compute_kl(mu_a, sigma_a, mu_b, sigma_b):
    """
    :param mu_a: mean vector of batch_size x dim
    :param sigma_a: standard deviation of batch_size x 1
    :param mu_b:
    :param sigma_b:
    :return: computes KL-Divergence between 2 spherical Guassian (a||b)
    """
    d = mu_a.shape[1]
    sigma_p_inv = 1.0 / sigma_b  # because diagonal
    tra = d * sigma_a * sigma_p_inv
    quadr = sigma_p_inv * torch.pow(mu_b - mu_a, 2).sum(1, keepdim=True)
    log_det = - d * torch.log(sigma_a * sigma_p_inv)
    res = 0.5 * (tra + quadr - d + log_det)
    return res


def restore_model(vocab_size, restore_name):
    checkpoint_dir = os.path.join('weights', restore_name)
    checkpoint_fns = os.listdir(checkpoint_dir)
    max_checkpoint_epoch, latest_checkpoint_idx = -1, -1
    for cidx, checkpoint_fn in enumerate(checkpoint_fns):
        checkpoint_epoch = int(checkpoint_fn[-5])
        max_checkpoint_epoch = max(max_checkpoint_epoch, checkpoint_epoch)
        if checkpoint_epoch == max_checkpoint_epoch:
            latest_checkpoint_idx = cidx
    latest_checkpoint_fn = os.path.join(checkpoint_dir, checkpoint_fns[latest_checkpoint_idx])
    print('Loading model from {}'.format(latest_checkpoint_fn))
    checkpoint_state = torch.load(latest_checkpoint_fn)
    print('Previous checkpoint at epoch={}...'.format(max_checkpoint_epoch))
    for k, v in checkpoint_state['losses'].items():
        print('{}={}'.format(k, v))
    args = argparse.ArgumentParser()
    for k, v in checkpoint_state['args'].items():
        print('{}={}'.format(k, v))
        setattr(args, k, v)

    vae_model = VAE(args, vocab_size)
    vae_model.load_state_dict(checkpoint_state['model_state_dict'])

    return args, vae_model, checkpoint_state['optimizer_state_dict']