from collections import defaultdict
import os
import re
import string
import sys

import argparse
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
import torch

from compute_utils import compute_kl
from model_utils import restore_model, tensor_to_np
from vae import VAE


PUNCTUATION = '-' + string.punctuation.replace( '-', '')
STOPWORDS = set(stopwords.words('english'))
sys.path.insert(0, '/home/ga2530/ClinicalBayesianSkipGram/preprocess/')


def get_known_ids(vocab, tokens):
    return list(filter(lambda id: id > -1, list(map(lambda tok: vocab.get_id(tok), tokens.split(' ')))))


def point_similarity(model, vocab, tokens_a, tokens_b):
    ids_a = get_known_ids(vocab, tokens_a)
    ids_b = get_known_ids(vocab, tokens_b)

    if len(ids_a) == 0 or len(ids_b) == 0:
        return 0.0

    embeddings = tensor_to_np(model.embeddings_mu.weight)

    rep_a = embeddings[ids_a, :].mean(0)
    rep_b = embeddings[ids_b, :].mean(0)
    sim = 1.0 - cosine(rep_a, rep_b)
    return sim


UMLS_BLACKLIST = set(['unidentified', 'otherwise', 'specified', 'nos'])  # UMLS concepts annoyingly includes these terms
TOKEN_BLACKLIST = set(string.punctuation).union(STOPWORDS).union(UMLS_BLACKLIST)


def tokenize_str(str, unique_only=False):
    filtered = re.sub(r'_%#\S+#%_', '', str).lower()
    filtered = re.sub(r'\b(\d+)\b', ' ', filtered)
    lead_trail_punc_regex = r'(\s[{}]+)|([{}]+\s)'.format(PUNCTUATION, PUNCTUATION)
    filtered = re.sub(lead_trail_punc_regex, ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    tokens = word_tokenize(filtered)
    tokens = [t for t in tokens if t not in TOKEN_BLACKLIST]
    tokens = [t.strip(string.punctuation) for t in tokens]
    if unique_only:
        tokens = list(set(tokens))
    return tokens


def preprocess_minnesota_dataset(window=5):
    in_fp = '../eval_data/minnesota/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt'
    out_fp = '../eval_data/minnesota/preprocessed_dataset_window_{}.csv'.format(window)
    # cols = ['sf', 'target_lf', 'sf_rep', 'start_idx', 'end_idx', 'section', 'context']
    df = pd.read_csv(in_fp, sep='|')
    df.dropna(subset=['sf', 'target_lf', 'context'], inplace=True)

    # Tokenize
    sf_occurrences = []  # When multiple of SF in context, keep track of which one the label is for
    tokenized_contexts = []

    valid_rows = []
    print('Filtering out invalid rows...')
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        sf_idxs = [m.start() for m in re.finditer(r'\b({})\b'.format(row['sf']), row['context'])]
        target_start_idx = int(row['start_idx'])

        valid = ' ' in row['context'][target_start_idx + len(row['sf']): target_start_idx + len(row['sf']) + 2]

        sf_occurrence_ct = np.where(np.array(sf_idxs) == target_start_idx)[0]
        if not valid or len(sf_occurrence_ct) == 0:
            # print('SF not present in context for row={}'.format(row_idx))
            valid_rows.append(False)
            tokenized_contexts.append(None)
            sf_occurrences.append(None)
        else:
            assert len(sf_occurrence_ct) == 1
            valid_rows.append(True)
            sf_occurrences.append(sf_occurrence_ct[0])
            tokenized_contexts.append(' '.join(tokenize_str(row['context'])))

    df['valid'] = valid_rows
    df['tokenized_context'] = tokenized_contexts
    df['sf_occurrences'] = sf_occurrences
    prev_n = df.shape[0]
    df = df[df['valid']]
    n = df.shape[0]
    print('Dropping {} rows because we couldn\'t find SF in the context.'.format(prev_n - n))

    trimmed_contexts = []
    print('Tokenizing and extracting context windows...')
    valid = []
    for row_idx, row in df.iterrows():
        row = row.to_dict()
        sf_label_order = int(row['sf_occurrences'])
        tokens = row['tokenized_context'].split()
        sf_idxs = np.where(np.array(tokens) == row['sf'].lower())[0]

        if len(sf_idxs) == 0 or sf_label_order >= len(sf_idxs):
            valid.append(False)
            trimmed_contexts.append(None)
        else:
            valid.append(True)
            sf_idx = sf_idxs[sf_label_order]
            start_idx = max(0, sf_idx - window)
            end_idx = min(sf_idx + window + 1, len(tokens))
            tc = tokens[start_idx:sf_idx] + tokens[sf_idx + 1: end_idx]
            trimmed_contexts.append(' '.join(tc))

    df['trimmed_tokens'] = trimmed_contexts
    df['valid'] = valid
    prev_n = df.shape[0]
    df = df[df['valid']]
    df.drop(columns='valid', inplace=True)
    print('Issues parsing into tokens for {} out of {} remaining examples'.format(prev_n - df.shape[0], prev_n))
    print('Finished preprocessing dataset.  Now saving it to {}'.format(out_fp))
    df.to_csv(out_fp, index=False)


def evaluate_acronyms(prev_args, model, vocab, mini=False):
    data_fp = '../eval_data/minnesota/preprocessed_dataset_window_{}.csv'.format(prev_args.window)
    if not os.path.exists(data_fp):
        print('Preprocessing Dataset...')
        preprocess_minnesota_dataset(prev_args.window)

    df = pd.read_csv(data_fp)

    if mini:
        df = df.sample(100, random_state=1992).reset_index(drop=True)

    context_ids = np.zeros([df.shape[0], prev_args.window * 2], dtype=int)
    center_ids = np.zeros([df.shape[0], ], dtype=int)
    ct = 0
    for row_idx, row in df.iterrows():
        assert row_idx == ct
        ct += 1
        row = row.to_dict()

        center_id = vocab.get_id(row['sf'].lower())
        center_ids[row_idx] = center_id
        context_id_seq = [vocab.get_id(token.lower()) for token in row['trimmed_tokens'].split()]
        context_ids[row_idx, :len(context_id_seq)] = context_id_seq

    center_id_tens = torch.LongTensor(center_ids).clamp_min_(0).to(prev_args.device)
    context_id_tens = torch.LongTensor(context_ids).clamp_min_(0).to(prev_args.device)
    # TODO add proper masking
    mask = torch.BoolTensor(torch.Size([center_id_tens.shape[0], context_id_tens.shape[-1]])).to(prev_args.device)
    mask.fill_(0)
    with torch.no_grad():
        z_mus, z_sigmas = model.encoder(center_id_tens, context_id_tens, mask)

    sense_fp = '../eval_data/minnesota/sense_inventory_ii'
    sense_df = pd.read_csv(sense_fp, sep='|')

    lfs = sense_df['LF'].unique().tolist()
    lf_sf_map = {}
    for row_idx, row in sense_df.iterrows():
        row = row.to_dict()
        lf_sf_map[row['LF']] = row['SF']
    tokenized_lfs = list(map(lambda t: tokenize_str(t, unique_only=True), lfs))
    tokenized_ids = [list(map(vocab.get_id, tokens)) for tokens in tokenized_lfs]
    lf_priors = list(map(lambda x: model._compute_priors(torch.LongTensor(x).clamp_min_(0).to(prev_args.device)),
                         tokenized_ids))

    # for each SF, create matrix of LFs and priors
    sf_lf_prior_map = {}
    for lf, priors in zip(lfs, lf_priors):
        sf = lf_sf_map[lf]

        if sf not in sf_lf_prior_map:
            sf_lf_prior_map[sf] = {
                'lf': [],
                'lf_mu': [],
                'lf_sigma': []
            }

        sf_lf_prior_map[sf]['lf'].append(lf)
        # Representation of phrases is just average of words (...for now)
        sf_lf_prior_map[sf]['lf_mu'].append(priors[0].mean(axis=0).unsqueeze(0))
        sf_lf_prior_map[sf]['lf_sigma'].append(priors[1].mean(axis=0).unsqueeze(0))

    for sf in sf_lf_prior_map.keys():
        sf_lf_prior_map[sf]['lf_mu'] = torch.cat(sf_lf_prior_map[sf]['lf_mu'], axis=0)
        sf_lf_prior_map[sf]['lf_sigma'] = torch.cat(sf_lf_prior_map[sf]['lf_sigma'], axis=0)

    num_correct, target_kld, median_kld = 0, 0.0, 0.0
    for row_idx, row in df.iterrows():
        sf = row['sf']
        z_mu, z_sigma = z_mus[row_idx].unsqueeze(0), z_sigmas[row_idx].unsqueeze(0)
        lf_obj = sf_lf_prior_map[sf]
        lf_map, prior_mus, prior_sigmas = lf_obj['lf'], lf_obj['lf_mu'], lf_obj['lf_sigma']
        num_lf = prior_mus.shape[0]
        divergences = compute_kl(z_mu.repeat(num_lf, 1), z_sigma.repeat(num_lf, 1), prior_mus, prior_sigmas)

        closest_lf_idx = divergences.squeeze(-1).argmin().item()
        predicted_lf = lf_map[closest_lf_idx]
        target_lf = row['target_lf']

        try:
            target_lf_idx = lf_map.index(target_lf)
        except ValueError:
            target_lf_idx = -1
            for idx, lf in enumerate(lf_map):
                for lf_chunk in lf.split(';'):
                    if target_lf == lf_chunk:
                        target_lf_idx = idx
                        break
        target_kld += divergences[target_lf_idx].item()
        median_kld += divergences.median().item()

        if predicted_lf == target_lf:
            num_correct += 1

    N = float(df.shape[0])
    print('Statistics from {} examples'.format(int(N)))
    print('Minnesota Acronym Accuracy={}.  KLD(SF Posterior||Target LF)={}. Median KLD(SF Posterior || LF_(i..n)={}'.
        format(num_correct / N, target_kld / N, median_kld / N))


def evaluate_mimic_acronyms(prev_args, model, vocab):
    print('Evaluating context-dependent representations via acronym disambiguation...')
    label_df = pd.read_csv('../eval_data/mimic_acronym_expansion_labels.csv')
    expansion_df = pd.read_csv('../eval_data/acronym_expansions.csv')
    expansion_df = expansion_df[expansion_df['lf_count'] > 0]

    target_window_size = prev_args.window

    sf_lf_map = defaultdict(set)
    for row_idx, row in expansion_df.iterrows():
        row = row.to_dict()
        sf_lf_map[row['sf']].add(row['lf'])

    contexts = label_df['context'].tolist()
    context_ids = []
    for row_idx, context in enumerate(contexts):
        sc = context.split()
        for cidx in range(len(sc)):
            if sc[cidx] == 'TARGETWORD':
                break

        left_idx = max(0, cidx - target_window_size)
        right_idx = min(cidx + target_window_size + 1, len(sc))
        context_tokens = sc[left_idx:cidx] + sc[cidx + 1: right_idx]
        context_ids.append([
            vocab.get_id(token.lower()) for token in context_tokens
        ])

    prior_kls = []
    posterior_kls = []

    for row_idx, row in label_df.iterrows():
        row = row.to_dict()
        sf, target_lfs = row['sf'], row['lf']
        context_id_seq = context_ids[row_idx]
        center_id = vocab.get_id(sf.lower())

        center_id_tens = torch.LongTensor([center_id]).clamp_min_(0).to(prev_args.device)
        context_id_tens = torch.LongTensor(context_id_seq).unsqueeze(0).clamp_min_(0).to(prev_args.device)

        mask = torch.BoolTensor(torch.Size([1, len(context_id_seq)])).to(prev_args.device)
        mask.fill_(0)

        p_mu, p_sigma = model._compute_priors(center_id_tens)
        min_prior_kl, min_posterior_kl = float('inf'), float('inf')

        with torch.no_grad():
            z_mu, z_sigma = model.encoder(center_id_tens, context_id_tens, mask)

        for target_expansion in target_lfs.split('|'):
            lf_ids = [vocab.get_id(token.lower()) for token in target_expansion.split()]
            lf_tens = torch.LongTensor(lf_ids).clamp_min_(0).to(prev_args.device)

            lf_mu, lf_sigma = model._compute_priors(lf_tens)

            avg_lf_mu = lf_mu.mean(axis=0)
            avg_lf_sigma = lf_sigma.mean(axis=0)

            prior_kl = compute_kl(p_mu, p_sigma, avg_lf_mu, avg_lf_sigma).item()
            posterior_kl = compute_kl(z_mu, z_sigma, avg_lf_mu, avg_lf_sigma).item()

            min_prior_kl = min(prior_kl, min_prior_kl)
            min_posterior_kl = min(posterior_kl, min_posterior_kl)
        prior_kls.append(min_prior_kl)
        posterior_kls.append(min_posterior_kl)

    avg_prior_kl = sum(prior_kls) / float(len(prior_kls))
    avg_posterior_distances = sum(posterior_kls) / float(len(posterior_kls))
    print('Avg Prior KLD={}. Avg Posterior KLD={}'.format(avg_prior_kl, avg_posterior_distances))


def evaluate_word_similarity(model, vocab):
    umnrs = {
        'name': 'UMNRS',
        'file': '../eval_data/UMNSRS_relatedness.csv',
        'label': 'Mean',
        't1': 'Term1',
        't2': 'Term2',
    }

    mayo = {
        'name': 'MayoSRS',
        'file': '../eval_data/MayoSRS.csv',
        'label': 'Mean',
        't1': 'TERM1',
        't2': 'TERM2',
    }

    sim_datasets = [umnrs, mayo]

    for didx, sim_dataset in enumerate(sim_datasets):
        word_sim_df = pd.read_csv(sim_dataset['file'])
        human_scores = word_sim_df[sim_dataset['label']].tolist()
        known_model_relatedness, known_human_scores = [], []
        for row_idx, row in word_sim_df.iterrows():
            row = row.to_dict()
            t1 = row[sim_dataset['t1']].lower()
            t2 = row[sim_dataset['t2']].lower()
            sim = point_similarity(model, vocab, t1, t2)
            if not sim == 0.0:  # means both terms are OOV
                known_human_scores.append(human_scores[row_idx])
                known_model_relatedness.append(sim)
        pear_corr, _ = pearsonr(known_model_relatedness, known_human_scores)
        spear_corr, _ = spearmanr(known_model_relatedness, known_human_scores)
        sim_datasets[didx]['pearson_correlation'] = pear_corr
        sim_datasets[didx]['spearman_correlation'] = spear_corr
        print('{} Evaluation\n\tWord Similarity --> Pearson Corr.={}, Spearman Corr.={}'.format(
            sim_dataset['name'], pear_corr, spear_corr))
    return sim_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Bayesian Skip Gram Model')

    # Functional Arguments
    parser.add_argument('-cpu', action='store_true', default=False)
    parser.add_argument('--eval_fp', default='../preprocess/data/')
    parser.add_argument('--experiment', default='debug', help='Save path in weights/ for experiment.')
    parser.add_argument('-acronym_mini', default=False, action='store_true')

    args = parser.parse_args()

    device_str = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    args.device = torch.device(device_str)
    print('Evaluating on {}...'.format(device_str))

    prev_args, vae_model, vocab, optimizer_state = restore_model(args.experiment)

    # Make sure it's NOT calculating gradients
    model = vae_model.to(args.device)
    model.eval()  # just sets .requires_grad = False

    print('\nEvaluations...')
    word_sim_results = evaluate_word_similarity(model, vocab)
    evaluate_acronyms(prev_args, model, vocab, mini=args.acronym_mini)
    evaluate_mimic_acronyms(prev_args, model, vocab)
