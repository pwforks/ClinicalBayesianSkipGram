import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from compute_utils import compute_kl, mask_2D


class VAE(nn.Module):
    def __init__(self, args, token_vocab_size, section_vocab_size):
        super(VAE, self).__init__()
        self.device = args.device
        self.encoder = Encoder(args, token_vocab_size, section_vocab_size)
        self.margin = args.hinge_loss_margin or 1.0

    def forward(self, center_ids, section_ids, context_ids, neg_context_ids, num_contexts):
        """
        :param center_ids: batch_size
        :param section_ids: batch_size
        :param context_ids: batch_size, 2 * context_window
        :param neg_context_ids: batch_size, 2 * context_window, num negative samples
        :param num_contexts: batch_size (how many context words for each center id - necessary for masking padding)
        :return: cost components: KL-Divergence (q(z|w,c) || p(z|w)) and max margin (reconstruction error)
        """
        # Mask padded context ids
        batch_size, num_context_ids = context_ids.size()
        ns = neg_context_ids.shape[-1]
        mask_size = torch.Size([batch_size, num_context_ids])
        mask = mask_2D(mask_size, num_contexts).to(self.device)

        # Compute center words
        mu_center, sigma_center = self.encoder(center_ids, section_ids)
        mu_center_tiled = mu_center.unsqueeze(1).repeat(1, num_context_ids, 1)
        sigma_center_tiled = sigma_center.unsqueeze(1).repeat(1, num_context_ids, 1)
        mu_center_flat = mu_center_tiled.view(batch_size * num_context_ids, -1)
        sigma_center_flat = sigma_center_tiled.view(batch_size * num_context_ids, -1)
        mu_center_tiled_ns = mu_center_tiled.repeat(1, ns, 1)
        sigma_center_tiled_ns = sigma_center_tiled.repeat(1, ns, 1)
        mu_center_flat_ns = mu_center_tiled_ns.view(batch_size * num_context_ids * ns, -1)
        sigma_center_flat_ns = sigma_center_tiled_ns.view(batch_size * num_context_ids * ns, -1)

        # Tile section ids for positive and negative samples
        section_ids_tiled = section_ids.unsqueeze(-1).repeat(1, num_context_ids)
        section_ids_tiled_ns = section_ids_tiled.unsqueeze(-1).repeat(1, 1, ns)

        # Compute positive and negative encoded samples
        mu_pos_context, sigma_pos_context = self.encoder(context_ids, section_ids_tiled)
        mu_neg_context, sigma_neg_context = self.encoder(neg_context_ids, section_ids_tiled_ns)

        # Flatten positive context
        mu_pos_context_flat = mu_pos_context.view(batch_size * num_context_ids, -1)
        sigma_pos_context_flat = sigma_pos_context.view(batch_size * num_context_ids, -1)

        # Flatten negative context
        mu_neg_context_flat = mu_neg_context.view(batch_size * num_context_ids * ns, -1)
        sigma_neg_context_flat = sigma_neg_context.view(batch_size * num_context_ids * ns, -1)

        # Compute KL-divergence between center words and negative and reshape
        kl_pos_flat = compute_kl(mu_center_flat, sigma_center_flat, mu_pos_context_flat, sigma_pos_context_flat)
        kl_neg_flat = compute_kl(mu_center_flat_ns, sigma_center_flat_ns, mu_neg_context_flat, sigma_neg_context_flat)
        kl_pos = kl_pos_flat.view(batch_size, num_context_ids)
        kl_neg = kl_neg_flat.view(batch_size, num_context_ids, ns)

        hinge_loss = (kl_pos - kl_neg.min(-1)[0] + self.margin).clamp_min_(0)
        hinge_loss.masked_fill(mask, 0)
        hinge_loss = hinge_loss.sum(1)
        return hinge_loss.mean()
