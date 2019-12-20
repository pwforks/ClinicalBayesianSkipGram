import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data


class Encoder(nn.Module):
    def __init__(self, args, token_vocab_size, section_vocab_size):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.f = nn.Linear(args.encoder_input_dim * 2, args.encoder_hidden_dim, bias=True)
        self.u = nn.Linear(args.encoder_hidden_dim, args.latent_dim, bias=True)
        self.v = nn.Linear(args.encoder_hidden_dim, 1, bias=True)

        self.token_embeddings = nn.Embedding(token_vocab_size, args.encoder_input_dim, padding_idx=0)
        self.section_embeddings = nn.Embedding(section_vocab_size, args.encoder_input_dim, padding_idx=0)
        
    def forward(self, center_ids, section_ids):
        """
        :param center_ids: LongTensor of batch_size
        :param context_ids: LongTensor of batch_size
        :param mask: BoolTensor of batch_size x 2 * context_window (which context_ids are just the padding idx)
        :return: mu (batch_size, latent_dim), logvar (batch_size, 1)
        """
        center_embedding = self.token_embeddings(center_ids)
        section_embedding = self.section_embeddings(section_ids)

        merged_embeds = self.dropout(torch.cat([center_embedding, section_embedding], dim=-1))

        h = self.dropout(F.relu(self.f(merged_embeds)))
        var_clamped = self.v(h).exp().clamp_min(1.0)
        return self.u(h), var_clamped
