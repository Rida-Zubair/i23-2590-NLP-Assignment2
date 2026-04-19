from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.context_embeddings = nn.Embedding(vocab_size, emb_dim)
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, center_words, pos_context_words, neg_context_words):
        v = self.center_embeddings(center_words)                 # [B, D]
        u_pos = self.context_embeddings(pos_context_words)      # [B, D]
        u_neg = self.context_embeddings(neg_context_words)      # [B, K, D]

        pos_score = torch.sum(v * u_pos, dim=1)                 # [B]
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2) # [B, K]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        return 0.5 * (self.center_embeddings.weight.data + self.context_embeddings.weight.data)
