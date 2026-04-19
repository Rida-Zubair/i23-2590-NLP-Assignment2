from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.crf import CRF


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_labels, embeddings=None, freeze_embeddings=False,
                 use_crf=False, num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(embeddings, dtype=torch.float))
        self.embedding.weight.requires_grad = not freeze_embeddings

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_labels)
        self.use_crf = use_crf
        self.crf = CRF(num_labels) if use_crf else None

    def forward(self, input_ids, lengths, tags: Optional[torch.Tensor] = None):
        mask = input_ids != 0
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.classifier(out)

        if tags is not None:
            if self.use_crf:
                return self.crf(logits, tags, mask[:, :logits.size(1)])
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fn(logits.view(-1, logits.size(-1)), tags[:, :logits.size(1)].reshape(-1))
        return logits

    def decode(self, input_ids, lengths):
        mask = input_ids != 0
        logits = self.forward(input_ids, lengths, tags=None)
        if self.use_crf:
            return self.crf.decode(logits, mask[:, :logits.size(1)])
        preds = logits.argmax(dim=-1)
        output = []
        for i, length in enumerate(lengths.tolist()):
            output.append(preds[i, :length].tolist())
        return output
