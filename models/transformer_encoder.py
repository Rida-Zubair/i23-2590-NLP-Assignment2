import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk: int):
        super().__init__()
        self.scale = math.sqrt(dk)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        return output, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_k=32, d_v=32, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.q_projs = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(num_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(num_heads)])
        self.out_proj = nn.Linear(num_heads * d_v, d_model)
        self.attn = ScaledDotProductAttention(dk=d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        head_outputs = []
        all_weights = []
        for i in range(self.num_heads):
            q = self.q_projs[i](x)
            k = self.k_projs[i](x)
            v = self.v_projs[i](x)
            out, weights = self.attn(q, k, v, mask=mask)
            head_outputs.append(out)
            all_weights.append(weights)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.dropout(self.out_proj(concat))
        weights = torch.stack(all_weights, dim=1)
        return output, weights


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.mha(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x, attn_weights


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, d_model=128, num_heads=4, d_ff=512, num_layers=4, num_classes=5,
                 max_len=257, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def make_mask(self, input_ids):
        batch_size, seq_len = input_ids.size()
        cls_pad = torch.ones(batch_size, 1, device=input_ids.device, dtype=torch.bool)
        token_mask = input_ids != self.pad_idx
        full_mask = torch.cat([cls_pad, token_mask], dim=1)
        return full_mask.unsqueeze(1).expand(-1, seq_len + 1, -1)

    def forward(self, input_ids):
        x = self.token_embeddings(input_ids)
        cls = self.cls_token.expand(input_ids.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos_encoding(x))
        mask = self.make_mask(input_ids)
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attention_maps.append(attn)
        logits = self.classifier(x[:, 0])
        return logits, attention_maps
