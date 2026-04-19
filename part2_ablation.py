"""Run ablations A1-A4.
A1: Unidirectional LSTM
A2: No dropout
A3: Random embedding initialisation
A4: Softmax output instead of CRF (handled by part2_train_ner.py)
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, EMB_DIR, OUTPUT_DIR, NER_TAGS, DEVICE
from models.bilstm_tagger import BiLSTMTagger
from utils.io_utils import set_seed
from utils.metrics import ner_entity_report


class ConllDataset(Dataset):
    def __init__(self, path, word2idx, label2idx):
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            tokens, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        self.samples.append((tokens, labels))
                    tokens, labels = [], []
                    continue
                tok, lab = line.split('\t')
                tokens.append(word2idx.get(tok, word2idx['<UNK>']))
                labels.append(label2idx[lab])
        if tokens:
            self.samples.append((tokens, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = max(lengths).item()
    inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, (tokens, labs) in enumerate(batch):
        inputs[i, :len(tokens)] = torch.tensor(tokens)
        labels[i, :len(labs)] = torch.tensor(labs)
    return inputs, labels, lengths


def eval_model(model, loader, idx2label, device):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            seqs = model.decode(x, lengths)
            for i, seq in enumerate(seqs):
                gold.append([idx2label[t] for t in y[i, :len(seq)].tolist()])
                pred.append([idx2label[t] for t in seq])
    return ner_entity_report(gold, pred)['overall']


def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu'
    with open(EMB_DIR / 'word2idx.json', 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    embeddings = np.load(EMB_DIR / 'embeddings_w2v.npy')
    label2idx = {t: i for i, t in enumerate(NER_TAGS)}
    idx2label = {i: t for t, i in label2idx.items()}
    train_ds = ConllDataset(DATA_DIR / 'ner_train.conll', word2idx, label2idx)
    test_ds = ConllDataset(DATA_DIR / 'ner_test.conll', word2idx, label2idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

    configs = {
        'A1_unidirectional': dict(bidirectional=False, dropout=0.5, embeddings=embeddings, use_crf=True),
        'A2_no_dropout': dict(bidirectional=True, dropout=0.0, embeddings=embeddings, use_crf=True),
        'A3_random_init': dict(bidirectional=True, dropout=0.5, embeddings=None, use_crf=True),
        'A4_softmax_output': dict(bidirectional=True, dropout=0.5, embeddings=embeddings, use_crf=False),
    }
    results = {}
    for name, cfg in configs.items():
        model = BiLSTMTagger(len(word2idx), embeddings.shape[1], 128, len(NER_TAGS),
                             embeddings=cfg['embeddings'], freeze_embeddings=False,
                             use_crf=cfg['use_crf'], dropout=cfg['dropout'], bidirectional=cfg['bidirectional']).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(3):
            model.train()
            for x, y, lengths in train_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                opt.zero_grad()
                loss = model(x, lengths, y)
                loss.backward()
                opt.step()
        results[name] = eval_model(model, test_loader, idx2label, device)

    with open(OUTPUT_DIR / 'ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('Saved ablation study results.')


if __name__ == '__main__':
    main()
