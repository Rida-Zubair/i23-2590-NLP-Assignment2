import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, EMB_DIR, MODEL_DIR, OUTPUT_DIR, PLOT_DIR, POS_TAGS, DEVICE
from models.bilstm_tagger import BiLSTMTagger
from utils.io_utils import set_seed
from utils.metrics import accuracy_score, macro_f1_score, confusion_matrix
from utils.plot_utils import plot_curve, plot_confusion_matrix


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
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (tokens, labs) in enumerate(batch):
        inputs[i, :len(tokens)] = torch.tensor(tokens)
        labels[i, :len(labs)] = torch.tensor(labs)
    return inputs, labels, lengths


def evaluate(model, loader, device):
    model.eval()
    gold_all, pred_all = [], []
    with torch.no_grad():
        for inputs, labels, lengths in loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            preds = model.decode(inputs, lengths)
            for i, seq in enumerate(preds):
                gold = labels[i, :len(seq)].tolist()
                gold_all.extend(gold)
                pred_all.extend(seq)
    acc = accuracy_score(gold_all, pred_all)
    f1 = macro_f1_score(gold_all, pred_all, num_classes=len(POS_TAGS))
    cm = confusion_matrix(gold_all, pred_all, num_classes=len(POS_TAGS))
    return acc, f1, cm


def train_mode(freeze_embeddings=False):
    set_seed()
    device = 'cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu'
    with open(EMB_DIR / 'word2idx.json', 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    embeddings = np.load(EMB_DIR / 'embeddings_w2v.npy')
    label2idx = {t: i for i, t in enumerate(POS_TAGS)}

    train_ds = ConllDataset(DATA_DIR / 'pos_train.conll', word2idx, label2idx)
    val_ds = ConllDataset(DATA_DIR / 'pos_val.conll', word2idx, label2idx)
    test_ds = ConllDataset(DATA_DIR / 'pos_test.conll', word2idx, label2idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

    model = BiLSTMTagger(len(word2idx), embeddings.shape[1], 128, len(POS_TAGS), embeddings=embeddings,
                         freeze_embeddings=freeze_embeddings, use_crf=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1, patience, wait = -1, 5, 0
    train_losses, val_losses = [], []
    for epoch in range(20):
        model.train()
        total = 0.0
        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            opt.zero_grad()
            loss = model(inputs, lengths, labels)
            loss.backward()
            opt.step()
            total += loss.item()
        train_losses.append(total / max(len(train_loader), 1))

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
                vloss += model(inputs, lengths, labels).item()
        val_losses.append(vloss / max(len(val_loader), 1))
        _, val_f1, _ = evaluate(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), MODEL_DIR / f"bilstm_pos_{'frozen' if freeze_embeddings else 'finetuned'}.pt")
        else:
            wait += 1
            if wait >= patience:
                break

    plot_curve(train_losses, f'POS Loss ({"frozen" if freeze_embeddings else "finetuned"})', 'Loss',
               PLOT_DIR / f'pos_loss_{"frozen" if freeze_embeddings else "finetuned"}.png', val_losses)

    model.load_state_dict(torch.load(MODEL_DIR / f"bilstm_pos_{'frozen' if freeze_embeddings else 'finetuned'}.pt", map_location=device))
    acc, f1, cm = evaluate(model, test_loader, device)
    plot_confusion_matrix(cm, POS_TAGS, 'POS Confusion Matrix', PLOT_DIR / f'pos_confmat_{"frozen" if freeze_embeddings else "finetuned"}.png')
    return {'accuracy': acc, 'macro_f1': f1}


def main():
    frozen = train_mode(True)
    finetuned = train_mode(False)
    with open(OUTPUT_DIR / 'pos_results.json', 'w', encoding='utf-8') as f:
        json.dump({'frozen': frozen, 'finetuned': finetuned}, f, indent=2)
    print('Saved POS models, curves, confusion matrix, and results.')


if __name__ == '__main__':
    main()
