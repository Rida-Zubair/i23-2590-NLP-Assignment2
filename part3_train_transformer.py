import json
import math

import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_DIR, EMB_DIR, MODEL_DIR, OUTPUT_DIR, PLOT_DIR, DEVICE, CLS_TOKEN
from models.transformer_encoder import TransformerClassifier
from utils.io_utils import simple_tokenize, set_seed
from utils.metrics import accuracy_score, macro_f1_score, confusion_matrix
from utils.plot_utils import plot_curve, plot_confusion_matrix, plot_attention_heatmap


class TopicDataset(Dataset):
    def __init__(self, items, word2idx, max_len=256):
        self.items = items
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        tokens = simple_tokenize(item['text'])[:self.max_len]
        ids = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(item['label'], dtype=torch.long), tokens


def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    toks = [b[2] for b in batch]
    return ids, y, toks


def cosine_with_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loader, device):
    model.eval()
    gold, pred = [], []
    batches = []
    with torch.no_grad():
        for ids, y, toks in loader:
            ids, y = ids.to(device), y.to(device)
            logits, attn = model(ids)
            p = logits.argmax(dim=-1)
            gold.extend(y.tolist())
            pred.extend(p.tolist())
            batches.append((ids.cpu(), y.cpu(), toks, attn))
    return {
        'accuracy': accuracy_score(gold, pred),
        'macro_f1': macro_f1_score(gold, pred, 5),
        'cm': confusion_matrix(gold, pred, 5),
        'batches': batches,
    }


def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu'
    with open(EMB_DIR / 'word2idx.json', 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    with open(DATA_DIR / 'topic_classification_splits.json', 'r', encoding='utf-8') as f:
        splits = json.load(f)

    train_loader = DataLoader(TopicDataset(splits['train'], word2idx), batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(TopicDataset(splits['val'], word2idx), batch_size=16, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(TopicDataset(splits['test'], word2idx), batch_size=16, shuffle=False, collate_fn=collate)

    model = TransformerClassifier(vocab_size=len(word2idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    total_steps = 20 * max(1, len(train_loader))
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_with_warmup(s, warmup_steps, total_steps))
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_f1 = -1
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for ids, y, _ in train_loader:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(ids)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += len(y)
        train_losses.append(total_loss / max(1, len(train_loader)))
        train_accs.append(correct / max(1, total))

        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for ids, y, _ in val_loader:
                ids, y = ids.to(device), y.to(device)
                logits, _ = model(ids)
                loss = criterion(logits, y)
                vloss += loss.item()
                vcorrect += (logits.argmax(dim=-1) == y).sum().item()
                vtotal += len(y)
        val_losses.append(vloss / max(1, len(val_loader)))
        val_accs.append(vcorrect / max(1, vtotal))
        val_eval = evaluate(model, val_loader, device)
        if val_eval['macro_f1'] > best_f1:
            best_f1 = val_eval['macro_f1']
            torch.save(model.state_dict(), MODEL_DIR / 'transformer_cls.pt')

    plot_curve(train_losses, 'Transformer Loss', 'Loss', PLOT_DIR / 'transformer_loss.png', val_losses)
    plot_curve(train_accs, 'Transformer Accuracy', 'Accuracy', PLOT_DIR / 'transformer_accuracy.png', val_accs)

    model.load_state_dict(torch.load(MODEL_DIR / 'transformer_cls.pt', map_location=device))
    test_eval = evaluate(model, test_loader, device)
    plot_confusion_matrix(test_eval['cm'], ['Politics','Sports','Economy','International','Health&Society'],
                          'Transformer Confusion Matrix', PLOT_DIR / 'transformer_confmat.png')

    # Save 3 example attention heatmaps from final layer, first 2 heads.
    saved = 0
    for ids, y, toks, attn_batches in test_eval['batches']:
        final_layer = attn_batches[-1]  # [B, H, T, T]
        for i in range(min(len(toks), 3 - saved)):
            seq_tokens = ['[CLS]'] + toks[i][:20]
            t = len(seq_tokens)
            for head in [0, 1]:
                weights = final_layer[i, head, :t, :t].cpu().numpy()
                plot_attention_heatmap(weights, seq_tokens, f'Article {saved+1} Head {head+1}',
                                       PLOT_DIR / f'attn_article{saved+1}_head{head+1}.png')
            saved += 1
            if saved >= 3:
                break
        if saved >= 3:
            break

    summary = {'accuracy': test_eval['accuracy'], 'macro_f1': test_eval['macro_f1']}
    with open(OUTPUT_DIR / 'transformer_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Saved transformer model, plots, and results.')


if __name__ == '__main__':
    main()
