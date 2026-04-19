import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CLEANED_PATH, EMB_DIR, OUTPUT_DIR, PLOT_DIR, VOCAB_SIZE, W2V_DIM, WINDOW_SIZE, NUM_NEGATIVE, LR_W2V, BATCH_SIZE_W2V, EPOCHS_W2V, DEVICE
from models.word2vec import SkipGramNegSampling
from utils.io_utils import read_lines, simple_tokenize, build_vocab, save_json, set_seed
from utils.plot_utils import plot_curve


class SkipGramDataset(Dataset):
    def __init__(self, tokenized_docs, word2idx, window_size=5):
        self.pairs = []
        for doc in tokenized_docs:
            ids = [word2idx.get(tok, word2idx['<UNK>']) for tok in doc]
            for i, center in enumerate(ids):
                left = max(0, i - window_size)
                right = min(len(ids), i + window_size + 1)
                for j in range(left, right):
                    if i != j:
                        self.pairs.append((center, ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def build_noise_distribution(tokenized_docs, word2idx):
    counter = Counter(tok for doc in tokenized_docs for tok in doc)
    freqs = np.zeros(len(word2idx), dtype=np.float64)
    for word, idx in word2idx.items():
        freqs[idx] = counter.get(word, 1)
    probs = freqs ** 0.75
    probs = probs / probs.sum()
    return probs


def collate_fn(batch, noise_probs, num_negative, vocab_size):
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    negatives = np.random.choice(vocab_size, size=(len(batch), num_negative), p=noise_probs)
    negatives = torch.tensor(negatives, dtype=torch.long)
    return centers, contexts, negatives


def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu'
    docs = read_lines(CLEANED_PATH)
    tokenized_docs = [simple_tokenize(doc) for doc in docs]
    word2idx, idx2word, _ = build_vocab(tokenized_docs, max_vocab=VOCAB_SIZE)
    noise_probs = build_noise_distribution(tokenized_docs, word2idx)

    dataset = SkipGramDataset(tokenized_docs, word2idx, window_size=WINDOW_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_W2V,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, noise_probs, NUM_NEGATIVE, len(word2idx)),
    )

    model = SkipGramNegSampling(vocab_size=len(word2idx), emb_dim=W2V_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_W2V)

    losses = []
    for epoch in range(EPOCHS_W2V):
        running = 0.0
        for step, (centers, pos_ctx, neg_ctx) in enumerate(loader, 1):
            centers = centers.to(device)
            pos_ctx = pos_ctx.to(device)
            neg_ctx = neg_ctx.to(device)
            optimizer.zero_grad()
            loss = model(centers, pos_ctx, neg_ctx)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 100 == 0:
                print(f'Epoch {epoch+1} Step {step} Loss {running / step:.4f}')
        losses.append(running / max(len(loader), 1))

    plot_curve(losses, 'Word2Vec Training Loss', 'Loss', PLOT_DIR / 'word2vec_loss.png')
    embeddings = model.get_embeddings().cpu().numpy()
    np.save(EMB_DIR / 'embeddings_w2v.npy', embeddings)
    save_json(word2idx, EMB_DIR / 'word2idx.json')
    torch.save(model.state_dict(), OUTPUT_DIR / 'skipgram_w2v.pt')
    print('Saved embeddings_w2v.npy and training plot.')


if __name__ == '__main__':
    main()
