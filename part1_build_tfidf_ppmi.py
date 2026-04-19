import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from config import CLEANED_PATH, METADATA_PATH, EMB_DIR, OUTPUT_DIR, PLOT_DIR, TOPIC_CATEGORIES, VOCAB_SIZE
from utils.io_utils import read_lines, simple_tokenize, build_vocab, save_json, load_metadata, infer_article_text, normalize_text
import matplotlib.pyplot as plt


def get_docs_and_topics():
    docs = read_lines(CLEANED_PATH)
    meta = load_metadata(METADATA_PATH) if METADATA_PATH.exists() else None
    topics = ['unknown'] * len(docs)
    if meta is None:
        return docs, topics

    if isinstance(meta, list) and len(meta) == len(docs):
        for i, item in enumerate(meta):
            text = infer_article_text(item)
            label = item.get('category') if isinstance(item, dict) else None
            if label is not None:
                topics[i] = str(label)
            elif text and normalize_text(text) == normalize_text(docs[i]):
                topics[i] = 'matched'
    return docs, topics


def compute_tfidf(doc_tokens, word2idx):
    vocab_size = len(word2idx)
    N = len(doc_tokens)
    tf = np.zeros((N, vocab_size), dtype=np.float32)
    df = np.zeros(vocab_size, dtype=np.int32)
    for d, tokens in enumerate(doc_tokens):
        counts = Counter(word2idx.get(t, word2idx['<UNK>']) for t in tokens)
        seen = set(counts.keys())
        for idx, c in counts.items():
            tf[d, idx] = c
        for idx in seen:
            df[idx] += 1
    idf = np.log(N / (1 + df + 1e-12))
    tfidf = tf * idf[None, :]
    return tfidf, idf


def top_words_per_topic(tfidf, topics, idx2word, top_k=10):
    groups = defaultdict(list)
    for i, t in enumerate(topics):
        groups[t].append(i)
    results = {}
    for topic, rows in groups.items():
        avg = tfidf[rows].mean(axis=0)
        best = np.argsort(-avg)[:top_k]
        results[topic] = [idx2word[i] for i in best if idx2word[i] not in ['<PAD>', '<UNK>', '<CLS>']]
    return results


def build_ppmi(doc_tokens, word2idx, window=5):
    vocab_size = len(word2idx)
    cooc = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for tokens in doc_tokens:
        ids = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
        for i, center in enumerate(ids):
            left = max(0, i - window)
            right = min(len(ids), i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = ids[j]
                cooc[center, context] += 1.0
    total = cooc.sum()
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    denom = row_sum @ col_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2((cooc * total + 1e-12) / (denom + 1e-12))
    ppmi = np.maximum(pmi, 0.0)
    ppmi[np.isnan(ppmi)] = 0.0
    ppmi[np.isinf(ppmi)] = 0.0
    return ppmi


def semantic_color_map(tokens):
    mapping = {}
    for tok in tokens:
        t = tok.lower()
        if t in {'pakistan', 'hukumat', 'wazir', 'siyasat'}:
            mapping[tok] = 'politics'
        elif t in {'cricket', 'match', 'team', 'player'}:
            mapping[tok] = 'sports'
        elif t in {'shehar', 'lahore', 'karachi', 'islamabad'}:
            mapping[tok] = 'geography'
        else:
            mapping[tok] = 'other'
    return mapping


def plot_tsne(ppmi, idx2word, counter):
    top_words = [w for w, _ in counter.most_common(200) if w in idx2word][:200]
    indices = [idx2word.index(w) for w in top_words]
    vectors = ppmi[indices]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(top_words)-1))
    points = tsne.fit_transform(vectors)
    classes = semantic_color_map(top_words)
    labels = sorted(set(classes.values()))
    plt.figure(figsize=(10, 8))
    for label in labels:
        mask = [classes[w] == label for w in top_words]
        coords = points[mask]
        plt.scatter(coords[:, 0], coords[:, 1], label=label, s=18)
    for i, word in enumerate(top_words[:50]):
        plt.text(points[i, 0], points[i, 1], word, fontsize=7)
    plt.title('t-SNE of top-200 PPMI token vectors')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ppmi_tsne.png')
    plt.close()


def nearest_neighbors(matrix, idx2word, query_words, top_k=5):
    sims = cosine_similarity(matrix)
    out = {}
    vocab_map = {w: i for i, w in enumerate(idx2word)}
    for q in query_words:
        if q not in vocab_map:
            out[q] = []
            continue
        q_idx = vocab_map[q]
        best = np.argsort(-sims[q_idx])[1: top_k + 1]
        out[q] = [idx2word[i] for i in best]
    return out


def main():
    docs, topics = get_docs_and_topics()
    doc_tokens = [simple_tokenize(doc) for doc in docs]
    word2idx, idx2word, counter = build_vocab(doc_tokens, max_vocab=VOCAB_SIZE)

    tfidf, _ = compute_tfidf(doc_tokens, word2idx)
    np.save(EMB_DIR / 'tfidf_matrix.npy', tfidf)

    topic_words = top_words_per_topic(tfidf, topics, idx2word)
    save_json(topic_words, OUTPUT_DIR / 'top_tfidf_words_per_topic.json')

    ppmi = build_ppmi(doc_tokens, word2idx, window=5)
    np.save(EMB_DIR / 'ppmi_matrix.npy', ppmi)
    save_json(word2idx, EMB_DIR / 'word2idx.json')

    plot_tsne(ppmi, idx2word, counter)

    queries = ['Pakistan', 'Hukumat', 'Adalat', 'Maeeshat', 'Fauj', 'Sehat', 'Taleem', 'Aabadi', 'Karachi', 'Cricket']
    nn = nearest_neighbors(ppmi, idx2word, queries, top_k=5)
    save_json(nn, OUTPUT_DIR / 'ppmi_nearest_neighbors.json')

    print('Saved TF-IDF, PPMI, t-SNE plot, topic words, and nearest neighbors.')


if __name__ == '__main__':
    main()
