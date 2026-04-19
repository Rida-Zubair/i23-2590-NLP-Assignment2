import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, VOCAB_SIZE, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def simple_tokenize(text: str) -> List[str]:
    # Urdu-friendly fallback: split on whitespace but isolate punctuation.
    text = re.sub(r'[،۔!?؛:()\[\]"\'“”‘’]', lambda m: f' {m.group(0)} ', text)
    return [tok for tok in text.split() if tok.strip()]


def sentence_split(article: str) -> List[List[str]]:
    parts = re.split(r'[۔!?\n]+', article)
    sentences = []
    for part in parts:
        toks = simple_tokenize(part)
        if toks:
            sentences.append(toks)
    return sentences


def load_metadata(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def build_vocab(doc_tokens: List[List[str]], max_vocab: int = VOCAB_SIZE) -> Tuple[Dict[str, int], List[str], Counter]:
    counter = Counter(tok for doc in doc_tokens for tok in doc)
    most_common = [w for w, _ in counter.most_common(max_vocab - 3)]
    idx2word = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN] + most_common
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word, counter


def numericalize(tokens: List[str], word2idx: Dict[str, int]) -> List[int]:
    unk = word2idx[UNK_TOKEN]
    return [word2idx.get(t, unk) for t in tokens]


def save_json(obj, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_article_text(meta_item):
    if isinstance(meta_item, str):
        return meta_item
    if isinstance(meta_item, dict):
        for key in ['text', 'article', 'content', 'body', 'cleaned_text', 'document']:
            if key in meta_item:
                return str(meta_item[key])
    return ''
