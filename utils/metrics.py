from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_score(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def extract_entities(tags: List[str]) -> List[Tuple[str, int, int]]:
    entities = []
    start = None
    ent_type = None
    for i, tag in enumerate(tags + ['O']):
        if tag.startswith('B-'):
            if start is not None:
                entities.append((ent_type, start, i - 1))
            ent_type = tag[2:]
            start = i
        elif tag.startswith('I-'):
            if start is None or ent_type != tag[2:]:
                if start is not None:
                    entities.append((ent_type, start, i - 1))
                ent_type = tag[2:]
                start = i
        else:
            if start is not None:
                entities.append((ent_type, start, i - 1))
                start = None
                ent_type = None
    return entities


def ner_entity_report(gold_sequences: List[List[str]], pred_sequences: List[List[str]]):
    by_type = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for gold, pred in zip(gold_sequences, pred_sequences):
        gold_set = set(extract_entities(gold))
        pred_set = set(extract_entities(pred))
        for ent in pred_set:
            if ent in gold_set:
                by_type[ent[0]]['tp'] += 1
            else:
                by_type[ent[0]]['fp'] += 1
        for ent in gold_set:
            if ent not in pred_set:
                by_type[ent[0]]['fn'] += 1

    report = {}
    total = {'tp': 0, 'fp': 0, 'fn': 0}
    for ent_type, d in by_type.items():
        p = d['tp'] / (d['tp'] + d['fp'] + 1e-12)
        r = d['tp'] / (d['tp'] + d['fn'] + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        report[ent_type] = {'precision': p, 'recall': r, 'f1': f1, **d}
        for k in total:
            total[k] += d[k]
    p = total['tp'] / (total['tp'] + total['fp'] + 1e-12)
    r = total['tp'] / (total['tp'] + total['fn'] + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    report['overall'] = {'precision': p, 'recall': r, 'f1': f1, **total}
    return report


def mean_reciprocal_rank(similarity_matrix: np.ndarray, labelled_pairs: List[Tuple[int, int]]) -> float:
    ranks = []
    for source_idx, target_idx in labelled_pairs:
        sims = similarity_matrix[source_idx]
        order = np.argsort(-sims)
        rank = np.where(order == target_idx)[0]
        if len(rank):
            ranks.append(1.0 / (int(rank[0]) + 1))
        else:
            ranks.append(0.0)
    return float(np.mean(ranks)) if ranks else 0.0
