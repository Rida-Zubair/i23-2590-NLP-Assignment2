import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import EMB_DIR, OUTPUT_DIR
from utils.metrics import mean_reciprocal_rank


QUERIES = [
    'پاکستان', 'حکومت', 'عدالت', 'معیشت', 'فوج', 'صحت', 'تعلیم', 'آبادی',
    'کرکٹ', 'انتخابات', 'قانون', 'تجارت', 'ہسپتال', 'اسکول', 'بینک', 'شہر'
]

ANALOGIES = [
    ('مرد', 'بادشاہ', 'عورت'),      # man:king::woman:?
    ('لاہور', 'پنجاب', 'کراچی'),    # Lahore:Punjab::Karachi:?
    ('ڈاکٹر', 'ہسپتال', 'استاد'),   # doctor:hospital::teacher:?
    ('کرکٹ', 'کھلاڑی', 'حکومت'),    # cricket:player::government:?
    ('اسکول', 'تعلیم', 'ہسپتال'),    # school:education::hospital:?
    ('فوج', 'سپاہی', 'عدالت'),      # army:soldier::court:?
    ('ووٹ', 'انتخابات', 'قانون'),   # vote:elections::law:?
    ('بینک', 'معیشت', 'ہسپتال'),     # bank:economy::hospital:?
    ('پاکستان', 'اسلام_آباد', 'سندھ'), # Pakistan:Islamabad::Sindh:?
    ('استاد', 'کلاس', 'ڈاکٹر'),     # teacher:class::doctor:?
    ('دریا', 'پانی', 'پہاڑ'),       # river:water::mountain:?
    ('شہر', 'آبادی', 'ملک'),        # city:population::country:?
    ('موسم', 'بارش', 'زمین'),       # weather:rain::earth:?
    ('کتاب', 'علم', 'دوا'),         # book:knowledge::medicine:?
    ('راجا', 'محل', 'کسان')         # king:palace::farmer:?
]


def load_vectors():
    with open(EMB_DIR / 'word2idx.json', 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {i: w for w, i in word2idx.items()}
    w2v = np.load(EMB_DIR / 'embeddings_w2v.npy')
    ppmi = np.load(EMB_DIR / 'ppmi_matrix.npy')
    return word2idx, idx2word, w2v, ppmi


def nearest_neighbors(vectors, word2idx, idx2word, query, top_k=10):
    if query not in word2idx:
        return []
    q_idx = word2idx[query]
    sims = cosine_similarity(vectors[q_idx:q_idx+1], vectors)[0]
    order = np.argsort(-sims)
    return [idx2word[i] for i in order if i != q_idx][:top_k]


def run_analogies(vectors, word2idx, idx2word):
    results = {}
    norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    for a, b, c in ANALOGIES:
        if any(w not in word2idx for w in [a, b, c]):
            results[f'{a}:{b}::{c}:?'] = []
            continue
        vec = norm[word2idx[b]] - norm[word2idx[a]] + norm[word2idx[c]]
        sims = norm @ vec
        for w in [a, b, c]:
            sims[word2idx[w]] = -1e9
        best = np.argsort(-sims)[:3]
        results[f'{a}:{b}::{c}:?'] = [idx2word[i] for i in best]
    return results


def main():
    word2idx, idx2word, w2v, ppmi = load_vectors()
    nn_out = {q: nearest_neighbors(w2v, word2idx, idx2word, q, top_k=10) for q in QUERIES}
    analogy_out = run_analogies(w2v, word2idx, idx2word)

    # 20 manually labelled word pairs for MRR evaluation - semantically related Urdu words
    labelled_tokens = [
        ('پاکستان', 'اسلام_آباد'),  # Pakistan - Islamabad (country-capital)
        ('کرکٹ', 'میچ'),           # Cricket - Match (sport-event)
        ('ہسپتال', 'ڈاکٹر'),       # Hospital - Doctor (place-profession)
        ('اسکول', 'استاد'),        # School - Teacher (place-profession)
        ('بینک', 'پیسہ'),          # Bank - Money (institution-currency)
        ('عدالت', 'جج'),           # Court - Judge (institution-profession)
        ('فوج', 'سپاہی'),          # Army - Soldier (organization-member)
        ('حکومت', 'وزیر'),         # Government - Minister (system-role)
        ('انتخابات', 'ووٹ'),       # Elections - Vote (process-action)
        ('قانون', 'عدل'),          # Law - Justice (concept-principle)
        ('معیشت', 'تجارت'),        # Economy - Trade (system-activity)
        ('تعلیم', 'علم'),          # Education - Knowledge (process-outcome)
        ('صحت', 'دوا'),            # Health - Medicine (state-treatment)
        ('آبادی', 'لوگ'),          # Population - People (collective-individual)
        ('ملک', 'قوم'),            # Country - Nation (territory-people)
        ('شہر', 'آبادی'),          # City - Population (place-inhabitants)
        ('دریا', 'پانی'),          # River - Water (feature-substance)
        ('پہاڑ', 'بلندی'),         # Mountain - Height (feature-attribute)
        ('موسم', 'بارش'),          # Weather - Rain (system-phenomenon)
        ('کھیل', 'کھلاڑی')         # Sport - Player (activity-participant)
    ]
    pairs = [(word2idx[a], word2idx[b]) for a, b in labelled_tokens if a in word2idx and b in word2idx]
    sim = cosine_similarity(w2v)
    mrr = mean_reciprocal_rank(sim, pairs)

    with open(OUTPUT_DIR / 'w2v_neighbors.json', 'w', encoding='utf-8') as f:
        json.dump(nn_out, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'w2v_analogies.json', 'w', encoding='utf-8') as f:
        json.dump(analogy_out, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'embedding_eval_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'mrr': mrr}, f, indent=2)

    print('Saved nearest neighbors, analogies, and MRR summary.')


if __name__ == '__main__':
    main()
