import json
from collections import Counter

from sklearn.model_selection import train_test_split

from config import CLEANED_PATH, METADATA_PATH, DATA_DIR, OUTPUT_DIR, TOPIC_CATEGORIES, SEED
from utils.io_utils import read_lines, load_metadata, simple_tokenize


def assign_category(meta_item):
    text = ''
    if isinstance(meta_item, dict):
        text = ' '.join(str(meta_item.get(k, '')) for k in ['title', 'text', 'article', 'content', 'body']).lower()
    else:
        text = str(meta_item).lower()
    for cat, keywords in TOPIC_CATEGORIES.items():
        if any(k.lower() in text for k in keywords):
            return cat - 1
    return 4  # fallback to Health & Society


def main():
    docs = read_lines(CLEANED_PATH)
    meta = load_metadata(METADATA_PATH)
    
    # Convert dict to list if needed and handle missing entries
    if isinstance(meta, dict):
        meta_list = []
        for i in range(len(docs)):
            key = str(i+1)
            if key in meta:
                meta_list.append(meta[key])
            else:
                # Create a fallback entry for missing metadata
                meta_list.append({'title': '', 'text': ''})
    else:
        meta_list = meta[:len(docs)]
        # Pad with empty entries if metadata is shorter than docs
        while len(meta_list) < len(docs):
            meta_list.append({'title': '', 'text': ''})
    
    labels = [assign_category(m) for m in meta_list]

    train_docs, temp_docs, train_y, temp_y = train_test_split(docs, labels, test_size=0.30, stratify=labels, random_state=SEED)
    val_docs, test_docs, val_y, test_y = train_test_split(temp_docs, temp_y, test_size=0.50, stratify=temp_y, random_state=SEED)

    payload = {
        'train': [{'text': d, 'label': y} for d, y in zip(train_docs, train_y)],
        'val': [{'text': d, 'label': y} for d, y in zip(val_docs, val_y)],
        'test': [{'text': d, 'label': y} for d, y in zip(test_docs, test_y)],
        'distribution': {
            'train': dict(Counter(train_y)),
            'val': dict(Counter(val_y)),
            'test': dict(Counter(test_y)),
        }
    }
    with open(DATA_DIR / 'topic_classification_splits.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'topic_class_distribution.json', 'w', encoding='utf-8') as f:
        json.dump(payload['distribution'], f, indent=2)
    print('Saved topic classification splits.')


if __name__ == '__main__':
    main()
