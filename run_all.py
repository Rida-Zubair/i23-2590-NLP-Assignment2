import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'part1_build_tfidf_ppmi.py',
    'part1_train_word2vec.py',
    'part1_evaluate_embeddings.py',
    'part2_prepare_sequence_data.py',
    'part2_train_pos.py',
    'part2_train_ner.py',
    'part2_ablation.py',
    'part3_prepare_cls_data.py',
    'part3_train_transformer.py',
]

for script in SCRIPTS:
    print(f'Running {script}')
    subprocess.run([sys.executable, str(ROOT / script)], check=True)
