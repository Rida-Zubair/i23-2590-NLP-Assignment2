# NLP Assignment 2 - File by File Solution

This project implements the full **Neural NLP Pipeline** in PyTorch from scratch, following the assignment requirements: TF-IDF, PPMI, Skip-gram Word2Vec, BiLSTM POS/NER, CRF, and a custom Transformer encoder.

## Folder Structure

- `config.py` - paths, constants, tags, hyperparameters
- `utils/io_utils.py` - reading data, tokenization, vocab building, helpers
- `utils/metrics.py` - accuracy, macro-F1, confusion matrix, NER entity scoring, MRR
- `utils/plot_utils.py` - plots for curves, confusion matrix, attention heatmaps
- `models/word2vec.py` - Skip-gram with negative sampling
- `models/crf.py` - CRF layer + Viterbi decoding
- `models/bilstm_tagger.py` - BiLSTM sequence labeling model for POS/NER
- `models/transformer_encoder.py` - custom Transformer encoder modules
- `part1_build_tfidf_ppmi.py` - TF-IDF, PPMI, t-SNE, top words, PPMI neighbours
- `part1_train_word2vec.py` - train Skip-gram Word2Vec and save embeddings
- `part1_evaluate_embeddings.py` - nearest neighbours, analogies, MRR summary
- `part2_prepare_sequence_data.py` - build POS/NER annotations and CoNLL files
- `part2_train_pos.py` - train/evaluate POS BiLSTM (frozen + fine-tuned embeddings)
- `part2_train_ner.py` - train/evaluate NER BiLSTM with and without CRF
- `part2_ablation.py` - run A1-A4 ablations
- `part3_prepare_cls_data.py` - build 5-class topic classification split
- `part3_train_transformer.py` - train/evaluate Transformer classifier and save heatmaps
- `run_all.py` - run every step in sequence

## Required Input Files
Put these files inside the `data/` folder:
- `cleaned.txt`
- `raw.txt`
- `Metadata.json`

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

## Important Notes

1. This code assumes **one article per line** in `cleaned.txt` and `raw.txt`.
2. `Metadata.json` can be a list of dictionaries. The loader is intentionally flexible, but you may need to map keys like `text`, `title`, `content`, `topic`, or `category` to your exact format.
3. For full marks, expand the **POS lexicon** and **NER gazetteers** in `part2_prepare_sequence_data.py` to match the assignment minimum counts.
4. Replace the placeholder manually labelled word-pair list in `part1_evaluate_embeddings.py` with your actual 20 labelled pairs for MRR.
5. The report and notebook are not included here; this package focuses on the code files.
