from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
EMB_DIR = ROOT / 'embeddings'
MODEL_DIR = ROOT / 'models_saved'
PLOT_DIR = ROOT / 'plots'
OUTPUT_DIR = ROOT / 'outputs'

for p in [DATA_DIR, EMB_DIR, MODEL_DIR, PLOT_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CLEANED_PATH = DATA_DIR / 'cleaned.txt'
RAW_PATH = DATA_DIR / 'raw.txt'
METADATA_PATH = DATA_DIR / 'Metadata.json'

VOCAB_SIZE = 10_000
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
CLS_TOKEN = '<CLS>'

W2V_DIM = 100
W2V_DIM_LARGE = 200
WINDOW_SIZE = 5
NUM_NEGATIVE = 10
LR_W2V = 1e-3
BATCH_SIZE_W2V = 512
EPOCHS_W2V = 5

SEED = 42
DEVICE = 'cuda'

POS_TAGS = ['NOUN','VERB','ADJ','ADV','PRON','DET','CONJ','POST','NUM','PUNC','UNK']
NER_TAGS = ['O','B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC']

TOPIC_CATEGORIES = {
    1: ['election', 'government', 'minister', 'parliament'],
    2: ['cricket', 'match', 'team', 'player', 'score'],
    3: ['inflation', 'trade', 'bank', 'gdp', 'budget'],
    4: ['un', 'treaty', 'foreign', 'bilateral', 'conflict'],
    5: ['hospital', 'disease', 'vaccine', 'flood', 'education'],
}
