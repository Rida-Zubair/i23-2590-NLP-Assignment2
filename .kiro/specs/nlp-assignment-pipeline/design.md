# Design Document: NLP Assignment Pipeline

## Overview

This document outlines the technical design for implementing a Neural NLP Pipeline for CS-4063 Assignment 2, focusing initially on the TF-IDF module with strict version control discipline and manual implementation constraints.

## Architecture

### System Components

```
NLP_Pipeline/
├── src/
│   ├── vocabulary/
│   │   ├── __init__.py
│   │   └── builder.py          # VocabularyBuilder class
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── processor.py        # DocumentProcessor class
│   ├── tfidf/
│   │   ├── __init__.py
│   │   ├── matrix.py          # TFIDFMatrix class
│   │   └── computer.py        # MatrixComputer class
│   ├── utils/
│   │   ├── __init__.py
│   │   └── io_utils.py        # File I/O utilities
│   └── __init__.py
├── data/                      # Input datasets
├── embeddings/               # Output matrices
├── tests/                    # Unit tests
└── requirements.txt
```

### Core Classes and Interfaces

#### VocabularyBuilder
```python
class VocabularyBuilder:
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = {}
    
    def build_from_corpus(self, corpus_path: str) -> None:
        """Build vocabulary from corpus file"""
    
    def get_word_index(self, word: str) -> int:
        """Get index for word, return UNK if not in vocab"""
    
    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary mappings to file"""
```

#### DocumentProcessor
```python
class DocumentProcessor:
    def __init__(self, vocabulary_builder: VocabularyBuilder):
        self.vocab_builder = vocabulary_builder
    
    def tokenize_document(self, text: str) -> List[str]:
        """Tokenize single document"""
    
    def process_corpus(self, corpus_path: str) -> List[List[int]]:
        """Process entire corpus into token indices"""
    
    def load_corpus(self, file_path: str) -> List[str]:
        """Load corpus from file"""
```

#### TFIDFMatrix
```python
class TFIDFMatrix:
    def __init__(self, vocab_size: int, num_docs: int):
        self.vocab_size = vocab_size
        self.num_docs = num_docs
        self.tf_matrix = torch.zeros(num_docs, vocab_size)
        self.df_vector = torch.zeros(vocab_size)
        self.tfidf_matrix = None
    
    def compute_tf(self, documents: List[List[int]]) -> None:
        """Compute term frequency matrix"""
    
    def compute_df(self, documents: List[List[int]]) -> None:
        """Compute document frequency vector"""
    
    def compute_tfidf(self) -> torch.Tensor:
        """Compute TF-IDF matrix using formula: TF * log(N / (1 + DF))"""
    
    def save_matrix(self, output_path: str) -> None:
        """Save TF-IDF matrix as numpy array"""
```

## Implementation Strategy

### Phase 1: TF-IDF Module Implementation

#### Git Workflow Strategy
1. **Branch Structure**: Create `feature/tfidf-module` branch from main
2. **Incremental Commits**: Minimum 5-10 commits with specific functionality
3. **Commit Strategy**: Each class/method gets its own commit
4. **Testing**: Unit tests committed alongside implementation

#### Mathematical Implementation

**Term Frequency (TF)**:
```
TF(word, document) = count(word, document) / total_words(document)
```

**Document Frequency (DF)**:
```
DF(word) = number_of_documents_containing(word)
```

**TF-IDF Formula**:
```
TF-IDF(word, document) = TF(word, document) × log(N / (1 + DF(word)))
```
Where N = total number of documents

#### PyTorch Implementation Details

**Tensor Operations**:
- Use `torch.zeros()` for matrix initialization
- Use `torch.log()` for logarithmic calculations
- Use tensor indexing for efficient matrix operations
- Avoid high-level PyTorch modules (nn.Module, etc.)

**Memory Efficiency**:
- Process documents in batches if corpus is large
- Use sparse tensor representations where appropriate
- Implement incremental matrix updates

## Data Flow

### TF-IDF Pipeline Flow
```
cleaned.txt → DocumentProcessor → tokenized_docs → VocabularyBuilder → vocab_mappings
                                                                           ↓
tokenized_docs + vocab_mappings → TFIDFMatrix → tf_computation → df_computation → tfidf_matrix
                                                                                        ↓
                                                                              embeddings/tfidf_matrix.npy
```

### File Processing Strategy
1. **Input**: Load `cleaned.txt` as primary corpus
2. **Tokenization**: Split on whitespace, convert to lowercase, remove punctuation
3. **Vocabulary**: Build from most frequent 10,000 words
4. **Matrix Construction**: Create term-document matrix
5. **TF-IDF Computation**: Apply mathematical formula
6. **Output**: Save as NumPy array for compatibility

## Testing Strategy

### Unit Test Coverage
- VocabularyBuilder: Test vocabulary construction, UNK handling
- DocumentProcessor: Test tokenization, corpus loading
- TFIDFMatrix: Test TF computation, DF computation, TF-IDF formula
- Integration: Test end-to-end pipeline with small sample data

### Test Data Strategy
- Create small sample corpus for unit tests
- Verify mathematical correctness with hand-calculated examples
- Test edge cases: empty documents, unknown words, single-word documents

## Constraints and Limitations

### Manual Implementation Requirements
- **NO** pretrained models or embeddings
- **NO** HuggingFace, Gensim, or similar libraries
- **NO** nn.Transformer, nn.MultiheadAttention, nn.TransformerEncoder
- **ONLY** basic PyTorch tensors and mathematical operations
- **MANUAL** implementation of all algorithms

### Performance Considerations
- Vocabulary limited to 10,000 most frequent words
- Efficient sparse matrix handling for large corpora
- Memory-conscious processing for large documents
- Incremental computation where possible

## Future Extensions

### PMI Module (Phase 2)
- Reuse VocabularyBuilder from TF-IDF module
- Implement co-occurrence matrix computation
- Calculate pointwise mutual information scores

### Word2Vec Module (Phase 3)
- Reuse vocabulary infrastructure
- Implement Skip-gram architecture manually
- Neural network training with gradient descent

## Correctness Properties

### Property 1: Vocabulary Consistency
**Property**: All modules must use identical word-to-index mappings
**Validation**: Vocabulary saved and loaded consistently across modules

### Property 2: TF-IDF Mathematical Correctness
**Property**: TF-IDF computation follows exact mathematical formula
**Validation**: Hand-calculated examples match computed results

### Property 3: Matrix Dimensionality
**Property**: Output matrix dimensions match vocabulary size and document count
**Validation**: Matrix shape equals (num_documents, vocab_size)

### Property 4: Non-negative Values
**Property**: All TF-IDF values must be non-negative
**Validation**: No negative values in final matrix

### Property 5: Sparse Representation Efficiency
**Property**: Zero TF-IDF values for words not appearing in documents
**Validation**: Matrix sparsity matches expected document-word relationships