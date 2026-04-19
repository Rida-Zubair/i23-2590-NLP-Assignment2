# Requirements Document

## Introduction

This document specifies the requirements for implementing a Neural NLP Pipeline for CS-4063 Assignment 2. The system must implement TF-IDF, PMI, and Word2Vec modules from scratch using PyTorch, with strict version control discipline and incremental development approach.

## Glossary

- **NLP_Pipeline**: The complete neural natural language processing system
- **TF_IDF_Module**: Term Frequency-Inverse Document Frequency computation module
- **PMI_Module**: Pointwise Mutual Information computation module  
- **Word2Vec_Module**: Word embedding generation module using neural networks
- **Vocabulary_Builder**: Component that constructs vocabulary from corpus
- **Document_Processor**: Component that processes and tokenizes text documents
- **Matrix_Computer**: Component that computes mathematical matrices (TF-IDF, PMI)
- **Neural_Trainer**: Component that trains neural network models
- **Git_Workflow**: Version control process with branching and incremental commits
- **Corpus**: The text dataset (cleaned.txt, raw.txt)
- **Metadata**: Label information stored in Metadata.json

## Requirements

### Requirement 1: Project Structure and Git Initialization

**User Story:** As a developer, I want a well-organized project structure with proper git initialization, so that I can maintain clean version control throughout development.

#### Acceptance Criteria

1. THE Git_Workflow SHALL initialize a new repository with proper .gitignore
2. THE NLP_Pipeline SHALL organize code into modular directories (src/, data/, embeddings/, tests/)
3. THE Git_Workflow SHALL create separate branches for each major module implementation
4. THE Git_Workflow SHALL enforce minimum 5-10 meaningful commits with descriptive messages
5. THE NLP_Pipeline SHALL maintain clean separation between data processing, model implementation, and utilities

### Requirement 2: Vocabulary Construction

**User Story:** As a developer, I want to build a vocabulary from the corpus, so that I can create consistent word-to-index mappings for all modules.

#### Acceptance Criteria

1. WHEN provided with cleaned.txt corpus, THE Vocabulary_Builder SHALL tokenize all documents
2. THE Vocabulary_Builder SHALL select the top 10,000 most frequent words for the vocabulary
3. THE Vocabulary_Builder SHALL map all words not in top 10,000 to a special <UNK> token
4. THE Vocabulary_Builder SHALL create bidirectional word-to-index and index-to-word mappings
5. THE Vocabulary_Builder SHALL save vocabulary mappings for reuse across modules
6. THE Document_Processor SHALL apply vocabulary mappings consistently across all text processing

### Requirement 3: TF-IDF Matrix Computation

**User Story:** As a developer, I want to compute TF-IDF representations, so that I can create document embeddings based on term importance.

#### Acceptance Criteria

1. THE TF_IDF_Module SHALL construct a term-document matrix using the built vocabulary
2. THE Matrix_Computer SHALL calculate term frequency (TF) for each word in each document
3. THE Matrix_Computer SHALL calculate document frequency (DF) for each word across the corpus
4. THE TF_IDF_Module SHALL compute TF-IDF using the formula: TF-IDF(w,d) = TF(w,d) × log(N / (1 + df(w)))
5. THE TF_IDF_Module SHALL save the final matrix as embeddings/tfidf_matrix.npy
6. THE TF_IDF_Module SHALL handle sparse matrix operations efficiently for large vocabularies

### Requirement 4: PMI Matrix Computation

**User Story:** As a developer, I want to compute Pointwise Mutual Information matrices, so that I can capture word co-occurrence relationships.

#### Acceptance Criteria

1. THE PMI_Module SHALL construct word co-occurrence matrices from the corpus
2. THE Matrix_Computer SHALL calculate joint probabilities for word pairs within context windows
3. THE PMI_Module SHALL compute PMI scores using statistical formulas
4. THE PMI_Module SHALL save PMI matrices in the embeddings/ directory
5. THE PMI_Module SHALL reuse the vocabulary from the TF-IDF module for consistency

### Requirement 5: Word2Vec Neural Implementation

**User Story:** As a developer, I want to implement Word2Vec from scratch, so that I can generate dense word embeddings using neural networks.

#### Acceptance Criteria

1. THE Word2Vec_Module SHALL implement Skip-gram architecture using PyTorch tensors only
2. THE Neural_Trainer SHALL train embeddings without using nn.Transformer, nn.MultiheadAttention, or nn.TransformerEncoder
3. THE Word2Vec_Module SHALL implement negative sampling for efficient training
4. THE Neural_Trainer SHALL use gradient descent optimization implemented manually
5. THE Word2Vec_Module SHALL save trained embeddings in the embeddings/ directory
6. THE Word2Vec_Module SHALL reuse the vocabulary from previous modules

### Requirement 6: Manual Implementation Constraints

**User Story:** As a student, I want to implement everything manually, so that I can demonstrate understanding of underlying algorithms.

#### Acceptance Criteria

1. THE NLP_Pipeline SHALL NOT use any pretrained models or embeddings
2. THE NLP_Pipeline SHALL NOT import HuggingFace, Gensim, or similar high-level libraries
3. THE Neural_Trainer SHALL NOT use nn.Transformer, nn.MultiheadAttention, or nn.TransformerEncoder
4. THE NLP_Pipeline SHALL implement all mathematical operations using basic PyTorch tensors
5. THE NLP_Pipeline SHALL demonstrate manual implementation of gradient computation where required

### Requirement 7: Data Integration and Processing

**User Story:** As a developer, I want to process the provided datasets, so that I can train and evaluate the NLP pipeline.

#### Acceptance Criteria

1. WHEN provided with cleaned.txt, THE Document_Processor SHALL load and tokenize the main corpus
2. WHEN provided with raw.txt, THE Document_Processor SHALL load the baseline comparison data
3. WHEN provided with Metadata.json, THE Document_Processor SHALL load and parse label information
4. THE Document_Processor SHALL handle different text encodings and formats gracefully
5. THE NLP_Pipeline SHALL maintain data integrity throughout all processing steps

### Requirement 8: Incremental Development and Testing

**User Story:** As a developer, I want to implement modules incrementally, so that I can test and validate each component before integration.

#### Acceptance Criteria

1. THE Git_Workflow SHALL create separate branches for TF-IDF, PMI, and Word2Vec implementations
2. THE NLP_Pipeline SHALL implement comprehensive unit tests for each module
3. WHEN each module is completed, THE Git_Workflow SHALL merge to main branch with proper testing
4. THE NLP_Pipeline SHALL provide integration points between modules for seamless data flow
5. THE NLP_Pipeline SHALL validate outputs at each stage before proceeding to next module

### Requirement 9: Output and Serialization

**User Story:** As a developer, I want to save computed embeddings and matrices, so that I can reuse results and submit assignment deliverables.

#### Acceptance Criteria

1. THE TF_IDF_Module SHALL save matrices as embeddings/tfidf_matrix.npy in NumPy format
2. THE PMI_Module SHALL save matrices in the embeddings/ directory with descriptive filenames
3. THE Word2Vec_Module SHALL save trained embeddings in standard format for evaluation
4. THE NLP_Pipeline SHALL create a results summary with model performance metrics
5. THE NLP_Pipeline SHALL ensure all output files are properly formatted for assignment submission

### Requirement 10: Code Quality and Documentation

**User Story:** As a developer, I want clean, well-documented code, so that the implementation is maintainable and meets academic standards.

#### Acceptance Criteria

1. THE NLP_Pipeline SHALL follow Python PEP 8 style guidelines consistently
2. THE NLP_Pipeline SHALL include comprehensive docstrings for all classes and functions
3. THE NLP_Pipeline SHALL organize code into logical modules with clear interfaces
4. THE Git_Workflow SHALL include meaningful commit messages describing each change
5. THE NLP_Pipeline SHALL include a comprehensive README with setup and usage instructions