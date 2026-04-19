# Implementation Plan: NLP Assignment Pipeline - TF-IDF Module

## Overview

This implementation plan focuses on Part 1.1 (TF-IDF Module) of the CS-4063 Neural NLP Pipeline assignment. The approach emphasizes strict version control discipline with incremental development, separate branches, and meaningful commits for each component. All implementations must be manual using only basic PyTorch tensors.

## Tasks

- [-] 1. Project initialization and git setup
  - [x] 1.1 Initialize git repository with proper .gitignore
    - Create .gitignore for Python projects (*.pyc, __pycache__, .env, etc.)
    - Initialize git repository with initial commit
    - _Requirements: 1.1, 1.3_
  
  - [-] 1.2 Create project directory structure
    - Create src/, data/, embeddings/, tests/ directories
    - Create __init__.py files for Python package structure
    - Add requirements.txt with PyTorch dependency
    - _Requirements: 1.2, 1.5_
  
  - [ ] 1.3 Create feature branch for TF-IDF development
    - Create and checkout `feature/tfidf-module` branch
    - Commit initial project structure
    - _Requirements: 1.3, 8.1_

- [ ] 2. Vocabulary builder implementation
  - [ ] 2.1 Implement VocabularyBuilder class structure
    - Create src/vocabulary/builder.py with class definition
    - Implement __init__ method with max_vocab_size parameter
    - Add word_to_idx, idx_to_word, word_counts attributes
    - _Requirements: 2.1, 2.4_
  
  - [ ] 2.2 Implement corpus tokenization and word counting
    - Add build_from_corpus method to read and tokenize cleaned.txt
    - Implement basic tokenization (lowercase, split on whitespace)
    - Count word frequencies across entire corpus
    - _Requirements: 2.1, 7.1_
  
  - [ ] 2.3 Implement vocabulary selection and UNK handling
    - Select top 10,000 most frequent words for vocabulary
    - Create bidirectional word-index mappings
    - Implement get_word_index method with UNK token support
    - _Requirements: 2.2, 2.3, 2.4_
  
  - [ ] 2.4 Implement vocabulary persistence
    - Add save_vocabulary and load_vocabulary methods
    - Save vocabulary mappings to JSON format for reuse
    - _Requirements: 2.5_
  
  - [ ]* 2.5 Write unit tests for VocabularyBuilder
    - Test vocabulary construction with sample data
    - Test UNK token handling for unknown words
    - Test vocabulary save/load functionality
    - _Requirements: 8.2_

- [ ] 3. Document processor implementation
  - [ ] 3.1 Implement DocumentProcessor class structure
    - Create src/preprocessing/processor.py with class definition
    - Initialize with VocabularyBuilder dependency
    - _Requirements: 2.6, 7.1_
  
  - [ ] 3.2 Implement document tokenization methods
    - Add tokenize_document method for single document processing
    - Add load_corpus method to read text files
    - Handle different text encodings gracefully
    - _Requirements: 7.1, 7.4_
  
  - [ ] 3.3 Implement corpus processing with vocabulary mapping
    - Add process_corpus method to convert text to token indices
    - Apply vocabulary mappings consistently
    - Handle documents with all UNK tokens
    - _Requirements: 2.6, 7.5_
  
  - [ ]* 3.4 Write unit tests for DocumentProcessor
    - Test tokenization with sample documents
    - Test vocabulary mapping application
    - Test edge cases (empty documents, all UNK words)
    - _Requirements: 8.2_

- [ ] 4. Checkpoint - Basic infrastructure complete
  - Ensure vocabulary and document processing tests pass
  - Commit all changes with descriptive messages
  - Ask the user if questions arise

- [ ] 5. TF-IDF matrix computation implementation
  - [ ] 5.1 Implement TFIDFMatrix class structure
    - Create src/tfidf/matrix.py with class definition
    - Initialize with vocab_size and num_docs parameters
    - Create PyTorch tensors for tf_matrix, df_vector, tfidf_matrix
    - _Requirements: 3.1, 6.4_
  
  - [ ] 5.2 Implement term frequency (TF) computation
    - Add compute_tf method using PyTorch tensor operations
    - Calculate TF as word_count / total_words per document
    - Use only basic tensor operations (no high-level modules)
    - _Requirements: 3.2, 6.4_
  
  - [ ] 5.3 Implement document frequency (DF) computation
    - Add compute_df method to count documents containing each word
    - Use PyTorch tensor operations for efficient counting
    - Store results in df_vector tensor
    - _Requirements: 3.3, 6.4_
  
  - [ ] 5.4 Implement TF-IDF formula computation
    - Add compute_tfidf method using formula: TF × log(N / (1 + DF))
    - Use torch.log for logarithmic calculations
    - Ensure all operations use basic PyTorch tensors only
    - _Requirements: 3.4, 6.4_
  
  - [ ]* 5.5 Write property test for TF-IDF mathematical correctness
    - **Property 2: TF-IDF Mathematical Correctness**
    - **Validates: Requirements 3.4**
    - Test with hand-calculated examples
    - Verify formula implementation accuracy
  
  - [ ] 5.6 Implement matrix serialization
    - Add save_matrix method to export as embeddings/tfidf_matrix.npy
    - Convert PyTorch tensor to NumPy array for compatibility
    - Ensure proper file path handling
    - _Requirements: 3.5, 9.1_
  
  - [ ]* 5.7 Write unit tests for TFIDFMatrix
    - Test TF computation with sample documents
    - Test DF computation accuracy
    - Test TF-IDF formula with known results
    - Test matrix dimensions and non-negative values
    - _Requirements: 8.2_

- [ ] 6. Integration and end-to-end pipeline
  - [ ] 6.1 Create main pipeline script
    - Create src/main.py to orchestrate entire TF-IDF pipeline
    - Load cleaned.txt corpus and process through all components
    - Generate final TF-IDF matrix output
    - _Requirements: 7.1, 8.4_
  
  - [ ] 6.2 Implement utility functions
    - Create src/utils/io_utils.py for file I/O operations
    - Add functions for loading datasets and saving results
    - Handle file path resolution and error cases
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [ ]* 6.3 Write integration tests
    - Test complete pipeline with sample data
    - Verify output matrix format and dimensions
    - Test with actual cleaned.txt if available
    - _Requirements: 8.2, 8.4_
  
  - [ ]* 6.4 Write property test for vocabulary consistency
    - **Property 1: Vocabulary Consistency**
    - **Validates: Requirements 2.5**
    - Test vocabulary save/load maintains identical mappings
  
  - [ ]* 6.5 Write property test for matrix dimensionality
    - **Property 3: Matrix Dimensionality**
    - **Validates: Requirements 3.1, 3.5**
    - Verify output dimensions match vocab_size × num_docs

- [ ] 7. Code quality and documentation
  - [ ] 7.1 Add comprehensive docstrings
    - Document all classes and methods with proper docstrings
    - Follow Google or NumPy docstring conventions
    - Include parameter types and return value descriptions
    - _Requirements: 10.2_
  
  - [ ] 7.2 Apply PEP 8 formatting
    - Format all code according to PEP 8 guidelines
    - Use consistent naming conventions
    - Ensure proper import organization
    - _Requirements: 10.1, 10.3_
  
  - [ ] 7.3 Create comprehensive README
    - Document setup instructions and dependencies
    - Provide usage examples for TF-IDF module
    - Include project structure explanation
    - _Requirements: 10.5_

- [ ] 8. Final git workflow and merge
  - [ ] 8.1 Review commit history
    - Ensure minimum 5-10 meaningful commits on feature branch
    - Verify each commit has descriptive message
    - Check that commits are logically organized
    - _Requirements: 1.4, 8.1, 10.4_
  
  - [ ] 8.2 Merge feature branch to main
    - Switch to main branch and merge feature/tfidf-module
    - Create merge commit with summary of TF-IDF implementation
    - Tag release as v1.0-tfidf for milestone tracking
    - _Requirements: 8.3_
  
  - [ ] 8.3 Prepare for next module development
    - Create placeholder branches for PMI and Word2Vec modules
    - Update project documentation with TF-IDF completion status
    - _Requirements: 8.1_

- [ ] 9. Final checkpoint - TF-IDF module complete
  - Ensure all tests pass and TF-IDF matrix is generated successfully
  - Verify embeddings/tfidf_matrix.npy exists with correct format
  - Ask the user if questions arise before proceeding to PMI module

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Minimum 5-10 commits required with descriptive messages
- All implementations must use only basic PyTorch tensors
- No pretrained models, HuggingFace, or high-level PyTorch modules allowed
- Focus on TF-IDF module only; PMI and Word2Vec will be separate phases
- Property tests validate mathematical correctness and consistency
- Integration tests ensure end-to-end pipeline functionality