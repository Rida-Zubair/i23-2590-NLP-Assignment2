"""
VocabularyBuilder module for constructing word vocabularies from text corpora.

This module implements vocabulary construction with support for:
- Top-K most frequent words selection
- Unknown token (UNK) handling for out-of-vocabulary words
- Bidirectional word-index mappings
- Vocabulary persistence for reuse across modules
"""

from typing import Dict, List, Optional
import json
from collections import Counter


class VocabularyBuilder:
    """
    Builds and manages word vocabularies from text corpora.
    
    The VocabularyBuilder creates vocabularies by selecting the most frequent
    words from a corpus and mapping all other words to an unknown token (UNK).
    This ensures consistent word-to-index mappings across all NLP modules.
    
    Attributes:
        max_vocab_size (int): Maximum number of words in vocabulary (default: 10000)
        word_to_idx (Dict[str, int]): Mapping from words to indices
        idx_to_word (Dict[int, str]): Mapping from indices to words
        word_counts (Dict[str, int]): Word frequency counts from corpus
        unk_token (str): Token used for unknown/out-of-vocabulary words
    """
    
    def __init__(self, max_vocab_size: int = 10000):
        """
        Initialize VocabularyBuilder with specified vocabulary size.
        
        Args:
            max_vocab_size (int): Maximum number of words to include in vocabulary.
                                Words beyond this limit are mapped to UNK token.
        """
        self.max_vocab_size = max_vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self.unk_token = "<UNK>"
        
        # Reserve index 0 for UNK token
        self.word_to_idx[self.unk_token] = 0
        self.idx_to_word[0] = self.unk_token
    
    def build_from_corpus(self, corpus_path: str) -> None:
        """
        Build vocabulary from corpus file.
        
        Reads the corpus file, tokenizes all documents, counts word frequencies,
        and selects the top max_vocab_size most frequent words for the vocabulary.
        All other words will be mapped to the UNK token.
        
        Args:
            corpus_path (str): Path to the corpus text file
            
        Raises:
            FileNotFoundError: If corpus file does not exist
            IOError: If corpus file cannot be read
        """
        try:
            # Read corpus file
            with open(corpus_path, 'r', encoding='utf-8') as file:
                corpus_text = file.read()
            
            # Tokenize entire corpus
            all_words = self._tokenize_text(corpus_text)
            
            # Count word frequencies
            self.word_counts = Counter(all_words)
            
            # Select top max_vocab_size - 1 words (reserve 1 slot for UNK)
            most_common_words = self.word_counts.most_common(self.max_vocab_size - 1)
            
            # Build word-to-index mappings (UNK already at index 0)
            for idx, (word, count) in enumerate(most_common_words, start=1):
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        except IOError as e:
            raise IOError(f"Error reading corpus file {corpus_path}: {e}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words using basic preprocessing.
        
        Applies the following preprocessing steps:
        1. Convert to lowercase
        2. Split on whitespace
        3. Remove punctuation and special characters
        4. Filter out empty strings
        
        Args:
            text (str): Raw text to tokenize
            
        Returns:
            List[str]: List of tokenized words
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Split on whitespace and filter empty strings
        words = [word.strip() for word in text.split() if word.strip()]
        
        return words
    
    def get_word_index(self, word: str) -> int:
        """
        Get index for word, return UNK index if word not in vocabulary.
        
        Args:
            word (str): Word to look up
            
        Returns:
            int: Index of word in vocabulary, or UNK index if not found
        """
        return self.word_to_idx.get(word, self.word_to_idx[self.unk_token])
    
    def get_word_from_index(self, index: int) -> str:
        """
        Get word from index, return UNK token if index not found.
        
        Args:
            index (int): Index to look up
            
        Returns:
            str: Word at index, or UNK token if index not found
        """
        return self.idx_to_word.get(index, self.unk_token)
    
    def save_vocabulary(self, path: str) -> None:
        """
        Save vocabulary mappings to JSON file for reuse.
        
        Args:
            path (str): Path where vocabulary JSON file will be saved
            
        Raises:
            IOError: If file cannot be written
        """
        # Implementation will be added in next task
        pass
    
    def load_vocabulary(self, path: str) -> None:
        """
        Load vocabulary mappings from JSON file.
        
        Args:
            path (str): Path to vocabulary JSON file
            
        Raises:
            FileNotFoundError: If vocabulary file does not exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        # Implementation will be added in next task
        pass
    
    @property
    def vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.word_to_idx)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word_to_idx
    
    def __repr__(self) -> str:
        """String representation of VocabularyBuilder."""
        return f"VocabularyBuilder(vocab_size={self.vocab_size}, max_size={self.max_vocab_size})"