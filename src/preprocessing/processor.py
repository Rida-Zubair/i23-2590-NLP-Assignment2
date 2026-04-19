"""
DocumentProcessor module for tokenizing and processing text documents.

This module handles document preprocessing including:
- Text tokenization and cleaning
- Vocabulary mapping application
- Corpus loading and processing
- Integration with VocabularyBuilder for consistent word-index mappings
"""

from typing import List, Optional, Union
import os
from ..vocabulary.builder import VocabularyBuilder


class DocumentProcessor:
    """
    Processes text documents using a vocabulary for consistent tokenization.
    
    The DocumentProcessor works with a VocabularyBuilder to ensure consistent
    word-to-index mappings across all text processing operations. It handles
    document loading, tokenization, and conversion to numerical representations.
    
    Attributes:
        vocab_builder (VocabularyBuilder): Vocabulary builder for word-index mappings
    """
    
    def __init__(self, vocabulary_builder: VocabularyBuilder):
        """
        Initialize DocumentProcessor with a VocabularyBuilder.
        
        Args:
            vocabulary_builder (VocabularyBuilder): Pre-built vocabulary for word mappings
        """
        self.vocab_builder = vocabulary_builder
    
    def tokenize_document(self, text: str) -> List[str]:
        """
        Tokenize single document using the same method as VocabularyBuilder.
        
        Args:
            text (str): Raw document text to tokenize
            
        Returns:
            List[str]: List of tokenized words
        """
        return self.vocab_builder._tokenize_text(text)
    
    def load_corpus(self, file_path: str) -> List[str]:
        """
        Load corpus from file and return list of documents.
        
        Args:
            file_path (str): Path to corpus file
            
        Returns:
            List[str]: List of document strings
            
        Raises:
            FileNotFoundError: If corpus file does not exist
            IOError: If file cannot be read
        """
        # Implementation will be added in next task
        pass
    
    def process_corpus(self, corpus_path: str) -> List[List[int]]:
        """
        Process entire corpus into token indices using vocabulary.
        
        Args:
            corpus_path (str): Path to corpus file
            
        Returns:
            List[List[int]]: List of documents as token index lists
            
        Raises:
            FileNotFoundError: If corpus file does not exist
            IOError: If file cannot be read
        """
        # Implementation will be added in next task
        pass
    
    def process_document(self, text: str) -> List[int]:
        """
        Process single document into token indices.
        
        Args:
            text (str): Document text to process
            
        Returns:
            List[int]: Document as list of vocabulary indices
        """
        words = self.tokenize_document(text)
        return [self.vocab_builder.get_word_index(word) for word in words]
    
    def documents_to_indices(self, documents: List[str]) -> List[List[int]]:
        """
        Convert list of document strings to lists of vocabulary indices.
        
        Args:
            documents (List[str]): List of document strings
            
        Returns:
            List[List[int]]: List of documents as token index lists
        """
        return [self.process_document(doc) for doc in documents]
    
    def indices_to_documents(self, document_indices: List[List[int]]) -> List[str]:
        """
        Convert lists of vocabulary indices back to document strings.
        
        Args:
            document_indices (List[List[int]]): List of documents as token index lists
            
        Returns:
            List[str]: List of reconstructed document strings
        """
        documents = []
        for doc_indices in document_indices:
            words = [self.vocab_builder.get_word_from_index(idx) for idx in doc_indices]
            documents.append(' '.join(words))
        return documents
    
    def get_document_stats(self, documents: Union[List[str], List[List[int]]]) -> dict:
        """
        Get statistics about processed documents.
        
        Args:
            documents: List of document strings or token index lists
            
        Returns:
            dict: Dictionary containing document statistics
        """
        if not documents:
            return {
                'num_documents': 0,
                'total_tokens': 0,
                'avg_doc_length': 0,
                'min_doc_length': 0,
                'max_doc_length': 0
            }
        
        # Convert to token lists if needed
        if isinstance(documents[0], str):
            doc_lengths = [len(self.tokenize_document(doc)) for doc in documents]
        else:
            doc_lengths = [len(doc) for doc in documents]
        
        total_tokens = sum(doc_lengths)
        
        return {
            'num_documents': len(documents),
            'total_tokens': total_tokens,
            'avg_doc_length': total_tokens / len(documents),
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths)
        }
    
    def filter_empty_documents(self, documents: List[str]) -> List[str]:
        """
        Filter out empty documents after tokenization.
        
        Args:
            documents (List[str]): List of document strings
            
        Returns:
            List[str]: List of non-empty documents
        """
        filtered_docs = []
        for doc in documents:
            tokens = self.tokenize_document(doc)
            if tokens:  # Only keep documents with at least one token
                filtered_docs.append(doc)
        return filtered_docs
    
    def __repr__(self) -> str:
        """String representation of DocumentProcessor."""
        return f"DocumentProcessor(vocab_size={self.vocab_builder.vocab_size})"