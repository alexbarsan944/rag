"""
Module for BM25-based text retrieval.
"""

import logging
import os
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Union
from rank_bm25 import BM25Okapi
import pickle

from src.config import CONFIG

logger = logging.getLogger(__name__)

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function that splits on non-alphanumeric characters."""
    # Convert to lowercase and split on non-alphanumeric
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    """BM25-based retrieval component using the rank_bm25 library."""

    def __init__(self, index_path: str = "data/bm25_index.pkl") -> None:
        """
        Initialize the BM25 retriever.
        
        Args:
            index_path: Path to save/load the BM25 index.
        """
        self.index_path = index_path
        self.bm25 = None
        self.chunks = []
        self.metadata = []
    
    def build_index(self, df: pd.DataFrame) -> None:
        """
        Build a BM25 index from the dataframe.
        
        Args:
            df: DataFrame containing the text chunks and metadata.
        """
        logger.info("Building BM25 index...")
        
        # Store the raw text chunks and metadata
        self.chunks = []
        self.metadata = []
        tokenized_corpus = []
        
        for _, row in df.iterrows():
            # Extract text and tokenize
            text = row["text"]
            self.chunks.append(text)
            
            # Tokenize the text into words (BM25 requires tokenized corpus)
            # Using our simple tokenizer instead of NLTK
            tokens = simple_tokenize(text)
            tokenized_corpus.append(tokens)
            
            # Store metadata for each chunk
            chunk_metadata = {
                "source_id": row.get("source_id", -1),
                "chunk_id": row.get("chunk_id", -1),
                "title": row.get("title", "Unknown"),
                "source": row.get("source", "Unknown"),
                "chunk_strategy": row.get("chunk_strategy", "unknown"),
            }
            self.metadata.append(chunk_metadata)
        
        # Create the BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save the index for future use
        self._save_index()
        
        logger.info(f"BM25 index built with {len(self.chunks)} documents.")
    
    def _save_index(self) -> None:
        """Save the BM25 index and metadata to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save the BM25 model and metadata
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"BM25 index saved to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load the BM25 index from disk.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not os.path.exists(self.index_path):
            logger.warning(f"BM25 index file not found at: {self.index_path}")
            return False
            
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            
            logger.info(f"BM25 index loaded with {len(self.chunks)} documents.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 3,
        return_metadata: bool = False,
        score_threshold: Optional[float] = None,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Search the corpus using BM25 ranking.
        
        Args:
            query: The query string.
            top_k: Number of results to return.
            return_metadata: Whether to return full result objects with metadata.
            score_threshold: Minimum score threshold.
            
        Returns:
            Either a list of text strings or a list of dictionaries with text and metadata.
        """
        if self.bm25 is None:
            logger.error("BM25 index not loaded.")
            return []
        
        # Tokenize the query using our simple tokenizer
        tokenized_query = simple_tokenize(query)
        
        # Get scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k document indices
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        # Filter by threshold if specified
        if score_threshold is not None:
            top_n_indices = [idx for idx in top_n_indices if scores[idx] >= score_threshold]
        
        # Format results
        results = []
        for idx in top_n_indices:
            score = scores[idx]
            if return_metadata:
                result = {
                    "text": self.chunks[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx] if idx < len(self.metadata) else {}
                }
                results.append(result)
            else:
                results.append(self.chunks[idx])
        
        return results 