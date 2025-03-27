"""
Module for hybrid retrieval combining embedding-based and BM25 search.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
import pandas as pd

from src.retriever import Retriever
from src.bm25_retriever import BM25Retriever
from src.config import CONFIG

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines vector-based and lexical (BM25) search for improved retrieval."""

    def __init__(self) -> None:
        """Initialize both retrievers."""
        self.vector_retriever = Retriever()
        self.bm25_retriever = BM25Retriever()
        self._load_or_build_indices()
    
    def _load_or_build_indices(self) -> None:
        """Load existing indices or build them if they don't exist."""
        # Vector retriever index is loaded in its constructor
        
        # Load or build BM25 index
        if not self.bm25_retriever.load_index():
            logger.info("Building new BM25 index...")
            # Load the same data used for vector search
            from src.data_loader import DataLoader
            data_loader = DataLoader()
            df = data_loader.load_and_generate_embeddings()
            self.bm25_retriever.build_index(df)
    
    def search(
        self, 
        query: str, 
        top_k: int = 3,
        return_metadata: bool = True,
        vector_weight: float = 0.7,
        rerank: bool = True,
        dedup: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both vector and BM25 retrieval.
        
        Args:
            query: The query string.
            top_k: Number of final results to return.
            return_metadata: Whether to include metadata in results.
            vector_weight: Weight for vector search scores (0.0-1.0).
                           BM25 weight will be (1-vector_weight).
            rerank: Whether to apply reranking.
            dedup: Whether to remove duplicate results.
            
        Returns:
            List of dictionaries with text, score, and metadata.
        """
        # The number of results to get from each retriever (2x top_k)
        retrieval_k = top_k * 2
        
        # Get results from both retrievers
        if rerank:
            vector_results = self.vector_retriever.search_with_reranking(
                query, 
                initial_top_k=retrieval_k * 2, 
                final_top_k=retrieval_k
            )
            # Convert to dict format if not already
            if vector_results and not isinstance(vector_results[0], dict):
                vector_results = [{"text": r, "score": 1.0} for r in vector_results]
        else:
            vector_results = self.vector_retriever.search(
                query, 
                top_k=retrieval_k, 
                return_metadata=True
            )
        
        bm25_results = self.bm25_retriever.search(
            query, 
            top_k=retrieval_k, 
            return_metadata=True
        )
        
        # Combine results
        combined_results = self._combine_results(
            vector_results, 
            bm25_results, 
            vector_weight, 
            dedup
        )
        
        # Return top-k results
        return combined_results[:top_k]
    
    def _combine_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        vector_weight: float,
        dedup: bool
    ) -> List[Dict[str, Any]]:
        """
        Combine and normalize results from vector and BM25 search.
        
        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            vector_weight: Weight for vector search (0.0-1.0).
            dedup: Whether to remove duplicates.
            
        Returns:
            Combined and ranked results.
        """
        bm25_weight = 1.0 - vector_weight
        
        # Normalize BM25 scores (they can be any positive number)
        if bm25_results:
            max_bm25_score = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                if max_bm25_score > 0:
                    r["score"] = r["score"] / max_bm25_score  # Normalize to 0-1
        
        # Create a unified dictionary of results
        # Use text as key to detect duplicates
        results_dict = {}
        
        # Add vector results
        for result in vector_results:
            # Weighted score
            result["score"] = result["score"] * vector_weight
            result["sources"] = ["vector"]
            results_dict[result["text"]] = result
        
        # Add or merge BM25 results
        for result in bm25_results:
            text = result["text"]
            if text in results_dict and dedup:
                # Combine scores if this is a duplicate
                results_dict[text]["score"] += result["score"] * bm25_weight
                results_dict[text]["sources"].append("bm25")
            else:
                # Add as new result
                result["score"] = result["score"] * bm25_weight
                result["sources"] = ["bm25"]
                results_dict[text] = result
        
        # Convert back to list and sort by score
        combined_results = list(results_dict.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results 