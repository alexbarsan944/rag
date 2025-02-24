"""
Module for retrieving relevant texts using the FAISS index.
"""

import numpy as np
import faiss
import logging
from typing import List

import pandas as pd

from src.embedding_service import EmbeddingService
from src.config import CONFIG

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves the top-k relevant texts from the FAISS index."""

    def __init__(self, df: pd.DataFrame, index: faiss.IndexFlatIP) -> None:
        """
        Args:
            df: The knowledge base DataFrame.
            index: A pre-built FAISS index.
        """
        self.df = df
        self.index = index
        self.embedding_service = EmbeddingService()

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Searches the FAISS index for the query.

        Args:
            query: The input query string.
            top_k: Number of top results to return.

        Returns:
            A list of relevant text strings.
        """
        query_vector = self.embedding_service.get_embedding(query)
        # Normalize the query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        results = [
            self.df.iloc[idx]["Text"] for idx in indices[0] if idx < len(self.df)
        ]
        logger.debug("Search results retrieved for query: %s", query)
        return results
