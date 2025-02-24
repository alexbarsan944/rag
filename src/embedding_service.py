"""
Module for generating embeddings using OpenAI API (openai>=1.0.0).
"""

import numpy as np
import openai
from openai import OpenAI
import logging
from src.config import CONFIG

logger = logging.getLogger(__name__)

client = OpenAI(api_key=CONFIG.OPENAI_API_KEY)


class EmbeddingService:
    """Service to generate embeddings for a given text."""

    def __init__(self, model: str = CONFIG.EMBEDDING_MODEL) -> None:
        """
        Args:
            model: The OpenAI embedding model to use.
        """
        self.model = model

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the provided text using OpenAI API v1.0+.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: The embedding vector as a NumPy array.
        """
        try:
            response = client.embeddings.create(model=self.model, input=[text])
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise RuntimeError("Failed to generate embedding.") from e
