"""
Module for generating embeddings using OpenAI API (openai>=1.0.0).
"""

import numpy as np
import openai
from openai import OpenAI
import logging
from typing import List, Dict, Any, Optional, Union
import time
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
        self.cache = {}  # Simple in-memory cache
        self.cache_hits = 0
        self.cache_misses = 0
        logger.debug(f"Initialized EmbeddingService with model {model}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the provided text using OpenAI API v1.0+.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray: The embedding vector as a NumPy array.
        """
        # Normalize the text to ensure consistent cache hits
        # (remove extra spaces and convert to lowercase)
        text_key = ' '.join(text.lower().split())
        
        # Check cache first
        if text_key in self.cache:
            self.cache_hits += 1
            logger.info(f"[EMBEDDING CACHE HIT] Hits: {self.cache_hits}, Misses: {self.cache_misses}")
            return self.cache[text_key]
            
        try:
            self.cache_misses += 1
            logger.info(f"[EMBEDDING CACHE MISS] Hits: {self.cache_hits}, Misses: {self.cache_misses}")
            response = client.embeddings.create(model=self.model, input=[text])
            embedding = response.data[0].embedding
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Cache the result
            self.cache[text_key] = embedding_array
            
            return embedding_array
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise RuntimeError("Failed to generate embedding.") from e

    def get_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 20,
        retry_limit: int = 3,
        backoff_time: float = 2.0,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches to optimize API usage.

        Args:
            texts: List of texts to generate embeddings for.
            batch_size: Number of texts to process in each API call (OpenAI limit is 2048).
            retry_limit: Number of times to retry on API errors.
            backoff_time: Initial time to wait before retrying, doubles on each retry.

        Returns:
            List of embedding vectors as NumPy arrays.
        """
        if not texts:
            return []

        results = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        for batch in batches:
            # Filter out texts that are already in cache
            to_process = []
            cached_indices = []
            
            for i, text in enumerate(batch):
                if text in self.cache:
                    cached_indices.append(i)
                else:
                    to_process.append(text)
            
            # If all texts in this batch are cached, skip the API call
            if not to_process:
                for text in batch:
                    results.append(self.cache[text])
                continue
                
            # Process the remaining texts with API call and retries
            retry_count = 0
            current_backoff = backoff_time
            
            while retry_count <= retry_limit:
                try:
                    response = client.embeddings.create(
                        model=self.model, 
                        input=to_process
                    )
                    
                    # Extract embeddings and put them in the right order
                    batch_results = [None] * len(batch)
                    
                    # Add cached embeddings first
                    for i, text_idx in enumerate(cached_indices):
                        batch_results[text_idx] = self.cache[batch[text_idx]]
                    
                    # Now add the new embeddings
                    for i, embedding_data in enumerate(response.data):
                        embedding = np.array(embedding_data.embedding, dtype=np.float32)
                        
                        # Find the position in the original batch
                        original_idx = next(j for j in range(len(batch)) 
                                             if j not in cached_indices and 
                                                batch[j] == to_process[i])
                        
                        batch_results[original_idx] = embedding
                        
                        # Cache the result
                        self.cache[to_process[i]] = embedding
                    
                    # Add all embeddings to results
                    results.extend(batch_results)
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count > retry_limit:
                        logger.error(f"Failed to get embeddings after {retry_limit} retries: {e}")
                        raise RuntimeError(f"Failed to generate embeddings batch after {retry_limit} retries.") from e
                    
                    logger.warning(f"Embeddings API error (attempt {retry_count}/{retry_limit}): {e}")
                    logger.info(f"Retrying in {current_backoff} seconds...")
                    time.sleep(current_backoff)
                    current_backoff *= 2  # Exponential backoff
        
        return results

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks of text and add them to the chunk data.
        
        Args:
            chunks: List of chunk dictionaries, each containing at least a 'text' key.
            
        Returns:
            The chunks with embeddings added under the 'embedding' key.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.get_embeddings_batch(texts)
        
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding.tolist()
            
        return chunks
