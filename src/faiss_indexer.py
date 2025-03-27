import os
import faiss
import numpy as np
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from .config import CONFIG

logger = logging.getLogger(__name__)


class FaissIndexer:
    """Class for building and managing a FAISS index."""

    def __init__(self):
        """Initialize the FAISS indexer."""
        self.index = None
        self.chunks = []
        self.metadata = []

    def build_index(self, df: pd.DataFrame) -> None:
        """Builds and saves a FAISS index from the DataFrame.
        
        Args:
            df: DataFrame containing the 'Vector' and 'Text' columns.
        """
        # Convert vectors to numpy array
        vectors = []
        self.chunks = []
        self.metadata = []
        
        for _, row in df.iterrows():
            vector = row["Vector"]
            if isinstance(vector, str):
                try:
                    vector = np.array(eval(vector), dtype=np.float32)
                except:
                    logger.warning(f"Could not parse vector: {vector[:100]}...")
                    continue
            elif isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            else:
                logger.warning(f"Unsupported vector type: {type(vector)}")
                continue
                
            vectors.append(vector)
            self.chunks.append(row["text"])
            
            # Store metadata for each chunk
            chunk_metadata = {
                "source_id": row.get("source_id", -1),
                "chunk_id": row.get("chunk_id", -1),
                "title": row.get("title", "Unknown"),
                "source": row.get("source", "Unknown"),
                "chunk_strategy": row.get("chunk_strategy", "unknown"),
            }
            self.metadata.append(chunk_metadata)
            
        if not vectors:
            logger.error("No valid vectors found for FAISS index.")
            raise ValueError("No valid vectors found for FAISS index.")
            
        # Convert to required format for FAISS
        vectors_np = np.vstack(vectors).astype('float32')
        
        # Normalize vectors for improved similarity search
        faiss.normalize_L2(vectors_np)
        
        # Create index - using IndexFlatIP for inner product similarity (cosine)
        vector_dimension = vectors_np.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)
        self.index.add(vectors_np)
        
        # Save index
        self._save_index()
        
        logger.info(f"FAISS index built with {len(vectors)} vectors.")

    def _save_index(self) -> None:
        """Save the index and associated metadata to disk."""
        # Make sure the directory exists
        os.makedirs(os.path.dirname(CONFIG.FAISS_INDEX_PATH), exist_ok=True)
        
        # Save the FAISS index
        logger.info(f"Saving FAISS index to {CONFIG.FAISS_INDEX_PATH}")
        faiss.write_index(self.index, CONFIG.FAISS_INDEX_PATH)
        
        # Save metadata in a separate file
        metadata_path = CONFIG.FAISS_INDEX_PATH + ".metadata.npz"
        np.savez_compressed(
            metadata_path,
            chunks=np.array(self.chunks, dtype=object),
            metadata=np.array(self.metadata, dtype=object)
        )
        logger.info(f"FAISS metadata saved to {metadata_path}")

    def load_index(self) -> bool:
        """Loads the FAISS index from disk.

        Returns:
            bool: True if the index was loaded successfully, False otherwise.
        """
        if not os.path.exists(CONFIG.FAISS_INDEX_PATH):
            logger.warning(f"FAISS index file not found at: {CONFIG.FAISS_INDEX_PATH}")
            return False

        try:
            # Load the FAISS index
            logger.info(f"Loading FAISS index from: {CONFIG.FAISS_INDEX_PATH}")
            self.index = faiss.read_index(CONFIG.FAISS_INDEX_PATH)
            
            # Load metadata
            metadata_path = CONFIG.FAISS_INDEX_PATH + ".metadata.npz"
            if os.path.exists(metadata_path):
                metadata_file = np.load(metadata_path, allow_pickle=True)
                self.chunks = metadata_file["chunks"].tolist()
                self.metadata = metadata_file["metadata"].tolist()
                logger.info(f"Loaded metadata for {len(self.chunks)} chunks")
            else:
                logger.warning(f"Metadata file not found at: {metadata_path}")
                
            logger.info("FAISS index successfully loaded.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
            
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches the FAISS index for the most similar vectors.
        
        Args:
            query_vector: The query vector.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing text and metadata for matches.
        """
        if self.index is None:
            logger.error("FAISS index not loaded.")
            return []
            
        # Ensure query vector is the right shape and type
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        
        # Normalize the query vector
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Return corresponding texts with metadata and scores
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                result = {
                    "text": self.chunks[idx],
                    "score": float(scores[0][i]),
                    "metadata": self.metadata[idx]
                }
                results.append(result)
                
        return results
