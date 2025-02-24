import os
import faiss
import logging
from .config import CONFIG

logger = logging.getLogger(__name__)


class FaissIndexer:
    """Class for building and managing a FAISS index."""

    def __init__(self, df):
        """
        Args:
            df: The knowledge base DataFrame.
        """
        self.df = df

    def load_index(self) -> faiss.IndexFlatIP:
        """Loads the FAISS index from disk.

        Returns:
            faiss.IndexFlatIP: The loaded FAISS index.

        Raises:
            FileNotFoundError: If the FAISS index file does not exist.
            RuntimeError: If FAISS fails to load the index.
        """
        if not os.path.exists(CONFIG.FAISS_INDEX_PATH):
            logger.error(f"FAISS index file not found at: {CONFIG.FAISS_INDEX_PATH}")
            raise FileNotFoundError("FAISS index file does not exist.")

        try:
            logger.info(f"Loading FAISS index from: {CONFIG.FAISS_INDEX_PATH}")
            index = faiss.read_index(CONFIG.FAISS_INDEX_PATH)
            logger.info("FAISS index successfully loaded.")
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise RuntimeError("Failed to load FAISS index.") from e
