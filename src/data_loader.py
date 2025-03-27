import pandas as pd
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from .embedding_service import EmbeddingService
from .text_chunker import TextChunker
from .config import CONFIG

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, 
                 csv_path: str = CONFIG.KNOWLEDGE_BASE_PATH, 
                 output_json_path: str = CONFIG.KNOWLEDGE_BASE_EMBEDDINGS_PATH,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 chunk_strategy: str = "paragraph") -> None:
        """
        Initialize the data loader with file paths and chunking settings.
        
        Args:
            csv_path: Path to the CSV knowledge base.
            output_json_path: Path to save embeddings and chunks.
            chunk_size: Maximum size of each document chunk.
            chunk_overlap: Overlap between consecutive chunks.
            chunk_strategy: Strategy to use for chunking (sliding_window, sentence, paragraph).
        """
        self.csv_path = csv_path
        self.output_json_path = output_json_path
        self.embedding_service = EmbeddingService()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy
        )

    def load_and_generate_embeddings(self) -> pd.DataFrame:
        """
        Loads the knowledge base. If the JSON file with embeddings exists, it loads that.
        Otherwise, it loads the original CSV, generates embeddings for missing values,
        saves the new data to JSON, and returns the DataFrame.
        """
        # Check if processed file exists
        if os.path.exists(self.output_json_path):
            logger.info(f"Loading existing embeddings file: {self.output_json_path}")
            return self.load_processed_file()

        # Load the raw CSV and process it
        logger.info(f"Loading and processing CSV file: {self.csv_path}")
        return self.process_raw_file()

    def load_processed_file(self) -> pd.DataFrame:
        """
        Load the pre-processed JSON file with embeddings.
        """
        try:
            df = pd.read_json(self.output_json_path)
            
            # Check if chunking exists in the data
            if "chunk_id" not in df.columns:
                logger.info("Previously processed file doesn't have chunking. Re-processing...")
                return self.process_raw_file()
                
            # Ensure vector format is correct
            df["Vector"] = df["Vector"].apply(self._fix_vector_format)
            
            # Check for missing embeddings
            missing_embeddings = df["Vector"].isna() | df["Vector"].apply(lambda x: x is None or len(x) == 0)
            
            if missing_embeddings.any():
                logger.info(f"Regenerating {missing_embeddings.sum()} missing embeddings...")
                chunks_to_process = df[missing_embeddings].to_dict('records')
                processed_chunks = self.embedding_service.embed_chunks(chunks_to_process)
                
                for chunk in processed_chunks:
                    idx = df[(df["chunk_id"] == chunk["chunk_id"]) & 
                             (df["source_id"] == chunk["source_id"])].index[0]
                    df.loc[idx, "Vector"] = chunk["embedding"]
                
                df.to_json(self.output_json_path, orient="records", indent=4)
                logger.info(f"Updated embeddings saved in: {self.output_json_path}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading processed file: {e}")
            logger.info("Processing raw file instead.")
            return self.process_raw_file()

    def process_raw_file(self) -> pd.DataFrame:
        """
        Process the raw CSV file, apply chunking, and generate embeddings.
        """
        try:
            # Load the CSV file
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"Knowledge base CSV file not found at: {self.csv_path}")
                
            df = pd.read_csv(self.csv_path)
            
            # Verify the CSV has the required 'Text' column
            if "Text" not in df.columns:
                logger.error(f"CSV file must contain a 'Text' column. Found columns: {df.columns.tolist()}")
                raise ValueError("Missing required 'Text' column in CSV file")
                
            all_chunks = []
            
            # Process each document, chunk it, and prepare for embedding
            for idx, row in df.iterrows():
                # Create metadata for this document
                metadata = {
                    "source_id": idx,
                    "title": row.get("Title", f"Document {idx}"),
                    "source": row.get("Source", "Unknown"),
                }
                
                # Chunk the text
                document_chunks = self.text_chunker.chunk_text(row["Text"], metadata)
                
                for chunk in document_chunks:
                    chunk_entry = {
                        "text": chunk["text"],
                        "source_id": metadata["source_id"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_strategy": chunk["chunk_strategy"],
                        "title": metadata["title"],
                        "source": metadata["source"],
                    }
                    all_chunks.append(chunk_entry)
            
            # Convert to DataFrame
            chunks_df = pd.DataFrame(all_chunks)
            
            # Generate embeddings in batch
            logger.info(f"Generating embeddings for {len(chunks_df)} chunks...")
            chunk_dicts = chunks_df.to_dict('records')
            processed_chunks = self.embedding_service.embed_chunks(chunk_dicts)
            
            # Convert back to DataFrame with embeddings
            result_df = pd.DataFrame(processed_chunks)
            result_df.rename(columns={"embedding": "Vector"}, inplace=True)
            
            # Save to JSON
            result_df.to_json(self.output_json_path, orient="records", indent=4)
            logger.info(f"Processed data with embeddings saved in: {self.output_json_path}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing raw file: {e}")
            raise

    def _fix_vector_format(self, val):
        """Ensure each value in 'Vector' is valid JSON."""
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list) and all(isinstance(x, (float, int)) for x in parsed):
                    return parsed
                logger.warning(f"Corrupted JSON in 'Vector' column: {val[:100]}...")
                return None
            except json.JSONDecodeError:
                logger.warning(f"Skipping corrupted JSON in 'Vector' column: {val[:100]}...")
                return None
        elif isinstance(val, list) or isinstance(val, np.ndarray):
            return list(val)
        return None
