"""
Module for text chunking strategies to optimize retrieval performance.
"""

import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles splitting of documents into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "sliding_window"
    ) -> None:
        """
        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
            chunk_strategy: Strategy to use for chunking (sliding_window, sentence, paragraph).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Splits the text into chunks based on the specified strategy.
        
        Args:
            text: The text to be chunked.
            metadata: Optional metadata to include with each chunk.
            
        Returns:
            List of dictionaries containing the chunks and their metadata.
        """
        if not text or text.strip() == "":
            logger.warning("Empty text provided for chunking")
            return []
            
        # Choose the appropriate chunking strategy
        if self.chunk_strategy == "sliding_window":
            chunks = self._sliding_window_chunks(text)
        elif self.chunk_strategy == "sentence":
            chunks = self._sentence_chunks(text)
        elif self.chunk_strategy == "paragraph":
            chunks = self._paragraph_chunks(text)
        else:
            logger.warning(f"Unknown chunking strategy: {self.chunk_strategy}. Using sliding window.")
            chunks = self._sliding_window_chunks(text)
            
        # Prepare the result with metadata
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "text": chunk,
                "chunk_strategy": self.chunk_strategy,
            }
            
            # Include original metadata with each chunk
            if metadata:
                chunk_data["metadata"] = metadata.copy()
                
            result.append(chunk_data)
            
        return result

    def _sliding_window_chunks(self, text: str) -> List[str]:
        """
        Creates chunks using a sliding window approach.
        
        Args:
            text: The text to be chunked.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # If this is not the first chunk and we're not at the end of the text,
            # include overlap with the previous chunk
            if start > 0:
                start = max(0, start - self.chunk_overlap)
                
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move to the next chunk, accounting for overlap
            start = end
            
        return chunks

    def _sentence_chunks(self, text: str) -> List[str]:
        """
        Chunks text by sentences, respecting the maximum chunk size.
        
        Args:
            text: The text to be chunked.
            
        Returns:
            List of text chunks.
        """
        # Simple sentence splitting - can be enhanced with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size, start a new chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap if possible
                current_chunk = self._get_overlap_text(current_chunk)
                
            current_chunk += sentence + " "
            
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks

    def _paragraph_chunks(self, text: str) -> List[str]:
        """
        Chunks text by paragraphs, respecting the maximum chunk size.
        
        Args:
            text: The text to be chunked.
            
        Returns:
            List of text chunks.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed the chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap if possible
                current_chunk = self._get_overlap_text(current_chunk)
                
            current_chunk += paragraph + "\n\n"
            
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
        
    def _get_overlap_text(self, text: str) -> str:
        """
        Gets the overlapping text from the end of a chunk.
        
        Args:
            text: The text from which to extract overlap.
            
        Returns:
            Text to include as overlap in the next chunk.
        """
        if len(text) <= self.chunk_overlap:
            return text
            
        return text[-self.chunk_overlap:] 