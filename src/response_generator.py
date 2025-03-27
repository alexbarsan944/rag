"""
Module for generating responses using OpenAI Chat models.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

from src.config import CONFIG

logger = logging.getLogger(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=CONFIG.OPENAI_API_KEY)


class ResponseGenerator:
    """Generates responses using a Chat model based on provided context."""

    def __init__(
        self,
        model_name: str = CONFIG.CHAT_MODEL,
        temperature: float = CONFIG.TEMPERATURE,
    ) -> None:
        """
        Args:
            model_name: The chat model to use.
            temperature: Temperature setting for response variability.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.cache = {}  # Cache for generated responses
        self.cache_hits = 0
        self.cache_misses = 0

    def generate(
        self, 
        query: str, 
        relevant_docs: Union[List[str], List[Dict[str, Any]]],
        include_metadata: bool = True,
    ) -> str:
        """Generates a response based on the context.

        Args:
            query: The user query.
            relevant_docs: A list of relevant documents (as context) or search results with metadata.
            include_metadata: Whether to include metadata in context (if available).

        Returns:
            The generated response as a string.
        """
        # Create a cache key based on query and context
        cache_key = (query, str(relevant_docs), include_metadata)
        
        # Check if we have a cached response
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.info(f"[RESPONSE CACHE HIT] Hits: {self.cache_hits}, Misses: {self.cache_misses}")
            return self.cache[cache_key]
        
        self.cache_misses += 1
        logger.info(f"[RESPONSE CACHE MISS] Hits: {self.cache_hits}, Misses: {self.cache_misses}")
        
        # Process the context based on type
        if relevant_docs and isinstance(relevant_docs[0], dict):
            context = self._format_context_with_metadata(relevant_docs, include_metadata)
        else:
            # Simple text list processing
            context = "\n\n".join(f"Document: {doc}" for doc in relevant_docs)
        
        # Create our system message with instructions
        system_message = (
            "You are an AI assistant that answers questions based on the provided context. "
            "Use only the information in the context to answer the question. "
            "If you don't know the answer based on the context, say so clearly - don't make up information."
        )
        
        # Build the user prompt with context
        user_prompt = (
            f"Use the following context to answer the query:\n\n"
            f"Context:\n{context}\n\nQuery: {query}"
        )
        
        logger.debug("Generating response for query: %s", query)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            result = response.choices[0].message.content
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
            
    def _format_context_with_metadata(
        self, 
        search_results: List[Dict[str, Any]], 
        include_metadata: bool
    ) -> str:
        """
        Formats search results including metadata into a structured context.
        
        Args:
            search_results: List of search result dictionaries.
            include_metadata: Whether to include metadata in the context.
            
        Returns:
            A formatted string with context information.
        """
        context_parts = []
        
        for i, result in enumerate(search_results):
            # Extract text and metadata
            text = result["text"]
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            
            # Format with or without metadata
            if include_metadata:
                source = metadata.get("source", "Unknown")
                title = metadata.get("title", "Untitled")
                
                context_part = (
                    f"Document {i+1} [Title: {title}, Source: {source}, "
                    f"Relevance: {score:.2f}]:\n{text}"
                )
            else:
                context_part = f"Document {i+1}:\n{text}"
                
            context_parts.append(context_part)
            
        return "\n\n".join(context_parts)
