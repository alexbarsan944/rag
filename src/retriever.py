from src.faiss_indexer import FaissIndexer
from src.embedding_service import EmbeddingService
from typing import List, Dict, Any, Optional, Union

class Retriever:
    """Retrieves the top-k relevant texts using FAISS."""

    def __init__(self) -> None:
        self.indexer = FaissIndexer()
        self.embedding_service = EmbeddingService()
        self.indexer.load_index()

    def search(
        self, 
        query: str, 
        top_k: int = 3, 
        return_metadata: bool = False,
        score_threshold: Optional[float] = None,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Searches FAISS for relevant documents.
        
        Args:
            query: The user query string.
            top_k: Number of results to return.
            return_metadata: Whether to return full result objects with metadata.
            score_threshold: Minimum similarity score threshold (0-1).
            
        Returns:
            Either a list of text strings or a list of dictionaries with text and metadata.
        """
        # Get the query vector
        query_vector = self.embedding_service.get_embedding(query)
        
        # Search using the vector
        search_results = self.indexer.search(query_vector, top_k)
        
        # Apply score threshold if specified
        if score_threshold is not None:
            search_results = [r for r in search_results if r["score"] >= score_threshold]
        
        # Return either just the text or the full results with metadata
        if return_metadata:
            return search_results
        else:
            return [result["text"] for result in search_results]
            
    def search_with_reranking(
        self,
        query: str,
        initial_top_k: int = 10,
        final_top_k: int = 3,
    ) -> List[str]:
        """
        Performs a search with basic reranking to improve results.
        
        Args:
            query: The user query string.
            initial_top_k: Number of initial results to fetch for reranking.
            final_top_k: Number of results to return after reranking.
            
        Returns:
            A list of text strings.
        """
        # Get more results than we need for reranking
        search_results = self.search(query, top_k=initial_top_k, return_metadata=True)
        
        # If no results found, return empty list
        if not search_results:
            return []
            
        # Simple reranking by boosting results from same source
        # This could be replaced with a more sophisticated reranking method
        grouped_sources = {}
        for result in search_results:
            # Check if metadata exists and has source_id
            metadata = result.get("metadata", {})
            source_id = metadata.get("source_id", "unknown")
            
            if source_id not in grouped_sources:
                grouped_sources[source_id] = []
            grouped_sources[source_id].append(result)
        
        # If we have no valid metadata for grouping, just return the top results without reranking
        if not grouped_sources:
            return [result["text"] for result in search_results[:final_top_k]]
            
        # If we have multiple sources, prioritize diverse sources first
        reranked_results = []
        sources_list = list(grouped_sources.values())
        
        # Check if sources_list is empty to avoid max() error
        if not sources_list:
            return [result["text"] for result in search_results[:final_top_k]]
        
        # Take the top result from each source first
        max_source_items = max(len(source_results) for source_results in sources_list)
        for i in range(max_source_items):
            for source_results in sources_list:
                if i < len(source_results):
                    reranked_results.append(source_results[i])
                    
        # Take the top final_top_k results
        return [result["text"] for result in reranked_results[:final_top_k]]