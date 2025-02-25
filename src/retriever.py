from src.elasticsearch_indexer import ElasticsearchIndexer
from src.embedding_service import EmbeddingService

class Retriever:
    """Retrieves the top-k relevant texts from Elasticsearch."""

    def __init__(self) -> None:
        self.indexer = ElasticsearchIndexer()
        self.embedding_service = EmbeddingService()

    def search(self, query: str, top_k: int = 3):
        """Searches Elasticsearch for relevant documents."""
        query_vector = self.embedding_service.get_embedding(query)
        return self.indexer.search(query_vector, top_k)