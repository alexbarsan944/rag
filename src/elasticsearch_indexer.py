import json
from elasticsearch import Elasticsearch
import logging
from src.config import CONFIG

logger = logging.getLogger(__name__)

class ElasticsearchIndexer:
    """Handles indexing and retrieval using Elasticsearch."""

    def __init__(self) -> None:
        """Initialize Elasticsearch connection with secure API authentication."""
        self.es = Elasticsearch(
            CONFIG.ELASTICSEARCH_HOST,
            headers={"Authorization": f"ApiKey {CONFIG.ELASTICSEARCH_API_KEY}"},
        )
        self.index_name = CONFIG.ELASTICSEARCH_INDEX

    def create_index(self) -> None:
        """Creates an Elasticsearch index for the knowledge base if it does not exist."""
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "text": {"type": "text"},
                                "embedding": {"type": "dense_vector", "dims": 1536},
                            }
                        }
                    },
                )
                logger.info(f"Elasticsearch index '{self.index_name}' created.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def get_existing_document_ids(self, ids):
        """Batch check for existing documents using Elasticsearch _mget API."""
        response = self.es.mget(index=self.index_name, body={"ids": ids})
        return {doc["_id"] for doc in response["docs"] if doc["found"]}

    def index_documents(self, df) -> None:
        """Indexes new or modified documents into Elasticsearch from the DataFrame."""
        doc_ids = df.index.astype(str).tolist()
        existing_ids = self.get_existing_document_ids(doc_ids)

        for idx, row in df.iterrows():
            doc_id = str(idx)
            if doc_id in existing_ids:
                logger.info(f"Skipping document {idx}: Already indexed.")
                continue

            try:
                vector = json.loads(row["Vector"]) if isinstance(row["Vector"], str) else row["Vector"]
                if not isinstance(vector, list) or not all(isinstance(v, (float, int)) for v in vector):
                    logger.error(f"Skipping document {idx}: Invalid embedding format {vector}")
                    continue

                self.es.index(
                    index=self.index_name,
                    id=doc_id,
                    body={"text": row["Text"], "embedding": vector},
                )
                logger.info(f"Indexed new document {idx}.")
            except Exception as e:
                logger.error(f"Error indexing document {idx}: {e}")

        logger.info(f"Document indexing process completed for '{self.index_name}'.")

    def search(self, query_vector, top_k=3):
        """Searches for relevant documents using vector similarity."""
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_vector},
                            },
                        }
                    },
                },
            )
            return [hit["_source"]["text"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
