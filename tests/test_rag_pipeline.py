import json  # ✅ Added missing import
import re
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from src.embedding_service import EmbeddingService
from src.elasticsearch_indexer import ElasticsearchIndexer
from src.retriever import Retriever
from src.response_generator import ResponseGenerator

@pytest.fixture
def test_df():
    """
    Creates a sample DataFrame with text data for retrieval testing.
    """
    data = {
        "Text": [
            "Neural networks were introduced in 1943 by McCulloch and Pitts.",
            "Deep learning is a subset of machine learning.",
        ],
        "Vector": [json.dumps([0.1] * 1536), json.dumps([0.2] * 1536)] 
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_elasticsearch_search():
    """Mock Elasticsearch search results to return relevant texts."""
    return {
        "hits": {
            "hits": [
                {"_source": {"text": "Neural networks were introduced in 1943 by McCulloch and Pitts."}}
            ]
        }
    }

@pytest.fixture
def dummy_retriever(test_df, monkeypatch, mock_elasticsearch_search):
    """
    Returns a Retriever instance that uses a mocked Elasticsearch search.
    """
    monkeypatch.setattr(EmbeddingService, "get_embedding", lambda self, text: np.array([0.1] * 1536))

    monkeypatch.setattr(ElasticsearchIndexer, "search", lambda self, query_vector, top_k: 
        [hit["_source"]["text"] for hit in mock_elasticsearch_search["hits"]["hits"]]
    )

    return Retriever()

def test_retrieval(dummy_retriever):
    """Tests that the retriever fetches the correct document."""
    query = "Who introduced neural networks?"
    results = dummy_retriever.search(query, top_k=1)
    assert "McCulloch and Pitts" in results[0], "Expected context not found"
