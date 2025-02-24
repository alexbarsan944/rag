import numpy as np
import pytest
from src.embedding_service import EmbeddingService


def dummy_embedding_create(model, input):
    """Mock function that mimics the OpenAI API embedding response."""
    return {
        "object": "list",
        "data": [{"embedding": np.random.rand(1536).tolist(), "index": 0}],
        "model": model,
    }


def test_get_embedding(monkeypatch):
    """
    Test that get_embedding returns the expected dummy vector.

    This test monkeypatches openai.embeddings.create to always return a dummy
    embedding. The expected numpy array is then verified.
    """
    monkeypatch.setattr("openai.embeddings.create", dummy_embedding_create)

    service = EmbeddingService()
    text = "Test text"
    vector = service.get_embedding(text)

    assert isinstance(vector, np.ndarray), "Returned embedding should be a numpy array."
    assert vector.shape == (1536,), f"Expected shape (1536,), but got {vector.shape}"
