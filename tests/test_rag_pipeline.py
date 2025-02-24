import re
import numpy as np
import pandas as pd
import pytest

from src.embedding_service import EmbeddingService
from src.faiss_indexer import FaissIndexer
from src.retriever import Retriever
from src.response_generator import ResponseGenerator

# --- Dummy Implementations for Testing ---


def dummy_get_embedding(text: str, model: str = "dummy") -> np.ndarray:
    """
    Dummy embedding function that returns a constant vector.
    This makes all texts appear identical for similarity search.
    """
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class DummyChatOpenAI:
    """
    Dummy chat model that extracts the context from the prompt and echoes it.
    This ensures that the response is completely based on the retrieved context.
    """

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt: str) -> str:
        """
        Extracts the context from the prompt and normalizes it by replacing newlines with spaces.
        Expected prompt format:
        "Use the following context to answer the query:

        Context: {context}

        Query: {query}"
        """
        match = re.search(r"Context:\s*(.*?)\s*Query:", prompt, re.DOTALL)
        if match:
            context = match.group(1).strip()
            normalized_context = " ".join(context.split())
            return normalized_context
        return "No context provided."


# --- Pytest Fixtures ---


@pytest.fixture
def test_df() -> pd.DataFrame:
    """
    Fixture that returns a small test DataFrame based on AI-related facts.
    """
    data = {
        "Text": [
            "The concept of neural networks was first introduced in the 1940s by Warren McCulloch and Walter Pitts.",
            "Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow in 2014.",
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_index(test_df, monkeypatch) -> any:
    """
    Build a FAISS index using the test DataFrame and dummy embeddings.
    Monkeypatch the EmbeddingService.get_embedding method and the index's search method.
    """
    # Patch the get_embedding method to use the dummy embedding.
    monkeypatch.setattr(EmbeddingService, "get_embedding", dummy_get_embedding)

    indexer = FaissIndexer(test_df)
    index = indexer.load_index()

    def dummy_search(query_vector, top_k):
        distances = np.array([[1.0] * top_k])
        indices = np.array([[0] * top_k])
        return distances, indices

    monkeypatch.setattr(index, "search", dummy_search)
    return index


@pytest.fixture
def dummy_retriever(test_df, dummy_index, monkeypatch) -> Retriever:
    """
    Return a Retriever instance that uses dummy embeddings.
    """
    monkeypatch.setattr(EmbeddingService, "get_embedding", dummy_get_embedding)
    return Retriever(test_df, dummy_index)


@pytest.fixture
def dummy_response_generator(monkeypatch) -> ResponseGenerator:
    """
    Return a ResponseGenerator instance that uses the DummyChatOpenAI.
    """
    monkeypatch.setattr("src.response_generator.ChatOpenAI", DummyChatOpenAI)
    return ResponseGenerator()


# --- Test Cases ---


def test_retrieval(dummy_retriever: Retriever):
    """
    Test that the retriever pulls the expected context from the dataset.

    With dummy embeddings and our patched search method, the top result should be the first row.
    """
    query = "Who introduced neural networks?"
    results = dummy_retriever.search(query, top_k=1)
    if isinstance(results, str):
        results = [results]
    assert isinstance(results, list)
    assert len(results) == 1
    assert (
        "Warren McCulloch and Walter Pitts" in results[0]
    ), "Expected context not found in retrieval results."


def test_generation(
    dummy_retriever: Retriever, dummy_response_generator: ResponseGenerator
):
    """
    Test that the response generator uses the retrieved context.

    With the dummy chat model that echoes the context, the final response should include known facts.
    """
    query = "Who introduced neural networks?"
    results = dummy_retriever.search(query, top_k=1)
    if isinstance(results, str):
        results = [results]
    response = dummy_response_generator.generate(query, results)
    assert (
        "Warren McCulloch and Walter Pitts" in response
    ), "The generated response did not use the expected context."
