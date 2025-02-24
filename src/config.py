"""
Configuration module.

Contains global settings, including API keys and file paths.
"""

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def get_env_variable(name: str) -> str:
    """
    Retrieves an environment variable and raises an error if it's not set.

    Args:
        name: The name of the environment variable.
    """
    value: Optional[str] = os.getenv(name)
    if value is None or value.strip() == "":
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return value


@dataclass(frozen=True)
class Config:
    """Configuration dataclass for RAG pipeline."""

    OPENAI_API_KEY: str = field(
        default_factory=lambda: get_env_variable("OPENAI_API_KEY")
    )
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base.csv"
    FAISS_INDEX_PATH: str = "data/faiss_index.idx"
    KNOWLEDGE_BASE_WITH_VECTORS: str = "data/knowledge_base_with_vectors.csv"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    CHAT_MODEL: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.1


CONFIG = Config()
