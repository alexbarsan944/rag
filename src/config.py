from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


def get_env_variable(name: str) -> str:
    """Retrieves an environment variable and raises an error if it's not set."""
    value: Optional[str] = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return value


@dataclass(frozen=True)
class Config:
    """Configuration dataclass for RAG pipeline."""

    # API Configuration
    OPENAI_API_KEY: str = field(
        default_factory=lambda: get_env_variable("OPENAI_API_KEY")
    )
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    )
    CHAT_MODEL: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    )
    TEMPERATURE: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", 0.1))
    )

    # File Paths
    KNOWLEDGE_BASE_PATH: str = field(default="data/knowledge_base.csv")
    KNOWLEDGE_BASE_EMBEDDINGS_PATH: str = field(
        default="data/knowledge_base_with_vectors.json"
    )
    FAISS_INDEX_PATH: str = field(default="data/faiss_index.bin")

    # Chunking Strategy Settings
    DEFAULT_CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    )
    DEFAULT_CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
    )
    DEFAULT_CHUNK_STRATEGY: str = field(
        default_factory=lambda: os.getenv("DEFAULT_CHUNK_STRATEGY", "paragraph")
    )
    
    # Embedding Settings
    EMBEDDING_BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
    )
    EMBEDDING_RETRY_LIMIT: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_RETRY_LIMIT", "3"))
    )

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "")
    )
    MLFLOW_EXPERIMENT_NAME: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "RAG_Experiment")
    )


# Initialize Config
CONFIG = Config()
