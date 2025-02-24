"""
Main module to run the Retrieval-Augmented Generation (RAG) pipeline.
"""

import logging
from src.data_loader import DataLoader
from src.faiss_indexer import FaissIndexer
from src.retriever import Retriever
from src.response_generator import ResponseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class RAGPipeline:
    """Coordinates the RAG pipeline: indexing, retrieving, and response generation."""

    def __init__(self) -> None:
        """Initializes the RAG pipeline by loading data and setting up components."""
        self.data_loader = DataLoader()
        self.df = self.data_loader.load()
        self.indexer = FaissIndexer(self.df)
        self.response_generator = ResponseGenerator()
        self.index = None

    def build_or_load_index(self) -> None:
        """Loads the FAISS index if available, otherwise builds a new one."""
        try:
            self.index = self.indexer.load_index()
            logging.info("Successfully loaded FAISS index.")
        except FileNotFoundError:
            logging.info("FAISS index not found. Building a new one...")
            self.index = self.indexer.build_index()
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            logging.info("Building a new FAISS index as a fallback...")
            self.index = self.indexer.build_index()

    def run(self) -> None:
        """Runs the RAG pipeline interactively."""
        self.build_or_load_index()
        retriever = Retriever(self.df, self.index)

        while True:
            query = input("\nEnter your query (or type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting RAG pipeline. Goodbye!")
                break

            relevant_docs = retriever.search(query)
            response = self.response_generator.generate(query, relevant_docs)
            print("\nResponse:", response)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run()
