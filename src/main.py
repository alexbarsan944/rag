"""
Main module to run the Retrieval-Augmented Generation (RAG) pipeline.
"""

import logging
import mlflow
from src.data_loader import DataLoader
from src.elasticsearch_indexer import ElasticsearchIndexer
from src.retriever import Retriever
from src.response_generator import ResponseGenerator
from src.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class RAGPipeline:
    """Coordinates the RAG pipeline: indexing, retrieving, and response generation."""

    def __init__(self) -> None:
        """Initializes the RAG pipeline by loading data and setting up components."""
        logging.info("Loading knowledge base from CSV...")
        self.data_loader = DataLoader()
        self.df = self.data_loader.load_and_generate_embeddings()

        logging.info("Initializing Elasticsearch indexer...")
        self.indexer = ElasticsearchIndexer()
        self.indexer.create_index()
        
        logging.info("Indexing documents into Elasticsearch...")
        self.indexer.index_documents(self.df)

        self.response_generator = ResponseGenerator()
        logging.info("RAG Pipeline initialized successfully.")

    def run(self) -> None:
        """Runs the RAG pipeline interactively."""
        retriever = Retriever()

        while True:
            query = input("\nEnter your query (or type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting RAG pipeline. Goodbye!")
                break

            with mlflow.start_run():
                mlflow.set_experiment(CONFIG.MLFLOW_EXPERIMENT_NAME)
                mlflow.log_param("query", query)

                relevant_docs = retriever.search(query, top_k=3)
                response = self.response_generator.generate(query, relevant_docs)

                mlflow.log_metric("retrieved_docs", len(relevant_docs))
                mlflow.log_text("\n".join(relevant_docs), "retrieved_documents")
                mlflow.log_text(response, "generated_response")

            print("\nResponse:", response)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run()
