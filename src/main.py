"""
Main module to run the Retrieval-Augmented Generation (RAG) pipeline.
"""

import logging
import os
import argparse
from src.data_loader import DataLoader
from src.faiss_indexer import FaissIndexer
from src.retriever import Retriever
from src.hybrid_retriever import HybridRetriever
from src.response_generator import ResponseGenerator
from src.config import CONFIG

# Try to import MLflow, but provide a fallback if it's not available
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Create a minimal mock implementation
    class MockMLflow:
        @staticmethod
        def set_tracking_uri(*args, **kwargs):
            pass
            
        @staticmethod
        def set_experiment(*args, **kwargs):
            pass
            
        @staticmethod
        def start_run(*args, **kwargs):
            class MockContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockContextManager()
            
        @staticmethod
        def log_param(*args, **kwargs):
            pass
            
        @staticmethod
        def log_metric(*args, **kwargs):
            pass
            
        @staticmethod
        def log_text(*args, **kwargs):
            pass
    
    mlflow = MockMLflow()
    logger = logging.getLogger(__name__)
    logger.warning("MLflow not available. Experiment tracking disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Coordinates the RAG pipeline: indexing, retrieving, and response generation."""

    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 chunk_strategy: str = "paragraph") -> None:
        """
        Initializes the RAG pipeline by loading data and setting up components.
        
        Args:
            chunk_size: Maximum size of each document chunk.
            chunk_overlap: Overlap between consecutive chunks.
            chunk_strategy: Strategy to use for chunking (sliding_window, sentence, paragraph).
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize data loader with chunking settings
        logger.info(f"Loading knowledge base with {chunk_strategy} chunking strategy...")
        self.data_loader = DataLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy
        )
        self.df = self.data_loader.load_and_generate_embeddings()

        logger.info("Initializing FAISS indexer...")
        self.indexer = FaissIndexer()
        
        # Check if FAISS index exists, rebuild if not
        if not os.path.exists(CONFIG.FAISS_INDEX_PATH) or not self.indexer.load_index():
            logger.info("Building new FAISS index...")
            self.indexer.build_index(self.df)
        else:
            logger.info("FAISS index loaded successfully.")

        self.response_generator = ResponseGenerator()
        
        # Query response cache to avoid reprocessing identical queries
        self.query_cache = {}
        
        logger.info("RAG Pipeline initialized successfully.")

    def run(self, 
            retrieval_method: str = "vector",
            use_reranking: bool = True, 
            include_metadata: bool = True,
            vector_weight: float = 0.7) -> None:
        """
        Runs the RAG pipeline interactively.
        
        Args:
            retrieval_method: Which retrieval method to use: "vector", "bm25", or "hybrid"
            use_reranking: Whether to use reranking for improved search results.
            include_metadata: Whether to include metadata in the generated response.
            vector_weight: Weight to give vector results in hybrid retrieval (0.0-1.0)
        """
        # Initialize appropriate retriever based on method
        if retrieval_method == "hybrid":
            logger.info("Initializing hybrid retriever (vector + BM25)...")
            retriever = HybridRetriever()
        else:
            # Default to vector retriever for both "vector" and "bm25" options
            # The "bm25" option will be handled separately in _retrieve_documents
            retriever = Retriever()
        
        logger.info(f"Starting interactive session (method: {retrieval_method}, reranking: {use_reranking}, metadata: {include_metadata})")
        if retrieval_method == "hybrid":
            logger.info(f"Hybrid search vector weight: {vector_weight}, BM25 weight: {1-vector_weight}")

        while True:
            try:
                query = input("\nEnter your query (or type 'exit' to quit): ").strip()
                if query.lower() == "exit":
                    print("Exiting RAG pipeline. Goodbye!")
                    break
                    
                if not query:
                    print("Please enter a query.")
                    continue

                # Check if the query is in the cache
                cache_key = (query, retrieval_method, use_reranking, include_metadata, vector_weight)
                if cache_key in self.query_cache:
                    logger.info("Using cached response for repeated query.")
                    print("\n[CACHE HIT] Using cached response to save API calls.")
                    response = self.query_cache[cache_key]
                    print("\nResponse:", response)
                    continue

                # Check if MLflow is configured
                use_mlflow = CONFIG.MLFLOW_TRACKING_URI != "" and MLFLOW_AVAILABLE
                
                if use_mlflow:
                    logger.debug("MLflow logging enabled")
                    mlflow.set_tracking_uri(CONFIG.MLFLOW_TRACKING_URI)
                    with mlflow.start_run():
                        mlflow.set_experiment(CONFIG.MLFLOW_EXPERIMENT_NAME)
                        mlflow.log_param("query", query)
                        mlflow.log_param("retrieval_method", retrieval_method)
                        mlflow.log_param("use_reranking", use_reranking)
                        mlflow.log_param("include_metadata", include_metadata)
                        
                        relevant_docs = self._retrieve_documents(retriever, query, retrieval_method, 
                                                               use_reranking, vector_weight)
                        
                        if not relevant_docs:
                            response = "I couldn't find any relevant information to answer your query."
                        else:
                            response = self.response_generator.generate(
                                query, 
                                relevant_docs, 
                                include_metadata=include_metadata
                            )
                        
                        mlflow.log_metric("retrieved_docs", len(relevant_docs))
                        mlflow.log_text(str(relevant_docs), "retrieved_documents")
                        mlflow.log_text(response, "generated_response")
                else:
                    relevant_docs = self._retrieve_documents(retriever, query, retrieval_method,
                                                           use_reranking, vector_weight)
                    
                    if not relevant_docs:
                        response = "I couldn't find any relevant information to answer your query."
                    else:
                        response = self.response_generator.generate(
                            query, 
                            relevant_docs, 
                            include_metadata=include_metadata
                        )
                
                # Cache the response
                self.query_cache[cache_key] = response
                print(f"\n[CACHE MISS] Query cached for future use. Cache size: {len(self.query_cache)}")

                print("\nResponse:", response)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nAn error occurred: {e}")
                print("Please try a different query or check the logs for details.")
    
    def _retrieve_documents(self, retriever, query, retrieval_method, use_reranking, vector_weight=0.7):
        """Helper method to retrieve documents with different methods."""
        if retrieval_method == "hybrid":
            return retriever.search(
                query,
                top_k=3,
                vector_weight=vector_weight,
                rerank=use_reranking
            )
        elif retrieval_method == "bm25":
            # Special case - we need to import and use BM25Retriever
            from src.bm25_retriever import BM25Retriever
            bm25_retriever = BM25Retriever()
            if not bm25_retriever.load_index():
                logger.info("Building BM25 index for the first time...")
                bm25_retriever.build_index(self.df)
            return bm25_retriever.search(
                query,
                top_k=3,
                return_metadata=True
            )
        else:  # Default vector retrieval
            if use_reranking:
                return retriever.search_with_reranking(
                    query, 
                    initial_top_k=10, 
                    final_top_k=3
                )
            else:
                return retriever.search(
                    query, 
                    top_k=3, 
                    return_metadata=True,
                    score_threshold=0.5
                )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Maximum size of each document chunk (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200,
        help="Overlap between consecutive chunks (default: 200)"
    )
    parser.add_argument(
        "--chunk-strategy", 
        type=str, 
        default="paragraph",
        choices=["sliding_window", "sentence", "paragraph"],
        help="Strategy to use for chunking (default: paragraph)"
    )
    parser.add_argument(
        "--no-reranking", 
        action="store_true",
        help="Disable reranking of search results"
    )
    parser.add_argument(
        "--no-metadata", 
        action="store_true",
        help="Exclude metadata from response generation"
    )
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="vector",
        choices=["vector", "bm25", "hybrid"],
        help="Retrieval method to use (default: vector)"
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.7,
        help="Weight for vector results in hybrid search (default: 0.7)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.chunk_strategy
    )
    
    pipeline.run(
        retrieval_method=args.retrieval_method,
        use_reranking=not args.no_reranking,
        include_metadata=not args.no_metadata,
        vector_weight=args.vector_weight
    )
