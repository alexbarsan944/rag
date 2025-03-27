# Retrieval-Augmented Generation (RAG) System Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Concepts](#basic-concepts)
   - [What is RAG?](#what-is-rag)
   - [What are Embeddings?](#what-are-embeddings)
     - [Technical Deep Dive: How Embeddings Work](#technical-deep-dive-how-embeddings-work)
   - [Vector Search](#vector-search)
     - [Technical Deep Dive: How Vector Search Works](#technical-deep-dive-how-vector-search-works)
3. [System Overview](#system-overview)
4. [Components](#components)
   - [Configuration (`config.py`)](#configuration)
   - [Data Loading (`data_loader.py`)](#data-loading)
   - [Embedding Service (`embedding_service.py`)](#embedding-service)
   - [Text Chunking (`text_chunker.py`)](#text-chunking)
     - [Technical Deep Dive: Chunking Algorithms](#technical-deep-dive-chunking-algorithms)
   - [Vector Storage & Search (`faiss_indexer.py`)](#vector-storage-search)
   - [BM25 Retrieval (`bm25_retriever.py`)](#bm25-retrieval)
     - [Technical Deep Dive: How BM25 Works](#technical-deep-dive-how-bm25-works)
   - [Hybrid Retrieval (`hybrid_retriever.py`)](#hybrid-retrieval)
   - [Retrieval (`retriever.py`)](#retrieval)
     - [Technical Deep Dive: Reranking](#technical-deep-dive-reranking)
   - [Response Generation (`response_generator.py`)](#response-generation)
   - [Main Pipeline (`main.py`)](#main-pipeline)
5. [System Optimizations](#system-optimizations)
   - [Memory Caching](#memory-caching)
   - [Batch Processing](#batch-processing)
   - [Reranking](#reranking)
   - [Hybrid Search](#hybrid-search)
6. [Running the System](#running-the-system)
7. [Advanced Topics](#advanced-topics)
   - [Performance Tuning](#performance-tuning)
   - [Cost Optimization](#cost-optimization)

## Introduction

This documentation explains our Retrieval-Augmented Generation (RAG) system from the ground up. Whether you're new to the field or an experienced data scientist, you'll find information about how the system works and why it's designed this way.

## Basic Concepts

### What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture that combines two key capabilities:

1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Creating natural language responses based on the retrieved information

Unlike traditional language models that only use their internal knowledge, RAG systems can access external knowledge sources, making them more accurate, up-to-date, and transparent.

For a junior data scientist: Think of it as giving an AI assistant the ability to "look up" information before answering questions, rather than just relying on what it already knows.

### What are Embeddings?

Embeddings are numerical representations of text (or other data) that capture semantic meaning. They translate words, sentences, or documents into vectors (lists of numbers) in a high-dimensional space where similar meanings are positioned close together.

For example, the phrases "I love machine learning" and "I enjoy AI" would be positioned closer together in the embedding space than "I love machine learning" and "The weather is nice today".

**How it works in simple terms:**
1. We take a piece of text
2. Pass it through a special AI model (like OpenAI's embedding models)
3. The model outputs a long list of numbers (typically hundreds or thousands)
4. These numbers represent the text's meaning in a mathematical space

In our system, we create embeddings for both:
- The documents in our knowledge base
- The user's questions

This allows us to find documents that are semantically similar to the questions.

#### Technical Deep Dive: How Embeddings Work

Let's dive deeper into how embedding models like OpenAI's text-embedding-ada-002 actually work:

1. **Tokenization**:
   - The raw text is first split into tokens (subword units)
   - For example, "machine learning" might become ["machine", "learning"]
   - Complex words are broken down further (e.g., "unhappy" → ["un", "happy"])
   - Each token is assigned a unique integer ID according to the model's vocabulary
   - Special tokens are added (like [START], [END], or [PAD] tokens)
   
   **Example**: The phrase "AI is transforming healthcare" might be tokenized as:
   ```
   ["AI", " is", " transform", "ing", " health", "care"]
   ```
   And converted to token IDs:
   ```
   [2781, 318, 5342, 1089, 7893, 5083]
   ```

2. **Token Embedding**:
   - Each token ID is mapped to an initial vector in a learned embedding table
   - This creates a sequence of dense vectors (e.g., 768 dimensions per token)
   - These initial embeddings only capture word-level information
   
   **Example**: The token "AI" (ID: 2781) might be mapped to a vector:
   ```
   [0.021, -0.412, 0.781, ..., -0.156]  # 768 values
   ```

3. **Transformer Architecture Processing**:
   - The token embeddings are passed through multiple transformer layers
   - Each transformer layer consists of:
     a. **Multi-head Self-Attention**: Allows each token to "attend" to all other tokens
        - Each token's representation is updated based on its relationships with other tokens
        - The "self-attention" mechanism calculates attention scores between all pairs of tokens
        - Multiple "heads" allow the model to capture different types of relationships simultaneously
     b. **Feed-Forward Networks**: Further process each token's representation
     c. **Layer Normalization and Residual Connections**: Stabilize and improve training
   
   **Example**: For our phrase "AI is transforming healthcare":
   - The token "healthcare" might strongly attend to "AI" (high attention score)
   - The token "transforming" might attend more to "healthcare" than to "is"
   - After the first transformer layer, the representation of "healthcare" now contains information from "AI" and "transforming"
   
   Attention scores visualization (simplified):
   ```
   [AI]       [is]       [transform] [ing]      [health]   [care]
   [1.0, 0.1, 0.2,       0.1,        0.3,       0.2]      # Attention for "AI"
   [0.2, 1.0, 0.3,       0.1,        0.1,       0.1]      # Attention for "is"
   [0.4, 0.2, 1.0,       0.3,        0.5,       0.4]      # Attention for "transform"
   ...
   ```

4. **Contextual Understanding**:
   - As tokens flow through transformer layers, they accumulate contextual information
   - A token like "bank" will have different representations in "river bank" vs. "bank account"
   - Position information is incorporated through positional encodings or embeddings
   
   **Example**: The token "bank" would have different vector representations:
   - In "I deposited money in the bank": [0.41, -0.21, ...]  # financial context
   - In "I sat by the river bank": [-0.12, 0.53, ...]  # geographical context

5. **Pooling Strategy**:
   - To get a single vector for the entire text:
     - Some models use the final representation of a special [CLS] token
     - Others average all token representations
     - More sophisticated pooling mechanisms may also be used
   
   **Example**: For our phrase, the model might:
   - Take the final [CLS] token representation: [0.53, -0.21, 0.87, ...]
   - Or compute the average of all 6 token vectors

6. **Final Projection**:
   - The pooled vector may go through a final projection layer to match the desired dimensionality
   - For Ada-002, this results in a 1536-dimensional vector
   - Vectors are typically L2-normalized (converted to unit vectors) to enable cosine similarity comparison
   
   **Example**: The final embedding for "AI is transforming healthcare" might be:
   ```
   [0.041, 0.032, -0.021, ..., 0.019]  # 1536 values
   ```
   
   After L2 normalization, all values are scaled so the vector has unit length:
   ```
   [0.012, 0.009, -0.006, ..., 0.005]  # Same direction but unit length
   ```

7. **Embedding Space Properties**:
   - The resulting vector space has fascinating geometric properties:
     - Semantic similarity corresponds to angular similarity
     - Analogies can be represented by vector arithmetic (e.g., "king" - "man" + "woman" ≈ "queen")
     - Different concepts organize along different directions in the space
   
   **Example**: In this space:
   - cos_similarity("AI is transforming healthcare", "AI is changing medicine") ≈ 0.92
   - cos_similarity("AI is transforming healthcare", "Weather forecast for today") ≈ 0.12

The dimensionality of embeddings (e.g., 1536 for Ada-002) creates a rich space that can encode complex semantic relationships. Each dimension doesn't necessarily correspond to a human-interpretable feature, but collectively they capture the text's meaning.

When calculating similarity between embeddings, the most common metric is cosine similarity, which measures the cosine of the angle between two vectors (ranging from -1 to 1, where 1 means identical direction).

### Vector Search

Once we have embeddings (vectors) for our documents and questions, we need a way to quickly find which document vectors are closest to our question vector. This is where vector search comes in.

**For beginners:**
Vector search is like finding similar items in a huge library instantly. Instead of comparing words, we compare the numerical patterns (vectors) that represent the meaning of text.

In our system, we use FAISS (Facebook AI Similarity Search), a powerful library that efficiently finds the nearest neighbors in high-dimensional spaces, even with millions of vectors.

#### Technical Deep Dive: How Vector Search Works

Our system uses FAISS (Facebook AI Similarity Search) for vector search. Here's how it works step-by-step:

1. **Vector Normalization**:
   - Before indexing, vectors are normalized to unit length (L2 normalization)
   - This converts the dot product between vectors into cosine similarity
   - For two normalized vectors, higher dot product = smaller angle = more similar
   
   **Example**: A vector [4, 3, 0, ...] would be normalized to:
   ```
   length = sqrt(4² + 3² + 0² + ...) = sqrt(25) = 5
   normalized = [4/5, 3/5, 0/5, ...] = [0.8, 0.6, 0, ...]
   ```

2. **Index Construction**:
   - Our implementation uses the `IndexFlatIP` index type, which stands for "Flat Index, Inner Product"
   - "Flat" means it stores all vectors in memory without compression
   - Each vector is stored with its exact values (no approximation)
   - This is the most accurate but most memory-intensive option
   
   **Example**: Building an index with 3 document vectors:
   ```python
   index = faiss.IndexFlatIP(1536)  # 1536 dimensions
   
   # Three normalized document vectors
   vectors = np.array([
       [0.1, 0.2, ..., 0.01],  # "AI in healthcare" document
       [0.3, 0.1, ..., 0.02],  # "Machine learning basics" document
       [0.2, 0.3, ..., 0.03]   # "Neural networks explained" document
   ], dtype='float32')
   
   index.add(vectors)  # Add vectors to the index
   ```

3. **Search Algorithm**:
   - When a query vector is submitted:
     a. The query vector is also normalized to unit length
     b. FAISS computes the dot product between the query and all stored vectors
     c. Results are sorted by dot product value (highest first)
     d. The top-k results are returned
   
   **Example**: Searching with query "How is AI used in medicine?":
   ```python
   query_vector = np.array([[0.15, 0.25, ..., 0.02]], dtype='float32')  # Normalized
   
   # Search for top 2 results
   distances, indices = index.search(query_vector, k=2)
   
   # Results might be:
   # distances = [[0.87, 0.56]]  # Similarity scores
   # indices = [[0, 2]]  # Document indices (0="AI in healthcare", 2="Neural networks explained")
   ```

4. **Performance Characteristics**:
   - Time complexity: O(n×d) where n = number of vectors, d = vector dimension
   - Memory usage: O(n×d×4) bytes (using 32-bit floats)
   - For our small dataset, this is efficient; for larger datasets, FAISS offers approximate methods:
     - Inverted file indices (IVF): Divide space into clusters for faster search
     - Product quantization (PQ): Compress vectors to use less memory
     - Hierarchical navigable small worlds (HNSW): Graph-based approach for faster search
   
   **Example**: Memory usage for 10,000 documents with 1536 dimensions:
   ```
   10,000 × 1536 × 4 bytes ≈ 61.44 MB for the raw vectors
   ```

5. **Similarity Scoring**:
   - The dot product between normalized vectors ranges from -1 to 1
   - A score of 1 means perfect similarity (vectors point in same direction)
   - A score of 0 means no correlation (vectors are orthogonal)
   - A score of -1 means perfect dissimilarity (vectors point in opposite directions)
   
   **Example**: Similarity scores interpretation:
   ```
   Query: "How does AI help doctors?"
   
   Result 1: "AI assists physicians in diagnosing diseases" (score: 0.92) → Very relevant
   Result 2: "Machine learning models can predict patient outcomes" (score: 0.78) → Relevant
   Result 3: "Computer vision is used in radiology" (score: 0.65) → Somewhat relevant
   Result 4: "The history of artificial intelligence" (score: 0.31) → Not very relevant
   ```

6. **Score Thresholding**:
   - Our retriever implements score thresholding to filter low-confidence matches
   - Results below a specified threshold (e.g., 0.5) can be discarded
   - This helps ensure that only truly relevant matches are returned
   
   **Example**: With threshold=0.7, only the top two results from above would be returned.

## System Overview

Our RAG system follows this workflow:

1. **Data Preparation**: 
   - Load documents from a CSV file
   - Split them into manageable chunks
   - Generate embeddings for each chunk
   - Store these embeddings in an efficient index

2. **Query Processing**:
   - Generate an embedding for the user's question
   - Retrieve the most relevant document chunks using vector search
   - Optional: Rerank the results to improve relevance

3. **Response Generation**:
   - Pass the user question and relevant document chunks to a language model
   - Generate a natural language response based only on the provided context

The system includes multiple caching mechanisms to reduce API calls and improve performance.

## Components

### Configuration

**File**: `config.py`

The configuration module centralizes all system settings. It uses environment variables (loaded from a `.env` file) and provides defaults when specific variables aren't set.

Key configurations include:
- API credentials (OpenAI API key)
- Model selections (embedding model, chat model)
- File paths
- Chunking strategy parameters
- Performance settings

For juniors: This is where all the "knobs and dials" of the system are defined.
For seniors: This design pattern allows for easy environment-specific configuration without code changes.

### Data Loading

**File**: `data_loader.py`

The `DataLoader` class handles:
- Loading the knowledge base from a CSV file
- Processing it into smaller, manageable chunks
- Generating embeddings for each chunk
- Saving the processed data to disk

Key features:
- Automatic detection of previously processed data
- Generation of missing embeddings
- Integration with chunking service
- Persistence of processed data to avoid reprocessing

### Embedding Service

**File**: `embedding_service.py`

The `EmbeddingService` manages:
- Converting text into vector embeddings
- Optimizing API calls through batching
- Caching results to avoid repeated API calls

Advanced features:
- In-memory caching with normalized text keys
- Batch processing to reduce API calls
- Retry logic with exponential backoff
- Cache statistics tracking

For juniors: This service turns text into numbers that the computer can use to find similarities.
For seniors: The service implements key optimizations like memory caching, request batching, and error resilience.

### Text Chunking

**File**: `text_chunker.py`

The `TextChunker` class implements different strategies for dividing text into smaller, manageable pieces:

1. **Sliding Window**: Simple fixed-size chunks with overlaps
2. **Sentence-based**: Chunks based on natural sentence boundaries
3. **Paragraph-based**: Chunks based on paragraph breaks

For each chunk, metadata from the original document is preserved, ensuring traceability.

For seniors: Different chunking strategies have trade-offs in terms of context preservation, retrieval efficiency, and semantic coherence. The system supports multiple strategies to enable experimentation.

#### Technical Deep Dive: Chunking Algorithms

Our system implements three different chunking strategies, each with its own algorithm:

1. **Sliding Window Chunking**:
   ```
   function sliding_window_chunks(text, chunk_size, chunk_overlap):
       chunks = []
       start = 0
       text_length = length(text)
       
       while start < text_length:
           # Determine end position
           end = min(start + chunk_size, text_length)
           
           # If not first chunk, include overlap with previous chunk
           if start > 0:
               start = max(0, start - chunk_overlap)
           
           # Extract chunk
           chunk = text[start:end]
           chunks.append(chunk)
           
           # Move to next position
           start = end
       
       return chunks
   ```

   - **Time Complexity**: O(n) where n is the text length
   - **Pros**: Simple, predictable chunk sizes
   - **Cons**: May split sentences or logical units
   
   **Example**: Text "The quick brown fox jumps over the lazy dog." with chunk_size=10, chunk_overlap=2:
   ```
   Chunk 1: "The quick "
   Chunk 2: "ck brown f"  # starts with overlap from previous chunk
   Chunk 3: "wn fox jum"
   Chunk 4: "x jumps ov"
   Chunk 5: "ps over th"
   Chunk 6: "r the lazy"
   Chunk 7: "lazy dog."
   ```

2. **Sentence-Based Chunking**:
   ```
   function sentence_chunks(text, chunk_size, chunk_overlap):
       # Split text into sentences using regex pattern
       sentences = split_by_regex(text, r'(?<=[.!?])\s+')
       chunks = []
       current_chunk = ""
       
       for sentence in sentences:
           # If adding this sentence exceeds chunk size, start new chunk
           if length(current_chunk) + length(sentence) > chunk_size and current_chunk:
               chunks.append(current_chunk)
               # Create overlap by taking end of previous chunk
               current_chunk = get_overlap_text(current_chunk, chunk_overlap)
           
           # Add sentence to current chunk
           current_chunk += sentence + " "
       
       # Add final chunk if not empty
       if current_chunk.strip():
           chunks.append(current_chunk.strip())
       
       return chunks
   ```

   - **Time Complexity**: O(n) where n is the text length
   - **Space Complexity**: O(n) for storing sentences
   - **Pros**: Preserves sentence boundaries
   - **Cons**: Chunks may vary in size significantly
   
   **Example**: Text "AI is amazing. It can recognize images. It can understand text. It can generate content." with chunk_size=30, chunk_overlap=10:
   ```
   Chunk 1: "AI is amazing. It can recognize images."
   Chunk 2: "images. It can understand text. It can generate content."  # starts with overlap
   ```
   Note how each chunk preserves complete sentences.

3. **Paragraph-Based Chunking**:
   ```
   function paragraph_chunks(text, chunk_size, chunk_overlap):
       # Split text by paragraph breaks
       paragraphs = split_by_regex(text, r'\n\s*\n')
       chunks = []
       current_chunk = ""
       
       for paragraph in paragraphs:
           paragraph = paragraph.strip()
           if not paragraph:
               continue
           
           # If adding this paragraph exceeds chunk size, start new chunk
           if length(current_chunk) + length(paragraph) > chunk_size and current_chunk:
               chunks.append(current_chunk)
               # Create overlap by taking end of previous chunk
               current_chunk = get_overlap_text(current_chunk, chunk_overlap)
           
           # Add paragraph to current chunk
           current_chunk += paragraph + "\n\n"
       
       # Add final chunk if not empty
       if current_chunk.strip():
           chunks.append(current_chunk.strip())
       
       return chunks
   ```

   - **Pros**: Preserves logical document structure
   - **Cons**: May result in very large chunks if paragraphs are long
   
   **Example**: Text with paragraphs:
   ```
   Neural networks are the backbone of modern AI.
   They consist of layers of artificial neurons.
   
   Transformers are a type of neural network.
   They rely on attention mechanisms.
   
   Large language models are built on transformers.
   They can generate human-like text.
   ```
   
   With chunk_size=100, chunk_overlap=20:
   ```
   Chunk 1: "Neural networks are the backbone of modern AI.
   They consist of layers of artificial neurons.
   
   Transformers are a type of neural network.
   They rely on attention mechanisms."
   
   Chunk 2: "mechanisms.
   
   Large language models are built on transformers.
   They can generate human-like text."
   ```
   Note how paragraph structure is preserved.

The overlap mechanism is critical for maintaining context across chunks:
```
function get_overlap_text(text, overlap_size):
    if length(text) <= overlap_size:
        return text
    
    return text[-overlap_size:]  # Return the last 'overlap_size' characters
```

**Example**: If the previous chunk ends with "...large language models use attention mechanisms." and overlap_size=15, the next chunk would start with "attention mechanisms."

Each strategy has trade-offs in terms of semantic coherence, chunk size consistency, and computational complexity. The choice depends on the document structure and specific retrieval needs.

### Vector Storage & Search

**File**: `faiss_indexer.py`

The `FaissIndexer` class:
- Creates and manages a FAISS index for efficient vector search
- Normalizes vectors for optimized cosine similarity
- Persists both the index and associated metadata to disk
- Provides fast vector similarity search functionality

For juniors: FAISS is a specialized database that can quickly find which document chunks are most similar to a question.
For seniors: The implementation uses FAISS's `IndexFlatIP` with L2 normalization, effectively implementing cosine similarity search, which generally performs well for semantic retrieval tasks.

### BM25 Retrieval

**File**: `bm25_retriever.py`

The `BM25Retriever` class implements the BM25 algorithm, a statistical ranking function used for lexical (keyword-based) search:

- Creates an index from tokenized document chunks
- Scores documents based on term frequency and inverse document frequency
- Provides keyword-based search to complement semantic search
- Persists the BM25 model and metadata to disk

For juniors: BM25 is a sophisticated version of keyword search that considers how rare or common words are across all documents.
For seniors: BM25 provides lexical matching that can catch exact terms that semantic search might miss, especially for rare terms, acronyms, or domain-specific terminology.

#### Technical Deep Dive: How BM25 Works

BM25 (Best Matching 25) is a probabilistic ranking function based on the probabilistic retrieval framework. Here's how it works:

1. **Document Tokenization**:
   - Each document is split into individual tokens (words)
   - For our implementation, we use a simple regex-based tokenizer that splits on non-alphanumeric characters
   
   **Example**: The document "AI assists physicians in diagnosing diseases" would be tokenized as:
   ```
   ["ai", "assists", "physicians", "in", "diagnosing", "diseases"]
   ```

2. **BM25 Score Calculation**:
   - For each query term, BM25 calculates a score based on:
     - Term frequency (TF): How often the term appears in the document
     - Inverse document frequency (IDF): How rare the term is across all documents
     - Document length normalization: Prevents bias toward longer documents
   
   The core BM25 formula is:
   ```
   score(D,Q) = ∑(IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl)))
   ```
   Where:
   - f(qi,D) is the frequency of term qi in document D
   - |D| is the length of document D
   - avgdl is the average document length
   - k1 and b are free parameters (typically k1 ∈ [1.2, 2.0] and b = 0.75)
   
   **Example**: For the query "diagnosing diseases":
   - "diagnosing" might be rare (high IDF) and appear once in our document
   - "diseases" might be somewhat common (medium IDF) and also appear once
   - The document length is 6 tokens, perhaps shorter than average
   - BM25 would give a high score due to both terms matching and the document being concise

3. **Key Properties**:
   - Term saturation: After a certain frequency, additional occurrences of a term provide diminishing returns
   - Length normalization: Adjusts for document length to avoid bias toward longer documents
   - Proportional to relevance: Higher scores indicate more relevant documents
   
   **Example**: If "diseases" appears 10 times in one document, BM25 won't give it 10× the weight compared to a document where it appears once.

4. **Comparison with TF-IDF**:
   - Both use term frequency and inverse document frequency
   - BM25 adds term frequency saturation (diminishing returns)
   - BM25 has better document length normalization
   - BM25 typically outperforms TF-IDF in information retrieval tasks

BM25 excels at finding documents containing exact query terms and rare, distinctive words, but it doesn't understand semantic relationships or synonyms the way embedding-based search does.

### Hybrid Retrieval

**File**: `hybrid_retriever.py`

The `HybridRetriever` class combines semantic search (using embeddings) with lexical search (using BM25):

- Manages both vector and BM25 indices
- Executes parallel searches using both methods
- Combines and normalizes results using a weighted approach
- Implements deduplication and score normalization

For juniors: Hybrid search gets "best of both worlds" by combining AI-powered semantic understanding with traditional keyword matching.
For seniors: This approach mitigates the weaknesses of each method - semantic search misses exact matches but captures meaning, while BM25 catches exact terms but misses synonyms.

### Retrieval

**File**: `retriever.py`

The `Retriever` class:
- Converts user queries to embeddings
- Searches the FAISS index for relevant chunks
- Implements score thresholding to filter low-relevance results
- Provides both standard and reranked search capabilities

Advanced features:
- Reranking to improve search quality
- Support for different result formats (text-only or with metadata)
- Configurable search parameters

#### Technical Deep Dive: Reranking

Our system implements a basic diversification-based reranking algorithm:

1. **Initial Retrieval**:
   - Fetch more results than needed (e.g., top-10 instead of top-3)
   - Each result includes text, metadata, and similarity score
   
   **Example**: For the query "What are the applications of AI?":
   ```
   Initial results (top-10):
   1. "AI in Healthcare" (score: 0.89, source_id: 6)
   2. "AI for Medical Diagnosis" (score: 0.87, source_id: 6)
   3. "AI in Medical Imaging" (score: 0.84, source_id: 6)
   4. "AI in Finance" (score: 0.82, source_id: 11)
   5. "AI for Fraud Detection" (score: 0.81, source_id: 11)
   6. "AI in Gaming" (score: 0.79, source_id: 28)
   7. "AI for Smart Cities" (score: 0.78, source_id: 34)
   8. "AI in Education" (score: 0.77, source_id: 24)
   9. "AI in Robotics" (score: 0.76, source_id: 14)
   10. "AI for Weather Prediction" (score: 0.75, source_id: 35)
   ```
   Notice that the top 3 results are all from the same source (healthcare, source_id: 6).

2. **Source Grouping**:
   ```
   function group_by_source(search_results):
       grouped_sources = {}
       for result in search_results:
           source_id = result.metadata.source_id
           if source_id not in grouped_sources:
               grouped_sources[source_id] = []
           grouped_sources[source_id].append(result)
       return grouped_sources
   ```
   
   **Example**: The results would be grouped as:
   ```
   Source 6 (Healthcare): [Result 1, Result 2, Result 3]
   Source 11 (Finance): [Result 4, Result 5]
   Source 28 (Gaming): [Result 6]
   Source 34 (Smart Cities): [Result 7]
   Source 24 (Education): [Result 8]
   Source 14 (Robotics): [Result 9]
   Source 35 (Weather): [Result 10]
   ```

3. **Round-Robin Selection**:
   ```
   function round_robin_rerank(grouped_sources, final_top_k):
       reranked_results = []
       sources_list = values(grouped_sources)
       
       # Find maximum items per source
       max_items = max(length(source) for source in sources_list)
       
       # Take one item from each source in each round
       for i in range(max_items):
           for source_results in sources_list:
               if i < length(source_results):
                   reranked_results.append(source_results[i])
       
       # Return only the requested number of top results
       return reranked_results[:final_top_k]
   ```
   
   **Example**: The reranking process:
   ```
   Round 1 (taking the top result from each source):
   - Source 6: "AI in Healthcare" (score: 0.89)
   - Source 11: "AI in Finance" (score: 0.82)
   - Source 28: "AI in Gaming" (score: 0.79)
   - Source 34: "AI for Smart Cities" (score: 0.78)
   - Source 24: "AI in Education" (score: 0.77)
   - Source 14: "AI in Robotics" (score: 0.76)
   - Source 35: "AI for Weather Prediction" (score: 0.75)
   
   Round 2 (taking the second result from sources that have more):
   - Source 6: "AI for Medical Diagnosis" (score: 0.87)
   - Source 11: "AI for Fraud Detection" (score: 0.81)
   
   Round 3 (taking the third result from sources that have more):
   - Source 6: "AI in Medical Imaging" (score: 0.84)
   
   Final reranked results (top-3):
   1. "AI in Healthcare" (score: 0.89, source_id: 6)
   2. "AI in Finance" (score: 0.82, source_id: 11)
   3. "AI in Gaming" (score: 0.79, source_id: 28)
   ```

4. **Benefits of Reranking**:
   - Diversity: Results come from multiple sources rather than concentrating on one
   - Coverage: Wider range of potentially relevant information
   - Context: Related but different perspectives on the query
   
   **Example**: Compare before and after reranking (top-3):
   ```
   Before reranking (all healthcare):
   1. "AI in Healthcare" (score: 0.89, source_id: 6)
   2. "AI for Medical Diagnosis" (score: 0.87, source_id: 6)
   3. "AI in Medical Imaging" (score: 0.84, source_id: 6)
   
   After reranking (diverse domains):
   1. "AI in Healthcare" (score: 0.89, source_id: 6)
   2. "AI in Finance" (score: 0.82, source_id: 11)
   3. "AI in Gaming" (score: 0.79, source_id: 28)
   ```

5. **Advanced Reranking Alternatives**:
   - **Cross-encoder reranking**: Uses a more powerful model to rerank candidate pairs
   - **Query expansion**: Adds terms to the original query
   - **MMR (Maximal Marginal Relevance)**: Balances relevance with diversity
   - **Learning-to-rank**: Trains a model specifically for ranking results

The current implementation is optimized for simplicity and efficiency, but more sophisticated reranking strategies could be integrated for improved performance.

### Response Generation

**File**: `response_generator.py`

The `ResponseGenerator` class:
- Formats the retrieved chunks into a structured context
- Sends the context and user query to a language model
- Processes the model's response
- Provides options for metadata inclusion in the context

Advanced features:
- Response caching to avoid duplicate API calls
- Structured context formatting
- Configurable temperature for response variability

### Main Pipeline

**File**: `main.py`

The `RAGPipeline` class:
- Coordinates all other components
- Provides an interactive command-line interface
- Handles configuration via command-line arguments
- Implements high-level query caching
- Manages experiment tracking (MLflow integration)

For juniors: This is the "main engine" that connects all parts together.
For seniors: The module implements the coordinator pattern with clear separation of concerns.

## System Optimizations

### Memory Caching

The system implements multiple levels of caching to minimize API calls and improve responsiveness:

1. **Embedding Cache** (in `EmbeddingService`):
   - Caches embeddings for text that has been processed before
   - Uses normalized text keys to increase cache hits
   - Tracks statistics on cache performance

2. **Response Cache** (in `ResponseGenerator`):
   - Caches generated responses for specific query-context combinations
   - Avoids repeated API calls for identical scenarios

3. **Query Cache** (in `RAGPipeline`):
   - High-level cache that stores entire query-response pairs
   - Completely bypasses the retrieval and generation steps for repeated queries
   - Accounts for different pipeline configurations (reranking, metadata inclusion)

For seniors: The multi-level caching strategy optimizes different parts of the pipeline independently, allowing for maximal reuse of intermediary results.

### Batch Processing

The system uses batch processing to optimize API calls:

- **Embedding Generation**: Processes chunks in configurable batches rather than one at a time
- **Error Handling**: Implements retry logic with exponential backoff for API failures

For seniors: Batch size needs to be tuned based on token limits, network reliability, and response time requirements.

### Reranking

The `Retriever` implements a basic diversification-based reranking algorithm:

1. Fetch more initial results than needed
2. Group by source document
3. Reorder to prioritize diversity of sources
4. Return the top K reranked results

For seniors: This simple diversification strategy can be replaced with more sophisticated reranking methods such as cross-encoders or query-specific reranking models.

### Hybrid Search

Our system implements hybrid retrieval that combines embedding-based search with BM25 lexical search:

1. **Complementary Search Methods**:
   - **Vector Search**: Captures semantic relationships and conceptual similarity
   - **BM25 Search**: Captures exact term matches and emphasizes rare terms

2. **Implementation Strategy**:
   - Run both search methods in parallel
   - Normalize scores from each method to 0-1 range
   - Apply configurable weights to each method (default 70% vector, 30% BM25)
   - Combine results by averaging scores for duplicate results
   - Sort by combined score

3. **Benefits**:
   - Improved recall by capturing both semantic and lexical matches
   - Better handling of domain-specific terminology and rare terms
   - More balanced results that consider both meaning and exact wording

4. **Optimal Use Cases**:
   - Queries containing specialized terminology or acronyms
   - Situations requiring exact matches of key terms
   - Handling both conceptual questions and fact-based retrieval

For seniors: The weighting between vector and BM25 scores can be tuned based on the domain, with more technical domains often benefiting from higher BM25 weights.

## Running the System

To run the system:

1. Ensure you have a knowledge base CSV file in the expected location (`data/knowledge_base.csv`)
2. Set up environment variables (particularly `OPENAI_API_KEY`)
3. Run the main module: `python -m src.main`

Command-line options:
- `--chunk-size`: Maximum size of each document chunk
- `--chunk-overlap`: Overlap between consecutive chunks
- `--chunk-strategy`: Chunking strategy (sliding_window, sentence, paragraph)
- `--no-reranking`: Disable reranking of search results
- `--no-metadata`: Exclude metadata from response generation
- `--retrieval-method`: Search method to use (vector, bm25, or hybrid)
- `--vector-weight`: Weight for vector results in hybrid search (default: 0.7)

## Advanced Topics

### Performance Tuning

For optimal performance, consider adjusting:

1. **Chunking Parameters**:
   - Larger chunks provide more context but reduce retrieval precision
   - More overlap improves retrieval of information that might cross chunk boundaries
   - Different chunking strategies may work better for different document types

2. **Vector Search Parameters**:
   - Initial and final top-k values affect recall vs. precision
   - Score thresholds can filter out low-confidence matches

3. **API Parameters**:
   - Batch sizes impact throughput and latency
   - Temperature affects response creativity vs. determinism

### Cost Optimization

API costs can be managed through:

1. **Caching Strategy**:
   - All three cache levels reduce API calls
   - Consider persistent caching across restarts for production deployments

2. **Model Selection**:
   - Cheaper embedding models for high-volume applications
   - Less powerful but more cost-effective completion models

3. **Batch Optimization**:
   - Maximizing batch sizes reduces per-token costs
   - Reducing vector dimensions can lower embedding costs

4. **Hybrid Approaches**:
   - Consider keyword pre-filtering before vector search for larger datasets
   - Use tiered models (cheaper models first, more expensive models only when needed)

By implementing these optimizations, our RAG system balances performance, cost, and accuracy for various use cases and volumes.
