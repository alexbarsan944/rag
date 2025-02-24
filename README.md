## RAG Pipeline Documentation

This document provides a complete overview and guide to the Retrieval-Augmented Generation (RAG) pipeline codebase. It covers the project’s purpose, high-level architecture, module descriptions, usage instructions, testing guidelines, and contributing information.

## Table of Contents

1. [Overview & Introduction](#overview--introduction)
2. [Project Structure](#project-structure)
3. [Module Documentation](#module-documentation)
   - [Configuration](#configuration)
   - [Data Loader](#data-loader)
   - [Embedding Service](#embedding-service)
   - [FAISS Indexer](#faiss-indexer)
   - [Retriever](#retriever)
   - [Response Generator](#response-generator)
   - [Main Pipeline](#main-pipeline)
4. [Usage Instructions](#usage-instructions)
   - [Installation](#installation)
   - [Running the Pipeline](#running-the-pipeline)
   - [Running Tests](#running-tests)
   - [Code Formatting](#code-formatting)
---

## Overview & Introduction

The **RAG Pipeline** is a Python-based application that integrates Retrieval-Augmented Generation into a single workflow. It combines the power of modern machine learning tools to generate context-aware responses by retrieving relevant information from a knowledge base. The system uses:

- **OpenAI's Embedding API:** Converts text into vector representations.
- **FAISS:** Provides fast similarity search over high-dimensional vectors.
- **LangChain Chat Models:** Generates responses based on the retrieved context.

### Main Features

- **Dataset Loading:** Reads a CSV file containing the knowledge base.
- **Embedding Generation:** Uses OpenAI's API to generate text embeddings.
- **Indexing with FAISS:** Builds and maintains an index for efficient vector search.
- **Retrieval:** Fetches the top-K relevant documents based on a query.
- **Response Generation:** Generates answers using a Chat model with the retrieved context.

### Problem Solved

The pipeline ensures that generated responses are grounded in factual data, reducing the risk of hallucination by relying on verified context extracted from a knowledge base.

### Prerequisites
- Python 3.8+ (tested with Python 3.12)
- Familiarity with machine learning pipelines, REST APIs, and vector search.
- Dependencies (see [Usage Instructions](#usage-instructions)).

---

## Project Structure

Below is the recommended project structure:

```
rag_project/
├── src/
│   ├── __init__.py              # Package initializer for source modules.
│   ├── config.py                # Global configuration (API keys, file paths, etc.).
│   ├── data_loader.py           # Module for loading and saving the dataset.
│   ├── embedding_service.py     # Module for generating embeddings using OpenAI.
│   ├── faiss_indexer.py         # Module for building and managing the FAISS index.
│   ├── retriever.py             # Module for retrieving documents using FAISS.
│   ├── response_generator.py    # Module for generating responses using a Chat model.
│   ├── main.py                  # Main entry point to run the RAG pipeline.
├── tests/
│   ├── __init__.py              # Package initializer for tests.
│   ├── test_data_loader.py      # Unit tests for data loader functionality.
│   ├── test_embedding_service.py# Unit tests for embedding service.
│   ├── test_rag_pipeline.py     # Integration tests for the overall RAG pipeline.
├── data/
│   └── knowledge_base.csv       # CSV file containing the knowledge base.
├── requirements.txt             # List of project dependencies.
└── README.md                    # This documentation file.
```

- **src/**: Contains all the main source code divided by functionality.
- **tests/**: Contains unit and integration tests to ensure the pipeline works correctly.
- **data/**: Stores the knowledge base and other data assets.
- **requirements.txt**: Lists all dependencies required to run the project.
- **README.md**: Provides an overview, usage instructions, and detailed documentation.

---

## Module Documentation

### Configuration
**File:** `src/config.py`
**Purpose:** Loads and validates configuration settings (such as API keys and file paths) using environment variables and a dataclass.

### Data Loader
**File:** `src/data_loader.py`
**Purpose:** Loads the knowledge base CSV file and saves pandas DataFrames.

### Embedding Service
**File:** `src/embedding_service.py`
**Purpose:** Generates embeddings using OpenAI’s API.

### FAISS Indexer
**File:** `src/faiss_indexer.py`
**Purpose:** Builds and manages a FAISS index using generated embeddings.

### Retriever
**File:** `src/retriever.py`
**Purpose:** Searches the FAISS index and retrieves relevant documents.

### Response Generator
**File:** `src/response_generator.py`
**Purpose:** Uses a Chat model to generate responses based on retrieved documents.

### Main Pipeline
**File:** `src/main.py`
**Purpose:** Orchestrates the entire RAG pipeline.

---

## Usage Instructions

### Installation

#### Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

#### Set Up the Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### Set Up Environment Variables
Create a `.env` file at the project root:
```bash
OPENAI_API_KEY=your_openai_api_key
```

---

### Running the Pipeline

To execute the full RAG pipeline, ensure that you are in the project root and run:

```bash
python -m src.main
```

This command ensures that the module is properly located and executed within the package structure.

---

### Running Tests
Using pytest from the project root:
```bash
pytest
```

---

### Code Formatting

The codebase follows **Black** formatting to ensure consistency and readability across all modules. 
Black is an opinionated code formatter that enforces a uniform style, reducing code review overhead and improving maintainability.

To format the code, run:

```bash
black .
```