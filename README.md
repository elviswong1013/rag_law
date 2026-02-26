# Agentic RAG System for Legal Documents

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangGraph for querying legal documents. The system supports multiple file formats (TXT, PDF, Excel) and uses Dashscope APIs for embeddings and LLM capabilities.

## Features

- **Multi-format Document Support**: Process TXT, PDF, and Excel files
- **Agentic RAG with LangGraph**: Intelligent retrieval with query refinement
- **Hybrid Search**: Combines similarity search and semantic search
- **Reranking**: Multiple reranking strategies (BM25, Cross-Encoder, Cohere, Hybrid)
- **Dashscope Integration**: Uses Alibaba Cloud's Dashscope for embeddings and LLM
- **Vector Database**: ChromaDB for efficient similarity search
- **Smart Chunking**: Paragraph-based and size-based text chunking

## Architecture

The system consists of several key components:

1. **Document Loader**: Loads and processes documents from various formats
2. **Text Chunker**: Splits documents into manageable chunks
3. **Embeddings**: Generates embeddings using Dashscope
4. **Vector Store**: ChromaDB for storing and querying embeddings
5. **Reranker**: Improves retrieval results with multiple strategies
6. **Agentic RAG**: LangGraph-based agent for intelligent querying

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` and add your Dashscope API key:
```
DASHSCOPE_API_KEY=your_actual_api_key_here
```

## Usage

### Basic Usage

```python
from main import RAGSystem

# Initialize the system
rag_system = RAGSystem()

# Load documents from a directory
documents = rag_system.load_documents(["path/to/your/documents"])

# Build the vector database
rag_system.build_vector_database(documents)

# Query the system
result = rag_system.query("What are the legal requirements for...")

print(result['answer'])
```

### Interactive Mode

Run the system in interactive mode:
```bash
python main.py --interactive
```

### Command Line

Run with a single query:
```bash
python main.py
```

## Configuration

Environment variables in `.env`:

- `DASHSCOPE_API_KEY`: Your Dashscope API key (required)
- `CHROMA_DB_PATH`: Path to store vector database (default: `./chroma_db`)
- `CHUNK_SIZE`: Size of text chunks (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K_RETRIEVAL`: Number of documents to retrieve (default: 10)
- `RERANK_TOP_K`: Number of documents after reranking (default: 5)
- `EMBEDDING_MODEL`: Dashscope embedding model (default: `text-embedding-v2`)
- `LLM_MODEL`: Dashscope LLM model (default: `qwen-plus`)

## Reranking Methods

The system supports multiple reranking strategies:

- **score**: Combines distance and length scores
- **bm25**: Uses BM25 algorithm for keyword-based reranking
- **cross_encoder**: Uses sentence-transformers CrossEncoder
- **cohere**: Uses Cohere's reranking API (requires Cohere API key)
- **hybrid**: Combines multiple methods for best results

## LangGraph Workflow

The agentic RAG system uses LangGraph with the following nodes:

1. **retrieve**: Performs similarity and semantic search
2. **rerank**: Reranks retrieved documents
3. **evaluate**: Evaluates if more information is needed
4. **generate**: Generates the final answer
5. **refine_query**: Refines the query if more searches are needed

The agent can perform multiple retrieval iterations with query refinement to find the most relevant information.

## File Structure

```
rag_law/
├── main.py              # Main application entry point
├── document_loader.py   # Document loading logic
├── text_chunker.py      # Text chunking strategies
├── embeddings.py        # Dashscope embeddings
├── vector_store.py      # ChromaDB wrapper
├── reranker.py          # Reranking strategies
├── llm.py              # Dashscope LLM wrapper
├── state.py            # LangGraph state definition
├── agentic_rag.py      # LangGraph-based RAG agent
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Requirements

- Python 3.8+
- Dashscope API key
- Internet connection for API calls

## License

MIT License
