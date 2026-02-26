# RAG System Architecture

## Overview

This is an agentic Retrieval-Augmented Generation (RAG) system built with LangGraph for querying legal documents. The system uses Dashscope APIs for embeddings and LLM capabilities, ChromaDB for vector storage, and implements multiple search and reranking strategies.

## System Components

### 1. Document Loader ([document_loader.py](file:///d:\from old pc\MO_Law_checking\rag_law\document_loader.py))
- Supports TXT, PDF, and Excel files
- Loads individual files or entire directories
- Extracts metadata (filename, file size, file type)

### 2. Text Chunker ([text_chunker.py](file:///d:\from old pc\MO_Law_checking\rag_law\text_chunker.py))
- Two chunking strategies:
  - **Size-based**: Fixed-size chunks with overlap
  - **Paragraph-based**: Preserves paragraph boundaries
- Maintains metadata for each chunk

### 3. Embeddings ([embeddings.py](file:///d:\from old pc\MO_Law_checking\rag_law\embeddings.py))
- Uses Dashscope's text-embedding-v2 model
- Batch processing for efficiency
- Separate methods for document and query embeddings

### 4. Vector Store ([vector_store.py](file:///d:\from old pc\MO_Law_checking\rag_law\vector_store.py))
- ChromaDB wrapper with persistent storage
- Three search methods:
  - **Similarity Search**: Cosine similarity based
  - **Semantic Search**: Score threshold filtering
  - **Hybrid Search**: Combines both methods

### 5. Reranker ([reranker.py](file:///d:\from old pc\MO_Law_checking\rag_law\reranker.py))
- Multiple reranking strategies:
  - **Score-based**: Combines distance and length scores
  - **BM25**: Keyword-based ranking
  - **Cross-Encoder**: Neural reranking with sentence-transformers
  - **Cohere**: Commercial reranking API
  - **Hybrid**: Combines multiple methods

### 6. LLM Wrapper ([llm.py](file:///d:\from old pc\MO_Law_checking\rag_law\llm.py))
- Dashscope LLM integration (qwen-plus model)
- Supports streaming responses
- Conversation history support

### 7. LangGraph State ([state.py](file:///d:\from old pc\MO_Law_checking\rag_law\state.py))
- Defines the RAG state structure
- Tracks messages, documents, answers, and search iterations

### 8. Agentic RAG ([agentic_rag.py](file:///d:\from old pc\MO_Law_checking\rag_law\agentic_rag.py))
- LangGraph-based agent with multiple nodes:
  - **retrieve**: Performs vector search
  - **rerank**: Reranks retrieved documents
  - **evaluate**: Determines if more info needed
  - **generate**: Creates final answer
  - **refine_query**: Improves query for next iteration
- Supports multiple retrieval iterations with query refinement

### 9. Main System ([main.py](file:///d:\from old pc\MO_Law_checking\rag_law\main.py))
- Orchestrates all components
- Provides both programmatic and interactive interfaces
- Handles configuration from environment variables

## Workflow

### Indexing Phase
```
Documents → Document Loader → Text Chunker → Embeddings → Vector Store
```

### Query Phase
```
Query → Embed Query → Similarity Search → Semantic Search → 
Combine Results → Rerank → Evaluate → Generate Answer
```

### Agentic Flow
```
Query → Retrieve → Rerank → Evaluate
         ↓                      ↓
    (needs more?)         (sufficient?)
         ↓                      ↓
   Refine Query → Retrieve → Generate Answer
```

## Key Features

1. **Multi-format Support**: Handles TXT, PDF, and Excel files seamlessly
2. **Hybrid Search**: Combines similarity and semantic search for better results
3. **Advanced Reranking**: Multiple strategies to improve relevance
4. **Agentic Retrieval**: LangGraph agent can refine queries and perform multiple searches
5. **Dashscope Integration**: Uses Alibaba Cloud's powerful AI services
6. **Persistent Storage**: ChromaDB with local persistence
7. **Configurable**: All parameters configurable via environment variables

## Configuration

All settings in `.env`:
- `DASHSCOPE_API_KEY`: Required for embeddings and LLM
- `CHROMA_DB_PATH`: Vector database storage location
- `CHUNK_SIZE`: Text chunk size (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K_RETRIEVAL`: Initial retrieval count (default: 10)
- `RERANK_TOP_K`: Final result count (default: 5)
- `EMBEDDING_MODEL`: Dashscope embedding model
- `LLM_MODEL`: Dashscope LLM model

## Usage Examples

### Basic Query
```python
from main import RAGSystem

rag = RAGSystem()
result = rag.query("What are the legal requirements?")
print(result['answer'])
```

### With Custom Configuration
```python
rag = RAGSystem()
rag.top_k = 20
rag.rerank_top_k = 10
result = rag.query("Your question", max_searches=3)
```

### Interactive Mode
```bash
python main.py --interactive
```

## Performance Considerations

1. **First Run**: Slower due to embedding generation
2. **Subsequent Runs**: Much faster using cached embeddings
3. **Batch Processing**: Embeddings processed in batches of 25
4. **Memory**: Vector database size depends on document count

## Extension Points

The system is designed for easy extension:

1. **New Document Types**: Add to `DocumentLoader`
2. **New Chunking Strategies**: Add to `TextChunker`
3. **New Reranking Methods**: Add to `Reranker`
4. **New LLM Models**: Configure in `.env`
5. **Custom Agents**: Extend `AgenticRAG` class
