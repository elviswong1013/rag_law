# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Key

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Dashscope API key:
```
DASHSCOPE_API_KEY=sk-your-actual-api-key-here
```

Get your API key from: https://dashscope.console.aliyun.com/

## Step 3: Run the System

### Option 1: Interactive Mode
```bash
python main.py --interactive
```

### Option 2: Test Script
```bash
python test_rag.py
```

### Option 3: Python Script
```python
from main import RAGSystem

# Initialize
rag_system = RAGSystem()

# Load documents (point to your data directory)
documents = rag_system.load_documents([r"d:\from old pc\MO_Law_checking"])

# Build vector database (only needed once)
rag_system.build_vector_database(documents)

# Query
result = rag_system.query("Your question here")
print(result['answer'])
```

## What Happens Behind the Scenes

1. **Document Loading**: Reads all TXT, PDF, and Excel files
2. **Chunking**: Splits documents into smaller, manageable pieces
3. **Embedding**: Converts text to vectors using Dashscope
4. **Vector Storage**: Stores embeddings in ChromaDB
5. **Query Processing**:
   - Performs similarity and semantic search
   - Reranks results for better relevance
   - Uses LangGraph agent for intelligent retrieval
   - Generates answer using Dashscope LLM

## Tips

- First run will take longer as it builds the vector database
- Subsequent runs will be much faster
- Adjust `TOP_K_RETRIEVAL` and `RERANK_TOP_K` in `.env` for different result sizes
- Use `rebuild=True` when adding new documents

## Troubleshooting

**Issue**: "DASHSCOPE_API_KEY not found"
- **Solution**: Make sure you created `.env` file with your API key

**Issue**: Slow first run
- **Solution**: Normal - it's building embeddings for all documents

**Issue**: Poor results
- **Solution**: Try adjusting `CHUNK_SIZE` or `RERANK_TOP_K` in `.env`
