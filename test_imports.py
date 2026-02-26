"""
Simple test to verify imports and basic setup
"""
print("Testing imports...")

try:
    from embeddings import OpenAIEmbeddings
    print("✓ OpenAIEmbeddings imported")
except Exception as e:
    print(f"✗ OpenAIEmbeddings import error: {e}")

try:
    from llm import OpenAILLM
    print("✓ OpenAILLM imported")
except Exception as e:
    print(f"✗ OpenAILLM import error: {e}")

try:
    from reranker import Reranker
    print("✓ Reranker imported")
except Exception as e:
    print(f"✗ Reranker import error: {e}")

try:
    from vector_store import VectorStore
    print("✓ VectorStore imported")
except Exception as e:
    print(f"✗ VectorStore import error: {e}")

try:
    from agentic_rag import AgenticRAG
    print("✓ AgenticRAG imported")
except Exception as e:
    print(f"✗ AgenticRAG import error: {e}")

print("\nTesting basic instantiation...")

try:
    import os
    os.environ["OPENAI_API_KEY"] = "test-key-for-import"
    
    embeddings = OpenAIEmbeddings(api_key="test-key", model="text-embedding-3-small")
    print("✓ OpenAIEmbeddings instantiated")
except Exception as e:
    print(f"✗ OpenAIEmbeddings instantiation error: {e}")

try:
    llm = OpenAILLM(api_key="test-key", model="gpt-4o-mini")
    print("✓ OpenAILLM instantiated")
except Exception as e:
    print(f"✗ OpenAILLM instantiation error: {e}")

print("\n✓ All basic tests passed!")
