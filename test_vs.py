"""
Test vector_store import
"""
import sys
print("Python version:", sys.version)

print("\n1. Testing chromadb...")
try:
    import chromadb
    print("   ✓ chromadb imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing text_chunker...")
try:
    from text_chunker import TextChunker
    print("   ✓ TextChunker imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing vector_store...")
try:
    from vector_store import VectorStore
    print("   ✓ VectorStore imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
