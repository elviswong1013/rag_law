"""
Test sentence-transformers import
"""
import sys
import importlib

print("Python version:", sys.version)

print("\n1. Testing sentence_transformers import...")
try:
    from sentence_transformers import CrossEncoder
    print("   ✓ CrossEncoder imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing cohere import...")
try:
    import cohere
    print("   ✓ cohere imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing Reranker...")
try:
    from reranker import Reranker
    r = Reranker(method="score")
    print("   ✓ Reranker instantiated")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nAll imports successful!")
