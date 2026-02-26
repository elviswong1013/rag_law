"""
Test to isolate the import issue
"""
import sys
print("Python version:", sys.version)

print("\n1. Testing OpenAIEmbeddings...")
from embeddings import OpenAIEmbeddings
print("   ✓ OpenAIEmbeddings imported")

print("\n2. Testing OpenAILLM...")  
from llm import OpenAILLM
print("   ✓ OpenAILLM imported")

print("\n3. Testing rank_bm25...")
from rank_bm25 import BM25Okapi
print("   ✓ BM25Okapi imported")

print("\nAll core imports successful!")
