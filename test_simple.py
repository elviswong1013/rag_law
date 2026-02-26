"""
Minimal test to verify imports
"""
print("Testing imports step by step...")

print("1. Testing OpenAIEmbeddings...")
from embeddings import OpenAIEmbeddings
print("   ✓ Done")

print("2. Testing OpenAILLM...")
from llm import OpenAILLM
print("   ✓ Done")

print("3. Testing Reranker...")
from reranker import Reranker
print("   ✓ Done")

print("All imports successful!")
