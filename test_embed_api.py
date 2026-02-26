"""Test Embeddings with custom base URL"""
import sys
print("Python:", sys.version)

print("Testing Embeddings with custom base URL...")
from embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key="sk-aeb5a9cdf6c440d69aa60c0a3aee65b4",
    model="text-embedding-3-small",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    auth_header="Authorization",
    auth_scheme="Bearer"
)

print("Generating embedding...")
vec = embeddings.embed_query("Hello world")
print(f"Embedding dimension: {len(vec)}")
print(f"First 5 values: {vec[:5]}")
print("✓ Done!")
