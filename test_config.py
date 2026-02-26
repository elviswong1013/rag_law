"""Test config loading"""
import sys
print("Python:", sys.version)

print("Loading config...")
from config import PipelineConfig
config = PipelineConfig.from_env()

print(f"LLM Provider: {config.llm.provider}")
print(f"LLM Model: {config.llm.model}")
print(f"LLM API Base: {config.llm.api_base}")
print(f"LLM API Key: {config.llm.api_key[:10]}..." if config.llm.api_key else "LLM API Key: None")
print(f"Auth Header: {config.llm.auth_header}")
print(f"Auth Scheme: {config.llm.auth_scheme}")
print(f"Embedding Model: {config.embedding.model}")
print("✓ Done!")
