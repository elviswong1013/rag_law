"""Test LLM with custom base URL"""
import sys
print("Python:", sys.version)

print("Testing LLM with custom base URL...")
from llm import OpenAILLM

llm = OpenAILLM(
    api_key="sk-aeb5a9cdf6c440d69aa60c0a3aee65b4",
    model="qwen3.5-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    auth_header="Authorization",
    auth_scheme="Bearer"
)

print("Generating response...")
response = llm.generate("What is 1+1?")
print(f"Response: {response}")
print("✓ Done!")
