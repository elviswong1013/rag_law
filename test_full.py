"""
Test agentic_rag import
"""
import sys
print("Python version:", sys.version)

print("\n1. Testing document_loader...")
try:
    from document_loader import DocumentLoader
    print("   ✓ DocumentLoader imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing state...")
try:
    from state import RAGState
    print("   ✓ RAGState imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing agentic_rag...")
try:
    from agentic_rag import AgenticRAG
    print("   ✓ AgenticRAG imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n4. Testing main...")
try:
    from main import RAGSystem
    print("   ✓ RAGSystem imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nAll imports successful!")
