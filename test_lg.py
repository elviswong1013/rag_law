"""
Test langgraph import only
"""
import sys
print("Python version:", sys.version)

print("\n1. Testing langgraph import (this might take time)...")
try:
    import langgraph
    print(f"   ✓ langgraph {langgraph.__version__} imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nDone!")
