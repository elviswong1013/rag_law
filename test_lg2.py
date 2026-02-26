"""
Test langgraph.graph import
"""
import sys
print("Python version:", sys.version)

print("\n1. Testing langgraph.graph...")
try:
    from langgraph.graph import StateGraph, END
    print("   ✓ StateGraph, END imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nDone!")
