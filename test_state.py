"""
Test state and langgraph import
"""
import sys
print("Python version:", sys.version)

print("\n1. Testing langgraph.graph...")
try:
    from langgraph.graph import StateGraph, END
    print("   ✓ StateGraph, END imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n2. Testing langgraph.graph.message...")
try:
    from langgraph.graph.message import add_messages
    print("   ✓ add_messages imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n3. Testing langchain_core.messages...")
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    print("   ✓ messages imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n4. Testing state...")
try:
    from state import RAGState
    print("   ✓ RAGState imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\nAll successful!")
