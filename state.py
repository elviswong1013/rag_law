from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    answer: str
    needs_more_info: bool
    search_count: int
    max_searches: int
    context: str
