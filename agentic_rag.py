from typing import Dict, Any, List, Set
from langchain_core.messages import HumanMessage, AIMessage
from state import RAGState
from vector_store import VectorStore
from embeddings import OpenAIEmbeddings
from reranker import Reranker
from llm import OpenAILLM


class AgenticRAG:
    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: OpenAIEmbeddings,
        reranker: Reranker,
        llm: OpenAILLM,
        top_k: int = 10,
        rerank_top_k: int = 5
    ) -> None:
        self.vector_store: VectorStore = vector_store
        self.embeddings: OpenAIEmbeddings = embeddings
        self.reranker: Reranker = reranker
        self.llm: OpenAILLM = llm
        self.top_k: int = top_k
        self.rerank_top_k: int = rerank_top_k
    
    def retrieve_node(self, state: RAGState) -> RAGState:
        query: str = state["query"]
        
        query_embedding: List[float] = self.embeddings.embed_query(query)
        
        similarity_results: List[Dict[str, Any]] = self.vector_store.similarity_search(
            query_embedding,
            top_k=self.top_k
        )
        
        semantic_results: List[Dict[str, Any]] = self.vector_store.semantic_search(
            query_embedding,
            top_k=self.top_k
        )
        
        combined_results: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        
        for result in similarity_results:
            if result['id'] not in seen_ids:
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        for result in semantic_results:
            if result['id'] not in seen_ids:
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        state["retrieved_docs"] = combined_results
        return state
    
    def rerank_node(self, state: RAGState) -> RAGState:
        query: str = state["query"]
        retrieved_docs: List[Dict[str, Any]] = state["retrieved_docs"]
        
        if not retrieved_docs:
            state["reranked_docs"] = []
            return state
        
        reranked: List[Dict[str, Any]] = self.reranker.rerank(
            retrieved_docs,
            query,
            top_k=self.rerank_top_k
        )
        
        state["reranked_docs"] = reranked
        return state
    
    def evaluate_retrieval_node(self, state: RAGState) -> RAGState:
        reranked_docs: List[Dict[str, Any]] = state["reranked_docs"]
        
        if not reranked_docs:
            state["needs_more_info"] = True
            state["context"] = ""
            return state
        
        context_parts: List[str] = []
        for i, doc in enumerate(reranked_docs):
            context_parts.append(f"[Document {i+1}]")
            context_parts.append(doc['content'])
            context_parts.append("")
        
        state["context"] = "\n".join(context_parts)
        
        if len(reranked_docs) < 3:
            state["needs_more_info"] = True
        else:
            state["needs_more_info"] = False
        
        return state
    
    def generate_answer_node(self, state: RAGState) -> RAGState:
        query: str = state["query"]
        context: str = state["context"]
        
        if not context:
            prompt = f"""你是一个专业的法律助手。请根据你的知识回答以下问题：

问题: {query}

如果信息不足，请诚实地说明你无法提供完整的答案。"""
        else:
            prompt = f"""你是一个专业的法律助手。请根据以下参考文档回答问题：

参考文档:
{context}

问题: {query}

要求:
1. 基于参考文档提供准确的法律信息
2. 如果参考文档中没有相关信息，请明确说明
3. 引用具体的法律条文或规定
4. 提供清晰、专业的回答"""
        
        answer = self.llm.generate(prompt, temperature=0.3, max_tokens=2000)
        
        state["answer"] = answer
        state["messages"].append(AIMessage(content=answer))
        
        return state
    
    def should_continue_node(self, state: RAGState) -> str:
        search_count: int = state["search_count"]
        max_searches: int = state["max_searches"]
        needs_more_info: bool = state["needs_more_info"]
        
        if search_count >= max_searches:
            return "generate"
        
        if needs_more_info:
            return "retrieve"
        
        return "generate"
    
    def refine_query_node(self, state: RAGState) -> RAGState:
        query: str = state["query"]
        answer: str = state.get("answer", "")
        
        if not answer:
            return state
        
        prompt: str = f"""原始问题: {query}

之前的回答: {answer}

请基于之前的回答，生成一个更具体、更详细的查询，以获取更多相关信息。只输出新的查询，不要其他内容。"""
        
        refined_query: str = self.llm.generate(prompt, temperature=0.5, max_tokens=200)
        
        state["query"] = refined_query
        state["search_count"] += 1
        
        return state
    
    def build_graph(self):
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("evaluate", self.evaluate_retrieval_node)
        workflow.add_node("generate", self.generate_answer_node)
        workflow.add_node("refine_query", self.refine_query_node)
        
        workflow.set_entry_point("retrieve")
        
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "evaluate")
        
        workflow.add_conditional_edges(
            "evaluate",
            self.should_continue_node,
            {
                "retrieve": "refine_query",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("refine_query", "retrieve")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def query(self, query: str, max_searches: int = 2) -> Dict[str, Any]:
        initial_state: RAGState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": "",
            "needs_more_info": False,
            "search_count": 0,
            "max_searches": max_searches,
            "context": ""
        }
        
        graph = self.build_graph()
        result: RAGState = graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "retrieved_docs": result["reranked_docs"],
            "search_count": result["search_count"]
        }
