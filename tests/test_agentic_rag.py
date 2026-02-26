import pytest
from unittest.mock import Mock, MagicMock, patch
from agentic_rag import AgenticRAG
from state import RAGState
from vector_store import VectorStore
from embeddings import OpenAIEmbeddings
from reranker import Reranker
from llm import OpenAILLM
import numpy as np
import tempfile
import shutil


@pytest.fixture
def temp_db_path():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def mock_vector_store():
    store = Mock(spec=VectorStore)
    store.similarity_search.return_value = [
        {
            'id': 'doc1_0',
            'content': 'Legal procedures require documentation.',
            'metadata': {'filename': 'doc1.txt'},
            'distance': 0.1
        },
        {
            'id': 'doc1_1',
            'content': 'Court systems follow protocols.',
            'metadata': {'filename': 'doc1.txt'},
            'distance': 0.2
        }
    ]
    store.semantic_search.return_value = [
        {
            'id': 'doc2_0',
            'content': 'Contracts must be binding.',
            'metadata': {'filename': 'doc2.txt'},
            'distance': 0.15
        }
    ]
    return store


@pytest.fixture
def mock_embeddings():
    embeddings = Mock(spec=OpenAIEmbeddings)
    embeddings.embed_query.return_value = np.random.rand(1536).tolist()
    return embeddings


@pytest.fixture
def mock_reranker():
    reranker = Mock(spec=Reranker)
    reranker.rerank.return_value = [
        {
            'id': 'doc1_0',
            'content': 'Legal procedures require documentation.',
            'metadata': {'filename': 'doc1.txt'},
            'distance': 0.1,
            'rerank_score': 0.9
        },
        {
            'id': 'doc2_0',
            'content': 'Contracts must be binding.',
            'metadata': {'filename': 'doc2.txt'},
            'distance': 0.15,
            'rerank_score': 0.85
        }
    ]
    return reranker


@pytest.fixture
def mock_llm():
    llm = Mock(spec=OpenAILLM)
    llm.generate.return_value = "Based on the retrieved documents, legal procedures require proper documentation and court systems must follow established protocols."
    return llm


@pytest.fixture
def agentic_rag(mock_vector_store, mock_embeddings, mock_reranker, mock_llm):
    return AgenticRAG(
        vector_store=mock_vector_store,
        embeddings=mock_embeddings,
        reranker=mock_reranker,
        llm=mock_llm,
        top_k=10,
        rerank_top_k=5
    )


class TestAgenticRAGNodes:
    def test_retrieve_node(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="What are legal procedures?",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.retrieve_node(state)
        
        assert len(result["retrieved_docs"]) > 0
        assert "content" in result["retrieved_docs"][0]
        mock_embeddings.embed_query.assert_called_once_with("What are legal procedures?")
    
    def test_retrieve_node_empty_results(self, agentic_rag):
        agentic_rag.vector_store.similarity_search.return_value = []
        agentic_rag.vector_store.semantic_search.return_value = []
        
        state = RAGState(
            messages=[],
            query="Test query",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.retrieve_node(state)
        
        assert len(result["retrieved_docs"]) == 0
    
    def test_rerank_node(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="legal procedures",
            retrieved                    'id':_docs=[
                {
 'doc1',
                    'content': 'Test content',
                    'metadata': {},
                    'distance': 0.1
                }
            ],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.rerank_node(state)
        
        assert len(result["reranked_docs"]) > 0
        mock_reranker.rerank.assert_called_once()
    
    def test_rerank_node_empty_input(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.rerank_node(state)
        
        assert len(result["reranked_docs"]) == 0
    
    def test_evaluate_retrieval_node_sufficient(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[
                {'id': '1', 'content': 'Doc 1'},
                {'id': '2', 'content': 'Doc 2'},
                {'id': '3', 'content': 'Doc 3'}
            ],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.evaluate_retrieval_node(state)
        
        assert result["needs_more_info"] == False
        assert len(result["context"]) > 0
    
    def test_evaluate_retrieval_node_insufficient(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[
                {'id': '1', 'content': 'Doc 1'}
            ],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.evaluate_retrieval_node(state)
        
        assert result["needs_more_info"] == True
    
    def test_evaluate_retrieval_node_empty(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.evaluate_retrieval_node(state)
        
        assert result["needs_more_info"] == True
        assert result["context"] == ""
    
    def test_generate_answer_node(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="What are legal procedures?",
            retrieved_docs=[],
            reranked_docs=[
                {'id': '1', 'content': 'Legal procedures require documentation.'}
            ],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context="[Document 1]\nLegal procedures require documentation.\n"
        )
        
        result = agentic_rag.generate_answer_node(state)
        
        assert len(result["answer"]) > 0
        assert len(result["messages"]) > 0
        mock_llm.generate.assert_called_once()
    
    def test_generate_answer_node_no_context(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="What are legal procedures?",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.generate_answer_node(state)
        
        assert len(result["answer"]) > 0
    
    def test_refine_query_node(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="What are legal procedures?",
            retrieved_docs=[],
            reranked_docs=[],
            answer="Based on documents, legal procedures require documentation.",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.refine_query_node(state)
        
        assert result["search_count"] == 1
        assert len(result["query"]) > 0
        mock_llm.generate.assert_called_once()
    
    def test_refine_query_node_no_answer(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="What are legal procedures?",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.refine_query_node(state)
        
        assert result["search_count"] == 0
        assert result["query"] == "What are legal procedures?"
    
    def test_should_continue_generate(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[
                {'id': '1'},
                {'id': '2'},
                {'id': '3'}
            ],
            answer="",
            needs_more_info=False,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.should_continue_node(state)
        
        assert result == "generate"
    
    def test_should_continue_retrieve(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=True,
            search_count=0,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.should_continue_node(state)
        
        assert result == "retrieve"
    
    def test_should_continue_max_searches(self, agentic_rag):
        state = RAGState(
            messages=[],
            query="test",
            retrieved_docs=[],
            reranked_docs=[],
            answer="",
            needs_more_info=True,
            search_count=2,
            max_searches=2,
            context=""
        )
        
        result = agentic_rag.should_continue_node(state)
        
        assert result == "generate"


class TestAgenticRAGIntegration:
    def test_build_graph(self, agentic_rag):
        graph = agentic_rag.build_graph()
        
        assert graph is not None
    
    def test_query_single_search(self, agentic_rag):
        agentic_rag.reranker.rerank.return_value = [
            {
                'id': 'doc1_0',
                'content': 'Legal procedures require documentation.',
                'metadata': {'filename': 'doc1.txt'},
                'distance': 0.1,
                'rerank_score': 0.9
            },
            {
                'id': 'doc2_0',
                'content': 'Contracts must be binding.',
                'metadata': {'filename': 'doc2.txt'},
                'distance': 0.15,
                'rerank_score': 0.85
            },
            {
                'id': 'doc3_0',
                'content': 'Court systems follow protocols.',
                'metadata': {'filename': 'doc3.txt'},
                'distance': 0.2,
                'rerank_score': 0.8
            }
        ]
        
        result = agentic_rag.query("What are legal procedures?", max_searches=1)
        
        assert "answer" in result
        assert "retrieved_docs" in result
        assert "search_count" in result
        assert len(result["answer"]) > 0
        assert result["search_count"] >= 0
    
    def test_query_multiple_searches(self, agentic_rag):
        agentic_rag.reranker.rerank.return_value = [
            {'id': '1', 'content': 'Doc 1', 'distance': 0.1}
        ]
        
        result = agentic_rag.query("What are legal procedures?", max_searches=2)
        
        assert "answer" in result
        assert result["search_count"] >= 0
    
    def test_query_with_empty_results(self, agentic_rag):
        agentic_rag.vector_store.similarity_search.return_value = []
        agentic_rag.vector_store.semantic_search.return_value = []
        agentic_rag.reranker.rerank.return_value = []
        
        result = agentic_rag.query("What are legal procedures?", max_searches=1)
        
        assert "answer" in result
        assert len(result["retrieved_docs"]) == 0
    
    def test_query_preserves_max_searches(self, agentic_rag):
        result = agentic_rag.query("Test query", max_searches=3)
        
        assert result["search_count"] <= 3


class TestAgenticRAGEdgeCases:
    def test_empty_query(self, agentic_rag):
        result = agentic_rag.query("", max_searches=1)
        
        assert "answer" in result
    
    def test_very_long_query(self, agentic_rag):
        long_query = "What are legal procedures? " * 100
        result = agentic_rag.query(long_query, max_searches=1)
        
        assert "answer" in result
    
    def test_query_with_special_characters(self, agentic_rag):
        special_query = "What are legal procedures? @#$%^&*()"
        result = agentic_rag.query(special_query, max_searches=1)
        
        assert "answer" in result
    
    def test_max_searches_zero(self, agentic_rag):
        result = agentic_rag.query("Test query", max_searches=0)
        
        assert "answer" in result
        assert result["search_count"] == 0
