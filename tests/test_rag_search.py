import pytest
from unittest.mock import Mock, MagicMock, patch
from embeddings import OpenAIEmbeddings
from vector_store import VectorStore
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
def mock_embeddings():
    def create_mock_response(texts):
        response = MagicMock()
        response.data = [
            MagicMock(embedding=np.random.rand(1536).tolist()) for _ in texts
        ]
        return response
    
    with patch('embeddings.OpenAI.embeddings.create') as mock:
        mock.side_effect = create_mock_response
        yield mock


@pytest.fixture
def mock_llm():
    with patch('llm.OpenAI.chat.completions.create') as mock:
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content='This is a test answer based on the retrieved documents.'))
        ]
        mock.return_value = response
        yield mock


@pytest.fixture
def embeddings_instance():
    return OpenAIEmbeddings(api_key="test_key", model="text-embedding-3-small")


@pytest.fixture
def llm_instance():
    return OpenAILLM(api_key="test_key", model="gpt-4o-mini")


@pytest.fixture
def vector_store(temp_db_path):
    return VectorStore(persist_directory=temp_db_path, collection_name="test_rag")


@pytest.fixture
def reranker():
    return Reranker(method="hybrid")


@pytest.fixture
def sample_search_results():
    return [
        {
            'id': 'doc1_0',
            'content': 'Legal procedures require careful documentation.',
            'metadata': {'filename': 'doc1.txt', 'chunk_index': 0},
            'distance': 0.1
        },
        {
            'id': 'doc1_1',
            'content': 'Court systems follow established protocols.',
            'metadata': {'filename': 'doc1.txt', 'chunk_index': 1},
            'distance': 0.2
        },
        {
            'id': 'doc2_0',
            'content': 'Contracts must be legally binding.',
            'metadata': {'filename': 'doc2.txt', 'chunk_index': 0},
            'distance': 0.3
        }
    ]


class TestEmbeddings:
    def test_embed_text(self, embeddings_instance, mock_embeddings):
        result = embeddings_instance.embed_text("Test text")
        
        assert isinstance(result, list)
        assert len(result) == 1536
        mock_embeddings.assert_called_once()
    
    def test_embed_documents(self, embeddings_instance, mock_embeddings):
        texts = ["Text 1", "Text 2", "Text 3"]
        results = embeddings_instance.embed_documents(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
    
    def test_embed_query(self, embeddings_instance, mock_embeddings):
        result = embeddings_instance.embed_query("Test query")
        
        assert isinstance(result, list)
        assert len(result) == 1536
    
    def test_embedding_failure(self, embeddings_instance):
        with patch('embeddings.OpenAI.embeddings.create') as mock:
            mock.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                embeddings_instance.embed_text("Test")


class TestLLM:
    def test_generate(self, llm_instance, mock_llm):
        result = llm_instance.generate("Test prompt")
        
        assert isinstance(result, str)
        assert "test answer" in result.lower()
        mock_llm.assert_called_once()
    
    def test_generate_with_temperature(self, llm_instance, mock_llm):
        result = llm_instance.generate(
            "Test prompt",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert isinstance(result, str)
    
    def test_generate_with_history(self, llm_instance, mock_llm):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = llm_instance.generate_with_history(messages)
        
        assert isinstance(result, str)
    
    def test_llm_failure(self, llm_instance):
        with patch('llm.OpenAI.chat.completions.create') as mock:
            mock.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                llm_instance.generate("Test prompt")


class TestReranker:
    def test_score_rerank(self, reranker, sample_search_results):
        query = "What are legal procedures?"
        results = reranker.score_rerank(sample_search_results, query)
        
        assert len(results) == 3
        assert 'rerank_score' in results[0]
        assert results == sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    def test_bm25_rerank(self, reranker, sample_search_results):
        query = "legal procedures"
        results = reranker.bm25_rerank(sample_search_results, query, top_k=2)
        
        assert len(results) == 2
        assert 'bm25_score' in results[0]
    
    def test_hybrid_rerank(self, reranker, sample_search_results):
        query = "legal procedures"
        results = reranker.hybrid_rerank(sample_search_results, query, top_k=2)
        
        assert len(results) == 2
    
    def test_rerank_with_custom_weights(self, reranker, sample_search_results):
        query = "Test query"
        weights = {'distance': 0.5, 'length': 0.5}
        results = reranker.score_rerank(sample_search_results, query, weights)
        
        assert len(results) == 3
        assert 'rerank_score' in results[0]
    
    def test_rerank_empty_results(self, reranker):
        query = "Test query"
        results = reranker.score_rerank([], query)
        
        assert len(results) == 0


class TestRAGSearch:
    def test_basic_search_flow(self, embeddings_instance, vector_store, reranker):
        from text_chunker import DocumentChunk
        
        chunks = [
            DocumentChunk(
                content="Legal documents require careful analysis.",
                metadata={"filename": "test.txt"},
                chunk_id="test_0"
            )
        ]
        
        embeddings = [np.random.rand(1536).tolist()]
        vector_store.add_documents(chunks, embeddings)
        
        query_embedding = embeddings_instance.embed_query("legal documents")
        results = vector_store.similarity_search(query_embedding, top_k=5)
        
        assert len(results) >= 1
        assert 'content' in results[0]
    
    def test_search_and_rerank_pipeline(self, embeddings_instance, vector_store, reranker):
        from text_chunker import DocumentChunk
        
        chunks = [
            DocumentChunk(
                content="Court procedures must follow legal standards.",
                metadata={"filename": "test.txt"},
                chunk_id="test_0"
            ),
            DocumentChunk(
                content="Evidence collection requires proper documentation.",
                metadata={"filename": "test.txt"},
                chunk_id="test_1"
            )
        ]
        
        embeddings = [np.random.rand(1536).tolist() for _ in range(2)]
        vector_store.add_documents(chunks, embeddings)
        
        query_embedding = embeddings_instance.embed_query("court procedures")
        initial_results = vector_store.similarity_search(query_embedding, top_k=10)
        
        reranked = reranker.rerank(initial_results, "court procedures", top_k=3)
        
        assert len(reranked) == 3
        assert len(reranked) <= len(initial_results)
    
    def test_search_with_filter(self, embeddings_instance, vector_store):
        from text_chunker import DocumentChunk
        
        chunks = [
            DocumentChunk(
                content="Document from file 1",
                metadata={"filename": "file1.txt"},
                chunk_id="file1_0"
            ),
            DocumentChunk(
                content="Document from file 2",
                metadata={"filename": "file2.txt"},
                chunk_id="file2_0"
            )
        ]
        
        embeddings = [np.random.rand(1536).tolist() for _ in range(2)]
        vector_store.add_documents(chunks, embeddings)
        
        query_embedding = embeddings_instance.embed_query("document")
        results = vector_store.similarity_search(
            query_embedding,
            top_k=10,
            filter_dict={"filename": "file1.txt"}
        )
        
        assert all(r['metadata']['filename'] == "file1.txt" for r in results)
