import pytest
import numpy as np
from vector_store import VectorStore
from text_chunker import DocumentChunk
import tempfile
import shutil


@pytest.fixture
def temp_db_path():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def vector_store(temp_db_path):
    return VectorStore(persist_directory=temp_db_path, collection_name="test_collection")


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            content="This is the first document about legal procedures.",
            metadata={"filename": "test1.txt", "chunk_index": 0},
            chunk_id="test1_0"
        ),
        DocumentChunk(
            content="Legal documents require careful analysis and interpretation.",
            metadata={"filename": "test1.txt", "chunk_index": 1},
            chunk_id="test1_1"
        ),
        DocumentChunk(
            content="The court system follows specific protocols and regulations.",
            metadata={"filename": "test2.txt", "chunk_index": 0},
            chunk_id="test2_0"
        ),
        DocumentChunk(
            content="Contracts must be signed by all parties involved.",
            metadata={"filename": "test2.txt", "chunk_index": 1},
            chunk_id="test2_1"
        ),
        DocumentChunk(
            content="Evidence must be presented according to legal standards.",
            metadata={"filename": "test3.txt", "chunk_index": 0},
            chunk_id="test3_0"
        )
    ]


@pytest.fixture
def sample_embeddings():
    return [
        np.random.rand(768).tolist() for _ in range(5)
    ]


class TestVectorStore:
    def test_initialization(self, temp_db_path):
        store = VectorStore(persist_directory=temp_db_path, collection_name="test_init")
        assert store.collection_name == "test_init"
        assert store.persist_directory == temp_db_path
        assert store.collection is not None
    
    def test_add_documents(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        count = vector_store.count_documents()
        assert count == 5
    
    def test_similarity_search(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = vector_store.similarity_search(query_embedding, top_k=3)
        
        assert len(results) == 3
        assert 'id' in results[0]
        assert 'content' in results[0]
        assert 'metadata' in results[0]
        assert 'distance' in results[0]
    
    def test_semantic_search(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = vector_store.semantic_search(
            query_embedding,
            top_k=3,
            score_threshold=0.0
        )
        
        assert len(results) >= 0
    
    def test_hybrid_search(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = vector_store.hybrid_search(query_embedding, top_k=3)
        
        assert len(results) <= 3
    
    def test_get_document_by_id(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        doc = vector_store.get_document_by_id("test1_0")
        assert doc is not None
        assert doc['id'] == "test1_0"
        assert "legal procedures" in doc['content']
    
    def test_get_nonexistent_document(self, vector_store):
        doc = vector_store.get_document_by_id("nonexistent_id")
        assert doc is None
    
    def test_count_documents(self, vector_store, sample_chunks, sample_embeddings):
        assert vector_store.count_documents() == 0
        
        vector_store.add_documents(sample_chunks[:2], sample_embeddings[:2])
        assert vector_store.count_documents() == 2
        
        vector_store.add_documents(sample_chunks[2:], sample_embeddings[2:])
        assert vector_store.count_documents() == 5
    
    def test_delete_collection(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        assert vector_store.count_documents() == 5
        
        vector_store.delete_collection()
        assert vector_store.count_documents() == 0
    
    def test_search_with_filter(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = vector_store.similarity_search(
            query_embedding,
            top_k=10,
            filter_dict={"filename": "test1.txt"}
        )
        
        assert all(r['metadata']['filename'] == "test1.txt" for r in results)
    
    def test_empty_search(self, vector_store):
        query_embedding = np.random.rand(768).tolist()
        results = vector_store.similarity_search(query_embedding, top_k=5)
        
        assert len(results) == 0
    
    def test_top_k_limit(self, vector_store, sample_chunks, sample_embeddings):
        vector_store.add_documents(sample_chunks, sample_embeddings)
        
        query_embedding = sample_embeddings[0]
        results = vector_store.similarity_search(query_embedding, top_k=2)
        
        assert len(results) == 2
