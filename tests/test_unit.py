import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import tempfile
import os


class TestVectorDatabaseUnit:
    """Unit tests for vector database functionality without actual ChromaDB"""
    
    def test_vector_store_initialization(self):
        """Test that vector store can be initialized"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_db = MagicMock()
            mock_client.return_value = mock_db
            mock_db.get_or_create_collection.return_value = MagicMock()
            
            from vector_store import VectorStore
            store = VectorStore(persist_directory="./test_db", collection_name="test")
            
            assert store.collection_name == "test"
            assert store.persist_directory == "./test_db"
    
    def test_add_documents_structure(self):
        """Test that add_documents has correct structure"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_db = MagicMock()
            mock_client.return_value = mock_db
            mock_db.get_or_create_collection.return_value = mock_collection
            
            from vector_store import VectorStore
            from text_chunker import DocumentChunk
            
            store = VectorStore(persist_directory="./test_db", collection_name="test")
            
            chunks = [
                DocumentChunk(
                    content="Test content",
                    metadata={"filename": "test.txt"},
                    chunk_id="test_0"
                )
            ]
            embeddings = [np.random.rand(1536).tolist()]
            
            store.add_documents(chunks, embeddings)
            
            mock_collection.add.assert_called_once()
    
    def test_similarity_search_structure(self):
        """Test that similarity search has correct structure"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'ids': [['doc1', 'doc2']],
                'documents': [['Content 1', 'Content 2']],
                'metadatas': [[{'filename': 'test1.txt'}, {'filename': 'test2.txt'}]],
                'distances': [[0.1, 0.2]]
            }
            mock_db = MagicMock()
            mock_client.return_value = mock_db
            mock_db.get_or_create_collection.return_value = mock_collection
            
            from vector_store import VectorStore
            store = VectorStore(persist_directory="./test_db", collection_name="test")
            
            query_embedding = np.random.rand(1536).tolist()
            results = store.similarity_search(query_embedding, top_k=2)
            
            assert len(results) == 2
            assert 'id' in results[0]
            assert 'content' in results[0]
            assert 'metadata' in results[0]
            assert 'distance' in results[0]
    
    def test_hybrid_search_combines_results(self):
        """Test that hybrid search combines similarity and semantic results"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                'ids': [['doc1']],
                'documents': [['Content 1']],
                'metadatas': [[{'filename': 'test1.txt'}]],
                'distances': [[0.1]]
            }
            mock_db = MagicMock()
            mock_client.return_value = mock_db
            mock_db.get_or_create_collection.return_value = mock_collection
            
            from vector_store import VectorStore
            store = VectorStore(persist_directory="./test_db", collection_name="test")
            
            query_embedding = np.random.rand(1536).tolist()
            results = store.hybrid_search(query_embedding, top_k=1)
            
            assert len(results) <= 1


class TestEmbeddingsUnit:
    """Unit tests for embeddings functionality"""
    
    def test_embeddings_initialization(self):
        """Test that embeddings can be initialized"""
        with patch('embeddings.OpenAI') as mock_openai:
            from embeddings import OpenAIEmbeddings
            embedder = OpenAIEmbeddings(api_key="test_key", model="test-model")
            
            assert embedder.api_key == "test_key"
            assert embedder.model == "test-model"
    
    def test_embed_text_structure(self):
        """Test that embed_text returns correct structure"""
        with patch('embeddings.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=np.random.rand(1536).tolist())]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            from embeddings import OpenAIEmbeddings
            embedder = OpenAIEmbeddings(api_key="test_key")
            
            result = embedder.embed_text("Test text")
            
            assert isinstance(result, list)
            assert len(result) == 1536
    
    def test_embed_documents_batch(self):
        """Test that embed_documents handles batch processing"""
        with patch('embeddings.OpenAI') as mock_openai:
            mock_client = MagicMock()
            
            def create_response(model, input):
                return MagicMock(data=[
                    MagicMock(embedding=np.random.rand(1536).tolist()) for _ in (input if isinstance(input, list) else [input])
                ])
            
            mock_client.embeddings.create.side_effect = create_response
            mock_openai.return_value = mock_client
            
            from embeddings import OpenAIEmbeddings
            embedder = OpenAIEmbeddings(api_key="test_key")
            
            texts = ["Text 1", "Text 2", "Text 3"]
            results = embedder.embed_documents(texts)
            
            assert len(results) == 3
            assert all(isinstance(r, list) for r in results)


class TestRerankerUnit:
    """Unit tests for reranking functionality"""
    
    def test_score_rerank(self):
        """Test score-based reranking"""
        from reranker import Reranker
        
        reranker = Reranker(method="score")
        
        results = [
            {'id': '1', 'content': 'Short text', 'distance': 0.3},
            {'id': '2', 'content': 'A much longer text with more content', 'distance': 0.1},
            {'id': '3', 'content': 'Medium length text here', 'distance': 0.2}
        ]
        
        reranked = reranker.score_rerank(results, "test query")
        
        assert len(reranked) == 3
        assert 'rerank_score' in reranked[0]
        assert reranked == sorted(reranked, key=lambda x: x['rerank_score'], reverse=True)
    
    def test_bm25_rerank(self):
        """Test BM25-based reranking"""
        from reranker import Reranker
        
        reranker = Reranker(method="bm25")
        
        results = [
            {'id': '1', 'content': 'Legal procedures require documentation', 'distance': 0.1},
            {'id': '2', 'content': 'Court systems follow protocols', 'distance': 0.2},
            {'id': '3', 'content': 'Contracts must be binding', 'distance': 0.3}
        ]
        
        reranked = reranker.bm25_rerank(results, "legal procedures", top_k=2)
        
        assert len(reranked) == 2
        assert 'bm25_score' in reranked[0]
    
    def test_hybrid_rerank(self):
        """Test hybrid reranking"""
        from reranker import Reranker
        
        reranker = Reranker(method="hybrid")
        
        results = [
            {'id': '1', 'content': 'Legal procedures require documentation', 'distance': 0.1},
            {'id': '2', 'content': 'Court systems follow protocols', 'distance': 0.2}
        ]
        
        reranked = reranker.hybrid_rerank(results, "legal", top_k=2)
        
        assert len(reranked) <= 2


class TestLLMUnit:
    """Unit tests for LLM functionality"""
    
    def test_llm_initialization(self):
        """Test that LLM can be initialized"""
        with patch('llm.OpenAI') as mock_openai:
            from llm import OpenAILLM
            llm = OpenAILLM(api_key="test_key", model="test-model")
            
            assert llm.api_key == "test_key"
            assert llm.model == "test-model"
    
    def test_generate_structure(self):
        """Test that generate returns correct structure"""
        with patch('llm.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='Test response'))
            ]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            from llm import OpenAILLM
            llm = OpenAILLM(api_key="test_key")
            
            result = llm.generate("Test prompt")
            
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_generate_with_parameters(self):
        """Test that generate accepts parameters"""
        with patch('llm.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='Test response'))
            ]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            from llm import OpenAILLM
            llm = OpenAILLM(api_key="test_key")
            
            result = llm.generate(
                "Test prompt",
                temperature=0.5,
                max_tokens=1000,
                top_p=0.9
            )
            
            assert isinstance(result, str)


class TestTextChunkerUnit:
    """Unit tests for text chunking functionality"""
    
    def test_chunk_by_size(self):
        """Test size-based chunking"""
        from text_chunker import TextChunker, DocumentChunk
        
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        
        chunks = chunker.chunk_by_size(text, {"filename": "test.txt"})
        
        assert len(chunks) > 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all('chunk_index' in c.metadata for c in chunks)
    
    def test_chunk_by_paragraph(self):
        """Test paragraph-based chunking"""
        from text_chunker import TextChunker, DocumentChunk
        
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        
        chunks = chunker.chunk_by_paragraph(text, {"filename": "test.txt"})
        
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_chunk_documents(self):
        """Test chunking multiple documents"""
        from text_chunker import TextChunker
        
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        
        documents = [
            {
                'content': 'Document 1 content here',
                'metadata': {'filename': 'doc1.txt', 'file_type': '.txt'}
            },
            {
                'content': 'Document 2 content here',
                'metadata': {'filename': 'doc2.txt', 'file_type': '.txt'}
            }
        ]
        
        chunks = chunker.chunk_documents(documents, method='size')
        
        assert len(chunks) >= 2
        assert all('file_path' in c.metadata for c in chunks)


class TestDocumentLoaderUnit:
    """Unit tests for document loading functionality"""
    
    def test_load_text_file(self):
        """Test loading text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content\nLine 2\nLine 3")
            temp_path = f.name
        
        try:
            from document_loader import DocumentLoader
            loader = DocumentLoader()
            result = loader.load_file(temp_path)
            
            assert 'content' in result
            assert 'Test content' in result['content']
            assert result['file_type'] == '.txt'
        finally:
            os.unlink(temp_path)
    
    def test_load_directory_structure(self):
        """Test directory loading structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            from document_loader import DocumentLoader
            
            loader = DocumentLoader()
            documents = loader.load_directory(temp_dir)
            
            assert isinstance(documents, list)
