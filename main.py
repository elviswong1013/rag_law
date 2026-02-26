import os
from dotenv import load_dotenv
from document_loader import DocumentLoader
from text_chunker import TextChunker
from embeddings import OpenAIEmbeddings
from vector_store import VectorStore
from reranker import Reranker
from llm import OpenAILLM
from agentic_rag import AgenticRAG
from tqdm import tqdm


class RAGSystem:
    def __init__(self, config: dict = None) -> None:
        load_dotenv()
        
        self.config: dict = config or self._load_config()
        
        self.api_key: str = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", 512))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))
        self.top_k: int = int(os.getenv("TOP_K_RETRIEVAL", 10))
        self.rerank_top_k: int = int(os.getenv("RERANK_TOP_K", 5))
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(self.api_key, self.embedding_model)
        self.vector_store: VectorStore = VectorStore(self.db_path)
        self.reranker: Reranker = Reranker(method="hybrid")
        self.llm: OpenAILLM = OpenAILLM(self.api_key, self.llm_model)
        
        self.document_loader: DocumentLoader = DocumentLoader()
        self.text_chunker: TextChunker = TextChunker(self.chunk_size, self.chunk_overlap)
        
        self.agentic_rag: AgenticRAG = AgenticRAG(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            reranker=self.reranker,
            llm=self.llm,
            top_k=self.top_k,
            rerank_top_k=self.rerank_top_k
        )
    
    def _load_config(self) -> dict:
        return {}
    
    def load_documents(self, paths: list) -> list:
        documents: list = []
        
        for path in paths:
            if os.path.isfile(path):
                try:
                    doc: dict = self.document_loader.load_file(path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading file {path}: {e}")
            elif os.path.isdir(path):
                try:
                    docs: list = self.document_loader.load_directory(path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading directory {path}: {e}")
        
        return documents
    
    def build_vector_database(self, documents: list, rebuild: bool = False) -> None:
        if rebuild:
            self.vector_store.delete_collection()
        
        if self.vector_store.count_documents() > 0:
            print(f"Vector database already contains {self.vector_store.count_documents()} documents.")
            print("Set rebuild=True to rebuild the database.")
            return
        
        print("Chunking documents...")
        chunks: list = self.text_chunker.chunk_documents(documents, method='paragraph')
        print(f"Created {len(chunks)} chunks.")
        
        print("Generating embeddings...")
        texts: list = [chunk.content for chunk in chunks]
        embeddings: list = []
        
        batch_size: int = 25
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch: list = texts[i:i + batch_size]
            batch_embeddings: list = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
        print("Adding documents to vector store...")
        self.vector_store.add_documents(chunks, embeddings)
        
        print(f"Vector database built successfully with {len(chunks)} documents.")
    
    def query(self, question: str, max_searches: int = 2) -> dict:
        result: dict = self.agentic_rag.query(question, max_searches)
        return result
    
    def interactive_query(self) -> None:
        print("\n" + "="*50)
        print("Legal RAG System - Interactive Mode")
        print("="*50)
        print("Type 'quit' or 'exit' to exit\n")
        
        while True:
            question: str = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching...")
            result: dict = self.query(question)
            
            print("\n" + "="*50)
            print("Answer:")
            print("="*50)
            print(result['answer'])
            
            print("\n" + "="*50)
            print(f"Search iterations: {result['search_count']}")
            print(f"Documents retrieved: {len(result['retrieved_docs'])}")
            print("="*50 + "\n")


def main() -> None:
    import sys
    
    data_directory: str = r"d:\from old pc\MO_Law_checking"
    
    rag_system: RAGSystem = RAGSystem()
    
    print("Loading documents...")
    documents: list = rag_system.load_documents([data_directory])
    print(f"Loaded {len(documents)} documents.")
    
    print("\nBuilding vector database...")
    rag_system.build_vector_database(documents, rebuild=False)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        rag_system.interactive_query()
    else:
        while True:
            question: str = input("\nEnter your question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result: dict = rag_system.query(question)
            
            print("\nAnswer:")
            print(result['answer'])
            print(f"\nSearch iterations: {result['search_count']}")


if __name__ == "__main__":
    main()
