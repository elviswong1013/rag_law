import os
from dotenv import load_dotenv
from document_loader import DocumentLoader
from text_chunker import TextChunker
from embeddings import OpenAIEmbeddings
from vector_store import VectorStore
from reranker import Reranker
from llm import OpenAILLM
from agentic_rag import AgenticRAG
from config import PipelineConfig
from tqdm import tqdm
import gc


class RAGSystem:
    def __init__(self, config: dict = None) -> None:
        llm_settings_path = os.path.join(os.path.dirname(__file__), 'llm_settings.env')
        if os.path.exists(llm_settings_path):
            load_dotenv(llm_settings_path)
        else:
            load_dotenv()
        
        self.config: PipelineConfig = PipelineConfig.from_env()
        
        llm_cfg = self.config.llm
        embed_cfg = self.config.embedding
        
        if not llm_cfg.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        
        self.chunk_size: int = self.config.chunking.chunk_size
        self.chunk_overlap: int = self.config.chunking.chunk_overlap
        self.top_k: int = self.config.retrieval.top_k
        self.rerank_top_k: int = self.config.reranker.top_k
        self.embedding_model: str = embed_cfg.model
        self.llm_model: str = llm_cfg.model
        self.db_path: str = self.config.vector_db.persist_directory
        
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            api_key=llm_cfg.api_key,
            model=self.embedding_model,
            base_url=llm_cfg.api_base,
            auth_header=llm_cfg.auth_header,
            auth_scheme=llm_cfg.auth_scheme
        )
        self.vector_store: VectorStore = VectorStore(self.db_path)
        self.reranker: Reranker = Reranker(method=self.config.reranker.method)
        self.llm: OpenAILLM = OpenAILLM(
            api_key=llm_cfg.api_key,
            model=self.llm_model,
            base_url=llm_cfg.api_base,
            auth_header=llm_cfg.auth_header,
            auth_scheme=llm_cfg.auth_scheme
        )
        
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
        
        print("Generating embeddings and saving to vector store...")
        
        BATCH_SIZE: int = 10
        SAVE_BATCH: int = 500
        
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing"):
            batch_chunks: list = chunks[i:i + BATCH_SIZE]
            batch_texts: list = [chunk.content for chunk in batch_chunks]
            
            batch_embeddings: list = self.embeddings.embed_documents(batch_texts)
            self.vector_store.add_documents(batch_chunks, batch_embeddings)
            
            del batch_texts
            del batch_embeddings
            
            if (i + BATCH_SIZE) % SAVE_BATCH == 0:
                gc.collect()
        
        del chunks
        gc.collect()
        
        print(f"Vector database built successfully.")
    
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
