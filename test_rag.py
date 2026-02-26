"""
Test script for the RAG system
"""
from main import RAGSystem


def test_basic_query():
    print("Testing basic query...")
    
    rag_system = RAGSystem()
    
    data_directory = r"d:\from old pc\MO_Law_checking"
    
    print("Loading documents...")
    documents = rag_system.load_documents([data_directory])
    print(f"Loaded {len(documents)} documents.")
    
    print("\nBuilding vector database...")
    rag_system.build_vector_database(documents, rebuild=False)
    
    test_queries = [
        "什么是澳门基本法?",
        "澳门刑法典的主要内容是什么?",
        "What are the requirements for business registration in Macau?"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)
        
        result = rag_system.query(query, max_searches=2)
        
        print("\nAnswer:")
        print(result['answer'])
        
        print(f"\nSearch iterations: {result['search_count']}")
        print(f"Documents retrieved: {len(result['retrieved_docs'])}")


if __name__ == "__main__":
    test_basic_query()
