from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from text_chunker import DocumentChunk


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "law_documents") -> None:
        self.persist_directory: str = persist_directory
        self.collection_name: str = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> None:
        ids: List[str] = [chunk.chunk_id for chunk in chunks]
        documents: List[str] = [chunk.content for chunk in chunks]
        metadatas: List[Dict[str, Any]] = [chunk.metadata for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        formatted_results: List[Dict[str, Any]] = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = self.similarity_search(query_embedding, top_k * 2, filter_dict)
        
        filtered_results: List[Dict[str, Any]] = [
            result for result in results
            if (1 - result['distance']) >= score_threshold
        ]
        
        return filtered_results[:top_k]
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        similarity_results: List[Dict[str, Any]] = self.similarity_search(query_embedding, top_k, filter_dict)
        semantic_results: List[Dict[str, Any]] = self.semantic_search(query_embedding, top_k, 0.6, filter_dict)
        
        combined: Dict[str, Dict[str, Any]] = {}
        for result in similarity_results:
            combined[result['id']] = result
            combined[result['id']]['similarity_rank'] = similarity_results.index(result)
        
        for result in semantic_results:
            if result['id'] in combined:
                combined[result['id']]['semantic_rank'] = semantic_results.index(result)
            else:
                combined[result['id']] = result
                combined[result['id']]['similarity_rank'] = len(similarity_results)
                combined[result['id']]['semantic_rank'] = semantic_results.index(result)
        
        final_results: List[Dict[str, Any]] = list(combined.values())
        final_results.sort(key=lambda x: (
            x.get('similarity_rank', 999) + x.get('semantic_rank', 999)
        ))
        
        return final_results[:top_k]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        results = self.collection.get(ids=[doc_id])
        
        if results['ids']:
            return {
                'id': results['ids'][0],
                'content': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
        return None
    
    def delete_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count_documents(self) -> int:
        return self.collection.count()
