from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np


class Reranker:
    def __init__(self, method: str = "score", cohere_api_key: Optional[str] = None) -> None:
        self.method: str = method
        self.cohere_api_key: Optional[str] = cohere_api_key
        self.cross_encoder: Optional[Any] = None
        self._cohere_client: Optional[Any] = None
    
    def score_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        if weights is None:
            weights = {'distance': 0.7, 'length': 0.3}
        
        for result in results:
            distance_score: float = 1 - result['distance']
            length_score: float = min(len(result['content']) / 500, 1.0)
            
            final_score: float = (
                weights['distance'] * distance_score +
                weights['length'] * length_score
            )
            result['rerank_score'] = final_score
        
        reranked: List[Dict[str, Any]] = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        return reranked
    
    def bm25_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        tokenized_corpus: List[List[str]] = [
            result['content'].lower().split()
            for result in results
        ]
        bm25: BM25Okapi = BM25Okapi(tokenized_corpus)
        
        tokenized_query: List[str] = query.lower().split()
        doc_scores: np.ndarray = bm25.get_scores(tokenized_query)
        
        for i, result in enumerate(results):
            result['bm25_score'] = float(doc_scores[i])
        
        reranked: List[Dict[str, Any]] = sorted(results, key=lambda x: x['bm25_score'], reverse=True)
        return reranked[:top_k]
    
    def cross_encoder_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        from sentence_transformers import CrossEncoder
        
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(model_name)
        
        pairs: List[List[str]] = [[query, result['content']] for result in results]
        scores: np.ndarray = self.cross_encoder.predict(pairs)
        
        for i, result in enumerate(results):
            result['cross_encoder_score'] = float(scores[i])
        
        reranked: List[Dict[str, Any]] = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
        return reranked[:top_k]
    
    def cohere_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int = 5,
        model: str = "rerank-multilingual-v3.0"
    ) -> List[Dict[str, Any]]:
        import cohere
        
        if not self.cohere_api_key:
            raise ValueError("Cohere API key is required for Cohere reranking")
        
        if self._cohere_client is None:
            self._cohere_client = cohere.Client(self.cohere_api_key)
        
        documents = [
            {"text": result['content']}
            for result in results
        ]
        
        response = self._cohere_client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=True
        )
        
        reranked_results = []
        for result in response.results:
            original_result = results[result.index]
            original_result['cohere_score'] = result.relevance_score
            reranked_results.append(original_result)
        
        return reranked_results
    
    def hybrid_rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int = 5,
        use_bm25: bool = True,
        use_cross_encoder: bool = False
    ) -> List[Dict[str, Any]]:
        if use_bm25:
            results = self.bm25_rerank(results, query, len(results))
        
        if use_cross_encoder:
            results = self.cross_encoder_rerank(results, query, top_k=len(results))
        
        results = self.score_rerank(results, query)
        
        return results[:top_k]
    
    def rerank(
        self,
        results: List[Dict[str, Any]],
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if self.method == "score":
            return self.score_rerank(results, query, kwargs.get('weights'))[:top_k]
        elif self.method == "bm25":
            return self.bm25_rerank(results, query, top_k)
        elif self.method == "cross_encoder":
            return self.cross_encoder_rerank(results, query, kwargs.get('model_name', "ms-marco-MiniLM-L-6-v2"), top_k)
        elif self.method == "cohere":
            return self.cohere_rerank(results, query, top_k, kwargs.get('model', "rerank-multilingual-v3.0"))
        elif self.method == "hybrid":
            return self.hybrid_rerank(results, query, top_k, kwargs.get('use_bm25', True), kwargs.get('use_cross_encoder', False))
        else:
            raise ValueError(f"Unknown reranking method: {self.method}")
