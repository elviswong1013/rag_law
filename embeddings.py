from typing import List
from openai import OpenAI
import numpy as np


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.client: OpenAI = OpenAI(api_key=api_key)
    
    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        batch_size: int = 25
        
        for i in range(0, len(texts), batch_size):
            batch: List[str] = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            batch_embeddings: List[List[float]] = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=query
        )
        
        return response.data[0].embedding
