from typing import List, Optional
from openai import OpenAI
import numpy as np


class OpenAIEmbeddings:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        auth_header: Optional[str] = None,
        auth_scheme: Optional[str] = None
    ) -> None:
        self.api_key: str = api_key
        self.model: str = model
        
        client_kwargs = {"api_key": api_key}
        
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip('/') + "/"
        
        if auth_header and auth_scheme:
            client_kwargs["default_headers"] = {
                auth_header: f"{auth_scheme} {api_key}"
            }
        
        self.client: OpenAI = OpenAI(**client_kwargs)
    
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
