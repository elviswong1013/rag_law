from typing import List, Dict, Any, Optional
from openai import OpenAI


class OpenAILLM:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
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
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.8
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        return response.choices[0].message.content
    
    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.8
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        return response.choices[0].message.content
    
    def stream_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.8
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
