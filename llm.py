from typing import List, Dict, Any, Optional
from openai import OpenAI


class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.client: OpenAI = OpenAI(api_key=api_key)
    
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
