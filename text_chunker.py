from typing import List, Dict, Any
import re
import gc
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class TextChunker:
    MAX_CHUNK_SIZE: int = 1500
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.chunk_size: int = min(chunk_size, self.MAX_CHUNK_SIZE)
        self.chunk_overlap: int = min(chunk_overlap, self.chunk_size // 2)
    
    def chunk_by_size(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        
        if not text or not text.strip():
            return chunks
            
        words: List[str] = text.split()
        
        if not words:
            return chunks
        
        start_idx: int = 0
        chunk_num: int = 0
        
        while start_idx < len(words):
            end_idx: int = min(start_idx + self.chunk_size, len(words))
            chunk_words: List[str] = words[start_idx:end_idx]
            chunk_content: str = ' '.join(chunk_words)
            
            chunk: DocumentChunk = DocumentChunk(
                content=chunk_content,
                metadata={
                    **metadata,
                    'chunk_index': chunk_num,
                    'chunk_type': 'size_based'
                },
                chunk_id=f"{metadata.get('filename', 'unknown')}_{chunk_num}"
            )
            chunks.append(chunk)
            
            start_idx = end_idx - self.chunk_overlap
            chunk_num += 1
            
            if chunk_num > 10000:
                break
        
        return chunks
    
    def chunk_by_paragraph(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        paragraphs: List[str] = re.split(r'\n\s*\n', text)
        chunks: List[DocumentChunk] = []
        current_chunk: str = ""
        chunk_num: int = 0
        
        for para in paragraphs:
            para: str = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunk: DocumentChunk = DocumentChunk(
                        content=current_chunk,
                        metadata={
                            **metadata,
                            'chunk_index': chunk_num,
                            'chunk_type': 'paragraph_based'
                        },
                        chunk_id=f"{metadata.get('filename', 'unknown')}_{chunk_num}"
                    )
                    chunks.append(chunk)
                    chunk_num += 1
                
                if len(para) > self.chunk_size:
                    sub_chunks: List[DocumentChunk] = self.chunk_by_size(para, metadata)
                    for sub_chunk in sub_chunks:
                        sub_chunk.metadata['chunk_type'] = 'paragraph_subchunk'
                        sub_chunk.chunk_id = f"{metadata.get('filename', 'unknown')}_{chunk_num}"
                        chunks.append(sub_chunk)
                        chunk_num += 1
                    current_chunk = ""
                else:
                    current_chunk = para
        
        if current_chunk:
            chunk: DocumentChunk = DocumentChunk(
                content=current_chunk,
                metadata={
                    **metadata,
                    'chunk_index': chunk_num,
                    'chunk_type': 'paragraph_based'
                },
                chunk_id=f"{metadata.get('filename', 'unknown')}_{chunk_num}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]], method: str = 'paragraph') -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []
        
        for i, doc in enumerate(documents):
            content: str = doc['content']
            metadata: Dict[str, Any] = doc['metadata']
            metadata['file_path'] = doc['file_path']
            metadata['file_type'] = doc['file_type']
            
            if method == 'size':
                chunks: List[DocumentChunk] = self.chunk_by_size(content, metadata)
            elif method == 'paragraph':
                chunks: List[DocumentChunk] = self.chunk_by_paragraph(content, metadata)
            else:
                raise ValueError(f"Unknown chunking method: {method}")
            
            all_chunks.extend(chunks)
            
            if i > 0 and i % 50 == 0:
                gc.collect()
        
        return all_chunks
