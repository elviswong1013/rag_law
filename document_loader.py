from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import pandas as pd
from pypdf import PdfReader


class DocumentLoader:
    def __init__(self) -> None:
        self.supported_extensions: List[str] = ['.txt', '.pdf', '.xlsx', '.xls']
    
    def load_text_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_pdf_file(self, file_path: str) -> str:
        text: str = ""
        reader: PdfReader = PdfReader(file_path)
        for page in reader.pages:
            extracted: Optional[str] = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    
    def load_excel_file(self, file_path: str) -> str:
        df: pd.DataFrame = pd.read_excel(file_path, engine='openpyxl')
        return df.to_string(index=False)
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext: str = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            content: str = self.load_text_file(file_path)
        elif file_ext == '.pdf':
            content: str = self.load_pdf_file(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            content: str = self.load_excel_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return {
            'file_path': file_path,
            'file_type': file_ext,
            'content': content,
            'metadata': {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
        }
    
    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        directory: Path = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc: Dict[str, Any] = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for file_path in file_paths:
            try:
                doc: Dict[str, Any] = self.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return documents
