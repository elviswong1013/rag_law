from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml
import os


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.8
    api_key: Optional[str] = None


@dataclass
class EmbeddingConfig:
    model: str
    batch_size: int = 25
    api_key: Optional[str] = None


@dataclass
class RerankerConfig:
    method: str = "hybrid"
    top_k: int = 5
    use_bm25: bool = True
    use_cross_encoder: bool = False
    cohere_api_key: Optional[str] = None


@dataclass
class VectorDBConfig:
    persist_directory: str = "./chroma_db"
    collection_name: str = "law_documents"
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    method: str = "paragraph"


@dataclass
class RetrievalConfig:
    top_k: int = 10
    score_threshold: float = 0.7
    max_searches: int = 2


@dataclass
class EvaluationConfig:
    use_llm_judge: bool = True
    judge_model: str = "gpt-4o-mini"
    evaluation_metrics: list = None
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "relevance",
                "accuracy",
                "completeness",
                "clarity"
            ]


@dataclass
class PipelineConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    vector_db: VectorDBConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            llm=LLMConfig(**config_dict.get('llm', {})),
            embedding=EmbeddingConfig(**config_dict.get('embedding', {})),
            reranker=RerankerConfig(**config_dict.get('reranker', {})),
            vector_db=VectorDBConfig(**config_dict.get('vector_db', {})),
            chunking=ChunkingConfig(**config_dict.get('chunking', {})),
            retrieval=RetrievalConfig(**config_dict.get('retrieval', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {}))
        )
    
    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        return cls(
            llm=LLMConfig(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
                top_p=float(os.getenv("LLM_TOP_P", "0.8")),
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            embedding=EmbeddingConfig(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "25")),
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            reranker=RerankerConfig(
                method=os.getenv("RERANKER_METHOD", "hybrid"),
                top_k=int(os.getenv("RERANK_TOP_K", "5")),
                use_bm25=os.getenv("RERANKER_USE_BM25", "true").lower() == "true",
                use_cross_encoder=os.getenv("RERANKER_USE_CROSS_ENCODER", "false").lower() == "true",
                cohere_api_key=os.getenv("COHERE_API_KEY")
            ),
            vector_db=VectorDBConfig(
                persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                collection_name=os.getenv("CHROMA_COLLECTION_NAME", "law_documents"),
                distance_metric=os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
            ),
            chunking=ChunkingConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
                method=os.getenv("CHUNKING_METHOD", "paragraph")
            ),
            retrieval=RetrievalConfig(
                top_k=int(os.getenv("TOP_K_RETRIEVAL", "10")),
                score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.7")),
                max_searches=int(os.getenv("MAX_SEARCHES", "2"))
            ),
            evaluation=EvaluationConfig(
                use_llm_judge=os.getenv("USE_LLM_JUDGE", "true").lower() == "true",
                judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini")
            )
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'llm': self.llm.__dict__,
            'embedding': self.embedding.__dict__,
            'reranker': self.reranker.__dict__,
            'vector_db': self.vector_db.__dict__,
            'chunking': self.chunking.__dict__,
            'retrieval': self.retrieval.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def save_yaml(self, config_path: str) -> None:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
