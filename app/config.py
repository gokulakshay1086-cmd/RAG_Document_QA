"""
Centralized configuration, loaded from environment variables / .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 4
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.5  # 1.0 = pure semantic, 0.0 = pure keyword (BM25)
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_candidate_k: int = 20  # candidates pulled before re-ranking

    # Storage
    vector_store_dir: str = "./data/vectorstore"
    upload_dir: str = "./data/uploads"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def vector_store_path(self) -> Path:
        p = Path(self.vector_store_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uses_openai(self) -> bool:
        return bool(self.openai_api_key.strip())


settings = Settings()
