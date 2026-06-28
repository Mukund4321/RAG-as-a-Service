from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # LLM backend
    llm_backend: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Database
    database_url: str = "sqlite:///./rag_service.db"

    # Redis cache (optional; falls back to in-memory)
    redis_url: str = ""
    cache_ttl_seconds: int = 3600

    # Storage paths
    indices_dir: str = "./indices"
    uploads_dir: str = "./uploads"

    # Auth
    api_key_length: int = 32
    admin_api_key: str = ""

    # Rate limiting (requests per minute per API key)
    rate_limit_rpm: int = 60

    # Retrieval
    default_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Observability
    log_level: str = "INFO"
    log_file: str = "logs/rag_service.jsonl"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
