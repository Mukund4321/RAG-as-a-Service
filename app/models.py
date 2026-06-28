from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


# ── Request / Response schemas ──────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str = "ingested"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    latency_ms: float
    tokens_used: int
    cached: bool = False


class EvalResult(BaseModel):
    precision_at_k: float
    recall_at_k: float
    mrr: float
    num_queries: int
    k: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIKeyCreateRequest(BaseModel):
    tenant_name: str = Field(..., min_length=1, max_length=128)


class APIKeyCreateResponse(BaseModel):
    api_key: str
    tenant_id: str
    tenant_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Internal DB models (SQLAlchemy-style dicts for raw SQL) ─────────────────────

def new_id() -> str:
    return str(uuid.uuid4())
