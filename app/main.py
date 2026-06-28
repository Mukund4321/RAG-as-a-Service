import time
import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import (
    DocumentUploadResponse, DocumentListResponse, DocumentListItem,
    QueryRequest, QueryResponse, EvalResult, HealthResponse,
    APIKeyCreateRequest, APIKeyCreateResponse, new_id,
)
from app.auth.api_keys import validate_api_key, create_api_key
from app.auth.rate_limiter import check_rate_limit
from app.ingestion.document_loader import load_document
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.retriever import retrieve, invalidate_cache
from app.generation.generator import generate_answer
from app.eval.eval_set import get_eval_set
from app.eval.metrics import compute_metrics
from app.observability.logger import get_logger, log_request
from app.database import db_cursor
from datetime import datetime

settings = get_settings()
logger = get_logger()

app = FastAPI(
    title="RAG-as-a-Service API",
    description="Multi-tenant, production-grade Retrieval-Augmented Generation API with evaluation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vs_manager = VectorStoreManager(settings.indices_dir)

os.makedirs(settings.uploads_dir, exist_ok=True)
os.makedirs(settings.indices_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def get_tenant(x_api_key: str = Header(..., alias="X-API-Key")):
    tenant = validate_api_key(x_api_key)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    check_rate_limit(x_api_key)
    return tenant


# ── Admin: create API key ───────────────────────────────────────────────────────

@app.post("/v1/admin/keys", response_model=APIKeyCreateResponse, tags=["Admin"])
def create_tenant_key(
    body: APIKeyCreateRequest,
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
):
    if x_admin_key != settings.admin_api_key or not settings.admin_api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin key")
    result = create_api_key(body.tenant_name)
    return result


# ── Documents ───────────────────────────────────────────────────────────────────

@app.post("/v1/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    tenant: dict = Depends(get_tenant),
):
    content = await file.read()
    raw_text = load_document(file.filename, content)
    chunks = chunk_text(raw_text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    embeddings = embed_chunks(chunks)

    doc_id = new_id()
    vs_manager.add_chunks(tenant["tenant_id"], doc_id, file.filename, chunks, embeddings)

    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO documents (document_id, tenant_id, filename, chunk_count) VALUES (?,?,?,?)",
            (doc_id, tenant["tenant_id"], file.filename, len(chunks)),
        )

    logger.info("document_ingested", extra={
        "tenant_id": tenant["tenant_id"],
        "document_id": doc_id,
        "filename": file.filename,
        "chunk_count": len(chunks),
    })

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        chunk_count=len(chunks),
    )


@app.get("/v1/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents(tenant: dict = Depends(get_tenant)):
    with db_cursor() as cur:
        cur.execute(
            "SELECT document_id, filename, chunk_count, ingested_at FROM documents WHERE tenant_id=?",
            (tenant["tenant_id"],),
        )
        rows = cur.fetchall()
    items = [
        DocumentListItem(
            document_id=r["document_id"],
            filename=r["filename"],
            chunk_count=r["chunk_count"],
            ingested_at=datetime.fromisoformat(r["ingested_at"]),
        )
        for r in rows
    ]
    return DocumentListResponse(documents=items, total=len(items))


@app.delete("/v1/documents/{document_id}", tags=["Documents"])
def delete_document(document_id: str, tenant: dict = Depends(get_tenant)):
    with db_cursor() as cur:
        cur.execute(
            "SELECT document_id FROM documents WHERE document_id=? AND tenant_id=?",
            (document_id, tenant["tenant_id"]),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        cur.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

    vs_manager.remove_document(tenant["tenant_id"], document_id)
    invalidate_cache(tenant["tenant_id"])
    return {"deleted": document_id}


# ── Query ───────────────────────────────────────────────────────────────────────

@app.post("/v1/query", response_model=QueryResponse, tags=["Query"])
def query(body: QueryRequest, tenant: dict = Depends(get_tenant)):
    t0 = time.monotonic()

    context_chunks, scores, cached = retrieve(
        vs_manager=vs_manager,
        tenant_id=tenant["tenant_id"],
        query=body.query,
        top_k=body.top_k,
    )

    answer, tokens_used = generate_answer(body.query, context_chunks)

    latency_ms = (time.monotonic() - t0) * 1000

    log_request(logger, tenant["tenant_id"], body.query, latency_ms, tokens_used, cached)

    citations = [
        {
            "document_id": c["document_id"],
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "chunk_text": c["text"],
            "score": float(scores[i]),
        }
        for i, c in enumerate(context_chunks)
    ] if body.include_sources else []

    return QueryResponse(
        query=body.query,
        answer=answer,
        citations=citations,
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens_used,
        cached=cached,
    )


# ── Evaluation ──────────────────────────────────────────────────────────────────

@app.get("/v1/eval/run", response_model=EvalResult, tags=["Evaluation"])
def run_eval(k: int = 5, tenant: dict = Depends(get_tenant)):
    eval_set = get_eval_set(tenant["tenant_id"])
    if not eval_set:
        raise HTTPException(status_code=404, detail="No eval set found for this tenant")

    results = []
    for item in eval_set:
        retrieved_chunks, _, _ = retrieve(
            vs_manager=vs_manager,
            tenant_id=tenant["tenant_id"],
            query=item["query"],
            top_k=k,
        )
        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
        results.append({
            "relevant_ids": item["relevant_chunk_ids"],
            "retrieved_ids": retrieved_ids,
        })

    metrics = compute_metrics(results, k=k)
    return EvalResult(**metrics, num_queries=len(eval_set), k=k)


# ── Health ──────────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse()
