import hashlib
import json
import numpy as np
from typing import List, Tuple
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import VectorStoreManager

# Simple in-memory query result cache (query_hash → (chunks, scores))
_query_cache: dict[str, tuple] = {}
MAX_CACHE_SIZE = 1024


def _cache_key(tenant_id: str, query: str, top_k: int) -> str:
    raw = f"{tenant_id}|{query}|{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()


def retrieve(
    vs_manager: VectorStoreManager,
    tenant_id: str,
    query: str,
    top_k: int = 5,
) -> Tuple[List[dict], np.ndarray, bool]:
    """
    Returns (chunks, scores, cache_hit).
    Results are scoped strictly to the tenant's own index.
    """
    key = _cache_key(tenant_id, query, top_k)
    if key in _query_cache:
        chunks, scores = _query_cache[key]
        return chunks, scores, True

    query_vec = embed_query(query)
    chunks, scores = vs_manager.search(tenant_id, query_vec, top_k)

    if len(_query_cache) >= MAX_CACHE_SIZE:
        oldest = next(iter(_query_cache))
        _query_cache.pop(oldest, None)
        _cache_owner.pop(oldest, None)
    _query_cache[key] = (chunks, scores)
    _cache_owner[key] = tenant_id

    return chunks, scores, False


# Separate mapping: cache_key → tenant_id, used for targeted eviction
_cache_owner: dict[str, str] = {}


def invalidate_cache(tenant_id: str) -> None:
    """Evict all cached queries for a tenant (call after document deletion)."""
    to_delete = [k for k, tid in _cache_owner.items() if tid == tenant_id]
    for k in to_delete:
        _query_cache.pop(k, None)
        _cache_owner.pop(k, None)
