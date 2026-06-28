import hashlib
import numpy as np
from typing import List
from functools import lru_cache
from app.config import get_settings

settings = get_settings()

# Module-level in-memory embedding cache (chunk_hash → embedding vector)
_embed_cache: dict[str, np.ndarray] = {}


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_chunks(chunks: List[str]) -> np.ndarray:
    model = _get_model()
    to_embed: List[int] = []
    result = np.zeros((len(chunks), settings.embedding_dim), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        h = _chunk_hash(chunk)
        if h in _embed_cache:
            result[i] = _embed_cache[h]
        else:
            to_embed.append(i)

    if to_embed:
        texts = [chunks[i] for i in to_embed]
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        for j, i in enumerate(to_embed):
            h = _chunk_hash(chunks[i])
            _embed_cache[h] = vecs[j]
            result[i] = vecs[j]

    return result


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0]


def cache_stats() -> dict:
    return {"cached_embeddings": len(_embed_cache)}
