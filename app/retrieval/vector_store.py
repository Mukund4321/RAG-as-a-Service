import os
import json
import numpy as np
import faiss
from typing import List, Optional
from threading import Lock


class VectorStoreManager:
    """
    Manages per-tenant FAISS flat-L2 indices backed by disk.
    Each tenant gets an isolated index + metadata file.
    Thread-safe for concurrent requests.
    """

    def __init__(self, indices_dir: str):
        self.indices_dir = indices_dir
        os.makedirs(indices_dir, exist_ok=True)
        self._locks: dict[str, Lock] = {}
        self._indices: dict[str, faiss.IndexFlatIP] = {}
        self._metadata: dict[str, List[dict]] = {}

    def _get_lock(self, tenant_id: str) -> Lock:
        if tenant_id not in self._locks:
            self._locks[tenant_id] = Lock()
        return self._locks[tenant_id]

    def _index_path(self, tenant_id: str) -> str:
        return os.path.join(self.indices_dir, f"{tenant_id}.faiss")

    def _meta_path(self, tenant_id: str) -> str:
        return os.path.join(self.indices_dir, f"{tenant_id}.json")

    def _load(self, tenant_id: str) -> None:
        idx_path = self._index_path(tenant_id)
        meta_path = self._meta_path(tenant_id)
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self._indices[tenant_id] = faiss.read_index(idx_path)
            with open(meta_path, "r") as f:
                self._metadata[tenant_id] = json.load(f)
        else:
            from app.config import get_settings
            dim = get_settings().embedding_dim
            self._indices[tenant_id] = faiss.IndexFlatIP(dim)
            self._metadata[tenant_id] = []

    def _save(self, tenant_id: str) -> None:
        faiss.write_index(self._indices[tenant_id], self._index_path(tenant_id))
        with open(self._meta_path(tenant_id), "w") as f:
            json.dump(self._metadata[tenant_id], f)

    def _ensure_loaded(self, tenant_id: str) -> None:
        if tenant_id not in self._indices:
            self._load(tenant_id)

    def add_chunks(
        self,
        tenant_id: str,
        document_id: str,
        filename: str,
        chunks: List[str],
        embeddings: np.ndarray,
    ) -> None:
        with self._get_lock(tenant_id):
            self._ensure_loaded(tenant_id)
            index = self._indices[tenant_id]
            meta = self._metadata[tenant_id]
            start_idx = index.ntotal
            index.add(embeddings.astype(np.float32))
            for i, chunk in enumerate(chunks):
                meta.append({
                    "chunk_id": f"{document_id}:{i}",
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                })
            self._save(tenant_id)

    def search(
        self,
        tenant_id: str,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> tuple[List[dict], np.ndarray]:
        with self._get_lock(tenant_id):
            self._ensure_loaded(tenant_id)
            index = self._indices[tenant_id]
            meta = self._metadata[tenant_id]

            if index.ntotal == 0:
                return [], np.array([])

            k = min(top_k, index.ntotal)
            scores, indices = index.search(query_vec.reshape(1, -1).astype(np.float32), k)
            results = [meta[i] for i in indices[0] if i >= 0]
            return results, scores[0][:len(results)]

    def remove_document(self, tenant_id: str, document_id: str) -> None:
        """
        Removes all chunks belonging to document_id by rebuilding the index.
        FAISS flat indices don't support in-place deletion.
        """
        with self._get_lock(tenant_id):
            self._ensure_loaded(tenant_id)
            meta = self._metadata[tenant_id]
            keep_indices = [i for i, m in enumerate(meta) if m["document_id"] != document_id]

            if not keep_indices:
                from app.config import get_settings
                dim = get_settings().embedding_dim
                self._indices[tenant_id] = faiss.IndexFlatIP(dim)
                self._metadata[tenant_id] = []
            else:
                old_index = self._indices[tenant_id]
                vectors = np.array([old_index.reconstruct(i) for i in keep_indices])
                from app.config import get_settings
                new_index = faiss.IndexFlatIP(get_settings().embedding_dim)
                new_index.add(vectors)
                self._indices[tenant_id] = new_index
                self._metadata[tenant_id] = [meta[i] for i in keep_indices]

            self._save(tenant_id)

    def tenant_chunk_count(self, tenant_id: str) -> int:
        with self._get_lock(tenant_id):
            self._ensure_loaded(tenant_id)
            return self._indices[tenant_id].ntotal
