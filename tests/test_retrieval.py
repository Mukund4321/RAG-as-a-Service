import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.retrieval.retriever import retrieve, _cache_key, _query_cache


class TestCacheKey:
    def test_different_tenants_produce_different_keys(self):
        k1 = _cache_key("tenant-1", "What is AI?", 5)
        k2 = _cache_key("tenant-2", "What is AI?", 5)
        assert k1 != k2

    def test_same_inputs_produce_same_key(self):
        k1 = _cache_key("tenant-1", "query", 5)
        k2 = _cache_key("tenant-1", "query", 5)
        assert k1 == k2

    def test_different_top_k_produces_different_key(self):
        k1 = _cache_key("tenant-1", "query", 3)
        k2 = _cache_key("tenant-1", "query", 7)
        assert k1 != k2


class TestRetriever:
    def setup_method(self):
        _query_cache.clear()

    def test_returns_cache_hit_on_second_call(self):
        mock_vs = MagicMock()
        mock_vs.search.return_value = (
            [{"chunk_id": "doc:0", "document_id": "doc", "filename": "test.txt",
              "chunk_index": 0, "text": "test chunk"}],
            np.array([0.95]),
        )

        with patch("app.retrieval.retriever.embed_query", return_value=np.zeros(384)):
            chunks1, scores1, cached1 = retrieve(mock_vs, "tenant-1", "test query", 5)
            chunks2, scores2, cached2 = retrieve(mock_vs, "tenant-1", "test query", 5)

        assert cached1 is False
        assert cached2 is True
        assert mock_vs.search.call_count == 1  # only called once

    def test_tenant_isolation(self):
        mock_vs = MagicMock()
        mock_vs.search.return_value = ([], np.array([]))

        with patch("app.retrieval.retriever.embed_query", return_value=np.zeros(384)):
            retrieve(mock_vs, "tenant-A", "query", 5)
            retrieve(mock_vs, "tenant-B", "query", 5)

        assert mock_vs.search.call_count == 2
        calls = [c.args[0] for c in mock_vs.search.call_args_list]
        assert "tenant-A" in calls
        assert "tenant-B" in calls

    def test_empty_index_returns_empty(self):
        mock_vs = MagicMock()
        mock_vs.search.return_value = ([], np.array([]))

        with patch("app.retrieval.retriever.embed_query", return_value=np.zeros(384)):
            chunks, scores, cached = retrieve(mock_vs, "tenant-1", "query", 5)

        assert chunks == []
        assert len(scores) == 0
