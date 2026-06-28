import pytest
from app.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank, compute_metrics


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b"], ["c", "d"], k=2) == 0.0

    def test_partial(self):
        score = precision_at_k(["a", "b"], ["a", "c", "d"], k=3)
        assert abs(score - 1/3) < 1e-6

    def test_k_larger_than_retrieved(self):
        score = precision_at_k(["a"], ["a"], k=5)
        assert abs(score - 0.2) < 1e-6

    def test_empty_retrieved(self):
        assert precision_at_k(["a"], [], k=5) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["a", "b"], ["c", "d"], k=2) == 0.0

    def test_partial_recall(self):
        score = recall_at_k(["a", "b", "c"], ["a", "d", "e"], k=3)
        assert abs(score - 1/3) < 1e-6

    def test_empty_relevant(self):
        assert recall_at_k([], ["a", "b"], k=5) == 0.0


class TestReciprocalRank:
    def test_first_is_relevant(self):
        assert reciprocal_rank(["a"], ["a", "b", "c"]) == 1.0

    def test_second_is_relevant(self):
        assert abs(reciprocal_rank(["b"], ["a", "b", "c"]) - 0.5) < 1e-6

    def test_none_relevant(self):
        assert reciprocal_rank(["z"], ["a", "b", "c"]) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank(["a"], []) == 0.0


class TestComputeMetrics:
    def test_perfect_retrieval(self):
        results = [
            {"relevant_ids": ["a", "b"], "retrieved_ids": ["a", "b", "c"]},
            {"relevant_ids": ["x"], "retrieved_ids": ["x", "y"]},
        ]
        m = compute_metrics(results, k=3)
        assert m["precision_at_k"] > 0
        assert m["recall_at_k"] == 1.0
        assert m["mrr"] == 1.0

    def test_empty_results(self):
        m = compute_metrics([], k=5)
        assert m["precision_at_k"] == 0.0
        assert m["recall_at_k"] == 0.0
        assert m["mrr"] == 0.0

    def test_mixed_results(self):
        results = [
            {"relevant_ids": ["a"], "retrieved_ids": ["b", "a"]},  # RR=0.5
            {"relevant_ids": ["x"], "retrieved_ids": ["x"]},        # RR=1.0
        ]
        m = compute_metrics(results, k=2)
        assert abs(m["mrr"] - 0.75) < 1e-4
