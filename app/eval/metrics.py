"""
Retrieval quality metrics used in the IEEE paper evaluation:
  - Precision@k: fraction of retrieved chunks that are relevant
  - Recall@k:    fraction of relevant chunks that were retrieved
  - MRR:         Mean Reciprocal Rank (rank of first relevant result)

Each item in results:
  {"relevant_ids": [...], "retrieved_ids": [...]}
"""

from typing import List


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in set(relevant))
    return hits / k


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in set(relevant))
    return hits / len(relevant)


def reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_metrics(results: List[dict], k: int = 5) -> dict:
    if not results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    p_scores, r_scores, rr_scores = [], [], []

    for item in results:
        rel = item["relevant_ids"]
        ret = item["retrieved_ids"]
        p_scores.append(precision_at_k(rel, ret, k))
        r_scores.append(recall_at_k(rel, ret, k))
        rr_scores.append(reciprocal_rank(rel, ret))

    return {
        "precision_at_k": round(sum(p_scores) / len(p_scores), 4),
        "recall_at_k": round(sum(r_scores) / len(r_scores), 4),
        "mrr": round(sum(rr_scores) / len(rr_scores), 4),
    }
