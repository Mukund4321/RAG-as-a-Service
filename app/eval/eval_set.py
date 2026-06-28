"""
Eval set management.

Format (stored per-tenant in indices/<tenant_id>_eval.json):
[
  {
    "query": "What is the capital of France?",
    "relevant_chunk_ids": ["doc-uuid:0", "doc-uuid:3"]
  },
  ...
]

The eval set is manually curated. Use the CLI helper below to add entries.
"""

import json
import os
from typing import List, Optional
from app.config import get_settings

settings = get_settings()


def _eval_path(tenant_id: str) -> str:
    return os.path.join(settings.indices_dir, f"{tenant_id}_eval.json")


def get_eval_set(tenant_id: str) -> Optional[List[dict]]:
    path = _eval_path(tenant_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_eval_set(tenant_id: str, eval_items: List[dict]) -> None:
    os.makedirs(settings.indices_dir, exist_ok=True)
    path = _eval_path(tenant_id)
    with open(path, "w") as f:
        json.dump(eval_items, f, indent=2)


def add_eval_item(tenant_id: str, query: str, relevant_chunk_ids: List[str]) -> None:
    current = get_eval_set(tenant_id) or []
    current.append({"query": query, "relevant_chunk_ids": relevant_chunk_ids})
    save_eval_set(tenant_id, current)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG eval set CLI")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--chunk-ids", nargs="+", required=True)
    args = parser.parse_args()
    add_eval_item(args.tenant_id, args.query, args.chunk_ids)
    print(f"Added eval item for tenant {args.tenant_id}")
