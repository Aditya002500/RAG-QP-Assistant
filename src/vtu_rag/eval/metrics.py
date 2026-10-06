"""Evaluation metrics computation module for VTU RAG.
Implements Hit@k, MRR, Refusal Accuracy, and Latency percentiles as defined in PRD Section 13.
"""

from typing import Any

import numpy as np


def match_gold_source(chunk: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Checks if a retrieved chunk matches a gold source specification."""
    # Check subject code if specified in gold
    if gold.get("subject_code") and chunk.get("subject_code", "").upper() != gold["subject_code"].upper():
        return False

    # Check module if specified in gold
    if gold.get("module") is not None and chunk.get("module") != gold["module"]:
        return False

    # Check doc_type if specified in gold
    if gold.get("doc_type") and chunk.get("doc_type", "").lower() != gold["doc_type"].lower():
        return False

    # Check year if specified in gold
    return not (gold.get("year") is not None and chunk.get("year") != gold["year"])


def chunk_matches_any_gold(
    chunk: dict[str, Any], gold_list: list[dict[str, Any]]
) -> bool:
    """Checks if a chunk matches any gold item in gold_list."""
    return any(match_gold_source(chunk, g) for g in gold_list)


def compute_hit_at_k(
    retrieved_chunks: list[dict[str, Any]],
    gold_list: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Hit@k: 1.0 if at least one gold item matches in top-k chunks, else 0.0."""
    if not gold_list:
        return 0.0

    top_chunks = retrieved_chunks[:k]
    for chunk in top_chunks:
        if chunk_matches_any_gold(chunk, gold_list):
            return 1.0
    return 0.0


def compute_reciprocal_rank(
    retrieved_chunks: list[dict[str, Any]],
    gold_list: list[dict[str, Any]],
) -> float:
    """MRR (Reciprocal Rank): 1 / rank of first gold hit, or 0.0 if not found."""
    if not gold_list:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk_matches_any_gold(chunk, gold_list):
            return 1.0 / rank
    return 0.0


def compute_refusal_accuracy(
    is_answerable: bool,
    status: str,
) -> float:
    """Refusal accuracy:
    1.0 if (unanswerable and status=='not_found') or (answerable and status=='answered')
    0.0 otherwise.
    """
    if not is_answerable:
        return 1.0 if status == "not_found" else 0.0
    else:
        return 1.0 if status == "answered" else 0.0


def compute_aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Computes aggregate Hit@1, Hit@3, Hit@5, MRR, Refusal Accuracy, and Latencies."""
    if not results:
        return {}

    answerable_items = [r for r in results if r.get("answerable", True)]
    unanswerable_items = [r for r in results if not r.get("answerable", True)]

    hits_1 = [compute_hit_at_k(r["retrieved"], r["gold"], k=1) for r in answerable_items]
    hits_3 = [compute_hit_at_k(r["retrieved"], r["gold"], k=3) for r in answerable_items]
    hits_5 = [compute_hit_at_k(r["retrieved"], r["gold"], k=5) for r in answerable_items]
    mrrs = [compute_reciprocal_rank(r["retrieved"], r["gold"]) for r in answerable_items]

    refusal_accs = [
        compute_refusal_accuracy(r["answerable"], r.get("status", "answered"))
        for r in results
    ]

    latencies = [r.get("latency", 0.0) for r in results if "latency" in r]

    return {
        "hit_at_1": float(np.mean(hits_1)) if hits_1 else 0.0,
        "hit_at_3": float(np.mean(hits_3)) if hits_3 else 0.0,
        "hit_at_5": float(np.mean(hits_5)) if hits_5 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "refusal_accuracy": float(np.mean(refusal_accs)) if refusal_accs else 0.0,
        "p50_latency": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "total_evaluated": len(results),
        "total_answerable": len(answerable_items),
        "total_unanswerable": len(unanswerable_items),
    }
