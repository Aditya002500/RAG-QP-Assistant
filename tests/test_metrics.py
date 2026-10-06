"""Unit tests for Hit@k, MRR, Refusal Accuracy, and aggregate metric calculations."""

from vtu_rag.eval.metrics import (
    compute_aggregate_metrics,
    compute_hit_at_k,
    compute_reciprocal_rank,
    compute_refusal_accuracy,
    match_gold_source,
)


def test_match_gold_source():
    chunk = {"subject_code": "21CS51", "module": 3, "doc_type": "syllabus"}
    gold = {"subject_code": "21CS51", "module": 3, "doc_type": "syllabus"}
    assert match_gold_source(chunk, gold) is True

    diff_chunk = {"subject_code": "21CS52", "module": 3, "doc_type": "syllabus"}
    assert match_gold_source(diff_chunk, gold) is False


def test_compute_hit_at_k():
    gold = [{"subject_code": "21CS51", "module": 2, "doc_type": "syllabus"}]
    retrieved = [
        {"subject_code": "21CS51", "module": 1, "doc_type": "syllabus"},
        {"subject_code": "21CS51", "module": 2, "doc_type": "syllabus"},
        {"subject_code": "21CS52", "module": 1, "doc_type": "syllabus"},
    ]

    assert compute_hit_at_k(retrieved, gold, k=1) == 0.0
    assert compute_hit_at_k(retrieved, gold, k=2) == 1.0
    assert compute_hit_at_k(retrieved, gold, k=3) == 1.0


def test_compute_reciprocal_rank():
    gold = [{"subject_code": "21CS52", "module": 3, "doc_type": "syllabus"}]
    # Rank 2 hit -> MRR = 0.5
    retrieved = [
        {"subject_code": "21CS51", "module": 1, "doc_type": "syllabus"},
        {"subject_code": "21CS52", "module": 3, "doc_type": "syllabus"},
    ]
    assert compute_reciprocal_rank(retrieved, gold) == 0.5

    # Not found -> 0.0
    assert compute_reciprocal_rank([{"subject_code": "21CS53"}], gold) == 0.0


def test_compute_refusal_accuracy():
    # Unanswerable question correctly refused
    assert compute_refusal_accuracy(is_answerable=False, status="not_found") == 1.0
    # Unanswerable question incorrectly answered
    assert compute_refusal_accuracy(is_answerable=False, status="answered") == 0.0

    # Answerable question correctly answered
    assert compute_refusal_accuracy(is_answerable=True, status="answered") == 1.0
    # Answerable question falsely refused
    assert compute_refusal_accuracy(is_answerable=True, status="not_found") == 0.0


def test_compute_aggregate_metrics():
    mock_results = [
        {
            "answerable": True,
            "gold": [{"subject_code": "21CS51", "module": 1, "doc_type": "syllabus"}],
            "retrieved": [{"subject_code": "21CS51", "module": 1, "doc_type": "syllabus"}],
            "status": "answered",
            "latency": 0.1,
        },
        {
            "answerable": False,
            "gold": [],
            "retrieved": [],
            "status": "not_found",
            "latency": 0.05,
        },
    ]
    agg = compute_aggregate_metrics(mock_results)
    assert agg["hit_at_1"] == 1.0
    assert agg["hit_at_5"] == 1.0
    assert agg["mrr"] == 1.0
    assert agg["refusal_accuracy"] == 1.0
    assert agg["total_evaluated"] == 2
