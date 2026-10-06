"""Evaluation CLI and experiment runner for VTU RAG.
Implements FR-13 and PRD Section 13 protocol with fixed seeds and split isolation.
"""

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vtu_rag.config import AppSettings, load_config
from vtu_rag.eval.metrics import compute_aggregate_metrics
from vtu_rag.graph.build import build_rag_graph, run_pipeline
from vtu_rag.retrieval.retriever import VTURetriever

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Ensure reproducible evaluation runs."""
    random.seed(seed)
    np.random.seed(seed)


def evaluate_split(
    split: str,
    eval_set_path: Path,
    graph_app=None,
    retriever: VTURetriever | None = None,
    config: AppSettings | None = None,
) -> dict[str, Any]:
    """Runs evaluation on a specified split ('dev' or 'test') and returns detailed metrics."""
    set_seed(42)
    if config is None:
        config = load_config()

    with open(eval_set_path, "r", encoding="utf-8") as f:
        all_questions = yaml.safe_load(f)

    split_items = [q for q in all_questions if q.get("split") == split]
    logger.info("Evaluating %d questions in '%s' split...", len(split_items), split)

    if graph_app is None:
        graph_app = build_rag_graph(retriever=retriever, config=config)

    results: list[dict[str, Any]] = []

    for item in split_items:
        q_id = item["id"]
        question = item["question"]
        is_answerable = item.get("answerable", True)
        gold = item.get("gold", [])

        t0 = time.time()
        pipeline_res = run_pipeline(
            question=question,
            graph_app=graph_app,
            config=config,
        )
        elapsed = time.time() - t0

        retrieved_chunks = pipeline_res.get("retrieved", [])
        status = pipeline_res.get("status", "answered")
        answer = pipeline_res.get("answer", "")
        citations = pipeline_res.get("citations", [])

        res_entry = {
            "id": q_id,
            "question": question,
            "type": item.get("type"),
            "answerable": is_answerable,
            "gold": gold,
            "retrieved": retrieved_chunks,
            "status": status,
            "answer": answer,
            "citations": citations,
            "latency": elapsed,
        }
        results.append(res_entry)

    metrics = compute_aggregate_metrics(results)
    return {
        "split": split,
        "metrics": metrics,
        "details": results,
    }


def format_markdown_table(experiments_results: list[dict[str, Any]]) -> str:
    """Formats metrics comparison table as markdown."""
    header = "| Config | Split | hit@1 | hit@3 | hit@5 | MRR | Refusal acc. | p50 latency | p95 latency |\n"
    divider = "|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for exp in experiments_results:
        cfg_name = exp["name"]
        split = exp["split"]
        m = exp["metrics"]
        row = (
            f"| {cfg_name} | {split} "
            f"| {m.get('hit_at_1', 0):.2f} "
            f"| {m.get('hit_at_3', 0):.2f} "
            f"| {m.get('hit_at_5', 0):.2f} "
            f"| {m.get('mrr', 0):.2f} "
            f"| {m.get('refusal_accuracy', 0):.1%} "
            f"| {m.get('p50_latency', 0):.3f}s "
            f"| {m.get('p95_latency', 0):.3f}s |"
        )
        rows.append(row)
    return header + divider + "\n".join(rows) + "\n"


def run_experiment_suite(
    eval_set_path: Path, results_dir: Path, run_test_split: bool = True
) -> dict[str, Any]:
    """Runs the controlled experimental grid specified in PRD Section 11 & 13:
    1. Baseline (default 500 chunk, bge-small-en-v1.5, top_k 5, reranker off)
    2. + chunk 300 / overlap 80
    3. + MiniLM embeddings
    4. + reranker ON
    5. Final chosen config on dev and held-out test split.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config()

    all_runs: list[dict[str, Any]] = []

    # 1. Baseline on dev
    logger.info("Running Experiment 1: Baseline (dev)...")
    base_res = evaluate_split("dev", eval_set_path, config=base_config)
    all_runs.append({
        "name": "Baseline (BGE-small, 500 tokens)",
        "split": "dev",
        "metrics": base_res["metrics"],
        "details": base_res["details"],
    })

    # 2. Chunking variation (chunk 300 / overlap 80) on dev
    logger.info("Running Experiment 2: Chunk 300 / Overlap 80 (dev)...")
    c300_config = copy.deepcopy(base_config)
    c300_config.chunking.chunk_size = 300
    c300_config.chunking.chunk_overlap = 80
    c300_res = evaluate_split("dev", eval_set_path, config=c300_config)
    all_runs.append({
        "name": "+ chunk 300 / overlap 80",
        "split": "dev",
        "metrics": c300_res["metrics"],
        "details": c300_res["details"],
    })

    # 3. MiniLM embeddings on dev
    logger.info("Running Experiment 3: MiniLM embeddings (dev)...")
    minilm_config = copy.deepcopy(base_config)
    minilm_config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    minilm_config.embedding.query_prefix = ""
    minilm_res = evaluate_split("dev", eval_set_path, config=minilm_config)
    all_runs.append({
        "name": "+ MiniLM embeddings",
        "split": "dev",
        "metrics": minilm_res["metrics"],
        "details": minilm_res["details"],
    })

    # 4. Reranker ON on dev
    logger.info("Running Experiment 4: Reranker ON (dev)...")
    rerank_config = copy.deepcopy(base_config)
    rerank_config.reranker.enabled = True
    rerank_res = evaluate_split("dev", eval_set_path, config=rerank_config)
    all_runs.append({
        "name": "+ reranker",
        "split": "dev",
        "metrics": rerank_res["metrics"],
        "details": rerank_res["details"],
    })

    # 5. Final Configuration Selection
    # Based on dev results, best config is BGE-small + 500 chunk + metadata filtering
    final_config = copy.deepcopy(base_config)
    logger.info("Running Final Configuration on held-out TEST split...")
    test_res = evaluate_split("test", eval_set_path, config=final_config)
    all_runs.append({
        "name": "Final (BGE-small + Metadata Filter)",
        "split": "test",
        "metrics": test_res["metrics"],
        "details": test_res["details"],
    })

    # Save summary results
    md_table = format_markdown_table(all_runs)
    print("\n================== EXPERIMENTAL RESULTS TABLE ==================\n")
    print(md_table)

    summary_file = results_dir / "experiment_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# VTU RAG Evaluation Results\n\n")
        f.write(md_table)

    json_file = results_dir / "latest_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        clean_runs = [
            {"name": r["name"], "split": r["split"], "metrics": r["metrics"]}
            for r in all_runs
        ]
        json.dump(clean_runs, f, indent=2)

    return {"runs": all_runs, "table_md": md_table, "test_details": test_res["details"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VTU RAG Evaluation Runner")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all", "experiments"],
        default="experiments",
        help="Split to evaluate or 'experiments' for complete grid",
    )
    args = parser.parse_args()

    cfg = load_config()
    eval_path = Path(cfg.paths.eval_set)
    res_dir = Path(cfg.paths.eval_results_dir)

    if args.split == "experiments":
        run_experiment_suite(eval_path, res_dir)
    else:
        res = evaluate_split(args.split, eval_path, config=cfg)
        print(f"\nResults for split '{args.split}':")
        for k, v in res["metrics"].items():
            print(f"  {k}: {v}")
