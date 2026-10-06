"""LangGraph workflow builder for VTU RAG.
Constructs the 6-node state graph matching PRD Section 9.
"""

import json
import logging
import time
from pathlib import Path
from typing import Literal

from langgraph.graph import END, StateGraph

from vtu_rag.config import AppSettings, load_config
from vtu_rag.graph.llm_client import LLMClient
from vtu_rag.graph.nodes import RAGNodeHandler
from vtu_rag.graph.state import RAGState
from vtu_rag.retrieval.retriever import VTURetriever

logger = logging.getLogger(__name__)


def build_rag_graph(
    retriever: VTURetriever | None = None,
    config: AppSettings | None = None,
):
    """Builds and compiles the StateGraph for VTU RAG workflow."""
    if config is None:
        config = load_config()

    if retriever is None:
        retriever = VTURetriever.load_from_config(config)

    llm_client = LLMClient(config.llm)
    handler = RAGNodeHandler(retriever=retriever, llm_client=llm_client, config=config)

    workflow = StateGraph(RAGState)

    # Add the 6 nodes
    workflow.add_node("analyze_query", handler.analyze_query)
    workflow.add_node("retrieve", handler.retrieve)
    workflow.add_node("grade", handler.grade)
    workflow.add_node("rewrite_query", handler.rewrite_query)
    workflow.add_node("generate", handler.generate)
    workflow.add_node("not_found", handler.not_found)

    # Set Entry Point
    workflow.set_entry_point("analyze_query")

    # analyze_query -> retrieve -> grade
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # Conditional routing after grading
    def route_grade(state: RAGState) -> Literal["generate", "rewrite_query", "not_found"]:
        grade = state.get("grade", "ok")
        retries = state.get("retries", 0)
        max_retries = config.grading.max_retries

        if grade == "ok":
            return "generate"
        if retries < max_retries:
            return "rewrite_query"
        return "not_found"

    workflow.add_conditional_edges(
        "grade",
        route_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "not_found": "not_found",
        },
    )

    # rewrite_query loops back to retrieve
    workflow.add_edge("rewrite_query", "retrieve")

    # Final terminations
    workflow.add_edge("generate", END)
    workflow.add_edge("not_found", END)

    app = workflow.compile()
    logger.info("LangGraph VTU RAG workflow successfully compiled.")
    return app


def run_pipeline(
    question: str,
    filters: dict | None = None,
    graph_app=None,
    config: AppSettings | None = None,
) -> RAGState:
    """Helper function to execute the full LangGraph pipeline on a question."""
    if graph_app is None:
        graph_app = build_rag_graph(config=config)

    initial_state: RAGState = {
        "question": question,
        "search_query": question,
        "rewritten_question": None,
        "intent": "qa",
        "filters": filters or {},
        "retrieved": [],
        "grade": "ok",
        "retries": 0,
        "answer": "",
        "citations": [],
        "status": "answered",
        "latency_breakdown": {},
    }

    final_state = graph_app.invoke(initial_state)

    # FR-14 Query log (JSONL): filters, retrieved chunk IDs, scores, latency
    try:
        log_path = Path("data/query_logs.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp": time.time(),
            "question": question,
            "filters": final_state.get("filters", {}),
            "retrieved_chunk_ids": [c.get("chunk_id") for c in final_state.get("retrieved", [])],
            "scores": [c.get("score") for c in final_state.get("retrieved", [])],
            "status": final_state.get("status"),
            "latency_breakdown": final_state.get("latency_breakdown", {}),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.debug("Failed to write query log: %s", e)

    return final_state
