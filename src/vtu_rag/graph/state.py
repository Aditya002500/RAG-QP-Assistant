"""LangGraph state schema for VTU RAG workflow."""

from typing import Any

from typing_extensions import TypedDict


class RAGState(TypedDict):
    question: str
    search_query: str
    rewritten_question: str | None
    intent: str  # 'qa' | 'pyq_lookup'
    filters: dict[str, Any]  # subject_code, module, doc_type, year
    retrieved: list[dict[str, Any]]  # chunk data + scores
    grade: str  # 'ok' | 'weak'
    retries: int  # count of rewrites performed
    answer: str
    citations: list[dict[str, Any]]
    status: str  # 'answered' | 'not_found'
    latency_breakdown: dict[str, float]
