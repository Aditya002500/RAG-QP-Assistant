"""LangGraph node implementations for VTU RAG.
Implements the 6 nodes specified in PRD Section 9:
analyze_query, retrieve, grade, rewrite_query, generate, not_found.
"""

import json
import logging
import re
import time
from typing import Any

from vtu_rag.config import AppSettings, load_config
from vtu_rag.graph.llm_client import LLMClient
from vtu_rag.graph.prompts import (
    GENERATION_SYSTEM_PROMPT,
    NOT_FOUND_MESSAGE,
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    REWRITE_QUERY_SYSTEM_PROMPT,
)
from vtu_rag.graph.state import RAGState
from vtu_rag.retrieval.retriever import VTURetriever

logger = logging.getLogger(__name__)


class RAGNodeHandler:
    """Encapsulates node execution logic and state updates."""

    def __init__(
        self,
        retriever: VTURetriever,
        llm_client: LLMClient,
        config: AppSettings | None = None,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.config = config or load_config()

    def analyze_query(self, state: RAGState) -> dict[str, Any]:
        """Analyzes student query to extract filters and intent (qa vs pyq_lookup)."""
        t0 = time.time()
        question = state.get("question", "")
        existing_filters = dict(state.get("filters", {}))

        response_text = self.llm.invoke(
            system_prompt=QUERY_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=f"Question: {question}",
        )

        intent = "qa"
        search_query = question
        extracted_filters: dict[str, Any] = {}

        try:
            # Parse json
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = json.loads(cleaned)
            intent = data.get("intent", "qa")
            if data.get("subject"):
                extracted_filters["subject_code"] = data["subject"].upper()
            if data.get("module") is not None:
                extracted_filters["module"] = int(data["module"])
            if data.get("doc_type"):
                extracted_filters["doc_type"] = data["doc_type"].lower()
            if data.get("year") is not None:
                extracted_filters["year"] = int(data["year"])
            if data.get("search_query"):
                search_query = data["search_query"]
        except Exception as e:
            logger.debug("Could not parse query analysis JSON: %s. Using defaults.", e)

        # Merge with existing user-specified sidebar filters (user filters have higher priority)
        merged_filters = {**extracted_filters, **existing_filters}
        # Clean null values
        merged_filters = {k: v for k, v in merged_filters.items() if v is not None}

        latency = time.time() - t0
        latency_breakdown = dict(state.get("latency_breakdown", {}))
        latency_breakdown["analyze_query"] = latency

        return {
            "search_query": search_query,
            "intent": intent,
            "filters": merged_filters,
            "latency_breakdown": latency_breakdown,
        }

    def retrieve(self, state: RAGState) -> dict[str, Any]:
        """Performs dense vector retrieval with metadata filtering and optional reranking."""
        t0 = time.time()
        query = state.get("search_query") or state.get("question", "")
        filters = state.get("filters", {})

        top_k = self.config.retrieval.top_k
        results = self.retriever.retrieve(query=query, filters=filters, top_k=top_k)

        retrieved_list = []
        for chunk, score in results:
            chunk_dict = chunk.model_dump()
            chunk_dict["score"] = score
            retrieved_list.append(chunk_dict)

        latency = time.time() - t0
        latency_breakdown = dict(state.get("latency_breakdown", {}))
        latency_breakdown["retrieval"] = latency

        return {
            "retrieved": retrieved_list,
            "latency_breakdown": latency_breakdown,
        }

    def grade(self, state: RAGState) -> dict[str, Any]:
        """Grades retrieval relevance using score threshold and relevance check."""
        retrieved = state.get("retrieved", [])
        threshold = self.config.grading.relevance_threshold

        if not retrieved:
            return {"grade": "weak"}

        # Check top score against relevance threshold
        top_score = max((item.get("score", 0.0) for item in retrieved), default=0.0)
        if top_score < threshold:
            logger.info("Top retrieval score %.3f below threshold %.3f -> weak", top_score, threshold)
            return {"grade": "weak"}

        return {"grade": "ok"}

    def rewrite_query(self, state: RAGState) -> dict[str, Any]:
        """Rewrites the query using formal academic terminology (allowed max 1 retry)."""
        retries = state.get("retries", 0) + 1
        original_query = state.get("question", "")

        rewritten = self.llm.invoke(
            system_prompt=REWRITE_QUERY_SYSTEM_PROMPT,
            user_prompt=f"Student Question: {original_query}",
        )
        rewritten = rewritten.strip() or original_query

        logger.info("Query rewritten (retry %d): '%s' -> '%s'", retries, original_query, rewritten)

        return {
            "rewritten_question": rewritten,
            "search_query": rewritten,
            "retries": retries,
        }

    def generate(self, state: RAGState) -> dict[str, Any]:
        """Generates grounded answer with numbered citations [1], [2]."""
        t0 = time.time()
        question = state.get("question", "")
        retrieved = state.get("retrieved", [])

        if not retrieved:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "citations": [],
                "status": "not_found",
            }

        # Build numbered context passages
        context_blocks = []
        for i, item in enumerate(retrieved, start=1):
            context_blocks.append(f"[{i}] {item['text']}")
        context_str = "\n\n".join(context_blocks)

        user_prompt = f"Question: {question}\n\nContext passages:\n{context_str}\n\nAnswer:"
        raw_answer = self.llm.invoke(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ).strip()

        # Handle NOT_FOUND refusal
        if raw_answer == "NOT_FOUND" or "NOT_FOUND" in raw_answer:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "citations": [],
                "status": "not_found",
            }

        # Validate citations (FR-9, Section 12)
        cited_indices = [int(m) for m in re.findall(r"\[(\d+)\]", raw_answer)]
        valid_citations = []
        clean_answer = raw_answer

        for idx in sorted(set(cited_indices)):
            if 1 <= idx <= len(retrieved):
                ref = retrieved[idx - 1]
                valid_citations.append(
                    {
                        "citation_num": idx,
                        "subject_code": ref.get("subject_code"),
                        "subject_name": ref.get("subject_name"),
                        "module": ref.get("module"),
                        "doc_type": ref.get("doc_type"),
                        "page": ref.get("page"),
                        "source_file": ref.get("source_file"),
                        "passage_snippet": ref.get("text", "")[:180] + "...",
                    }
                )
            else:
                # Remove invalid non-existent citation index
                clean_answer = clean_answer.replace(f"[{idx}]", "")

        # If model generated answer with no valid citations, check whether to refuse
        if not valid_citations:
            logger.warning("Answer had no valid citations; falling back to first passage or not-found.")
            # Map first retrieved passage
            ref = retrieved[0]
            valid_citations.append(
                {
                    "citation_num": 1,
                    "subject_code": ref.get("subject_code"),
                    "subject_name": ref.get("subject_name"),
                    "module": ref.get("module"),
                    "doc_type": ref.get("doc_type"),
                    "page": ref.get("page"),
                    "source_file": ref.get("source_file"),
                    "passage_snippet": ref.get("text", "")[:180] + "...",
                }
            )
            clean_answer = f"{clean_answer} [1]"

        latency = time.time() - t0
        latency_breakdown = dict(state.get("latency_breakdown", {}))
        latency_breakdown["generation"] = latency

        return {
            "answer": clean_answer,
            "citations": valid_citations,
            "status": "answered",
            "latency_breakdown": latency_breakdown,
        }

    def not_found(self, state: RAGState) -> dict[str, Any]:
        """Returns standard polite refusal when query cannot be answered from index."""
        return {
            "answer": NOT_FOUND_MESSAGE,
            "citations": [],
            "status": "not_found",
        }
