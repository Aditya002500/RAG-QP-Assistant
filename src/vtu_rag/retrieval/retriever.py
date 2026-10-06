"""Retriever module connecting vector search, metadata filtering, and reranking.
Implements FR-7 and Section 11 retrieval notes.
"""

import logging
from pathlib import Path
from typing import Any

from vtu_rag.config import AppSettings, load_config
from vtu_rag.index.embed import EmbeddingManager
from vtu_rag.index.store import VectorStore
from vtu_rag.ingest.chunk import ChunkMetadata
from vtu_rag.retrieval.rerank import CrossEncoderReranker

logger = logging.getLogger(__name__)


class VTURetriever:
    """End-to-end retriever with metadata filtering and reranker support."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingManager,
        config: AppSettings | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or load_config()
        self.reranker = CrossEncoderReranker(self.config.reranker)

    @classmethod
    def load_from_config(cls, config: AppSettings | None = None) -> "VTURetriever":
        if config is None:
            config = load_config()

        store = VectorStore.load(
            Path(config.paths.faiss_index),
            Path(config.paths.metadata_file),
        )
        embedder = EmbeddingManager(config.embedding)
        return cls(store=store, embedder=embedder, config=config)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """Retrieves top_k chunks for the given query with optional filters and reranking."""
        k = top_k or self.config.retrieval.top_k
        query_vec = self.embedder.embed_query(query)

        active_filters = filters if self.config.retrieval.use_metadata_filter else None
        candidate_k = self.config.retrieval.candidate_k

        candidates = self.store.search(
            query_vector=query_vec,
            top_k=candidate_k if self.reranker.config.enabled else k,
            candidate_k=candidate_k,
            filters=active_filters,
        )

        if self.reranker.config.enabled and candidates:
            return self.reranker.rerank(query, candidates, top_n=k)

        return candidates[:k]
