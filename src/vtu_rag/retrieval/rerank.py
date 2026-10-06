"""Cross-encoder reranking module for VTU RAG.
Reranks candidate chunks using a trained cross-encoder model.
"""

import logging

from sentence_transformers import CrossEncoder

from vtu_rag.config import RerankerConfig
from vtu_rag.ingest.chunk import ChunkMetadata

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved candidate passages using a CrossEncoder."""

    def __init__(self, config: RerankerConfig | None = None):
        if config is None:
            config = RerankerConfig()
        self.config = config
        self._model: CrossEncoder | None = None

        if self.config.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info("Loading CrossEncoder model: %s", self.config.model_name)
            self._model = CrossEncoder(self.config.model_name)
        except Exception as e:
            logger.warning(
                "Could not load cross-encoder model %s: %s. Reranking will be disabled.",
                self.config.model_name,
                e,
            )
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[tuple[ChunkMetadata, float]],
        top_n: int | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """Reranks a list of (ChunkMetadata, initial_score) tuples using cross-encoder."""
        if not self.config.enabled or self._model is None or not candidates:
            return candidates[: top_n or self.config.top_n]

        target_n = top_n or self.config.top_n
        pairs = [(query, chunk.text) for chunk, _ in candidates]

        try:
            scores = self._model.predict(pairs)
            # Combine chunk with cross-encoder score
            scored_candidates = [
                (chunk, float(score))
                for (chunk, _), score in zip(candidates, scores)
            ]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[:target_n]
        except Exception as e:
            logger.warning("Reranking failed: %s. Returning dense retrieval order.", e)
            return candidates[:target_n]
