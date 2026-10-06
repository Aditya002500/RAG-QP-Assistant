"""Configuration loader and schema definition for VTU RAG."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class AppMeta(BaseModel):
    name: str = "VTU Syllabus & Question-Paper Assistant"
    version: str = "1.0.0"
    scheme: str = "2022"
    branch: str = "CSE"
    semester: int = 5


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    index_dir: str = "data/index"
    chunks_file: str = "data/processed/chunks.jsonl"
    faiss_index: str = "data/index/vtu_index.faiss"
    metadata_file: str = "data/index/metadata.jsonl"
    eval_set: str = "eval/eval_set.yaml"
    eval_results_dir: str = "eval/results"
    query_log_file: str = "data/query_logs.jsonl"


class ChunkingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 80
    min_chunk_chars: int = 30


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    normalize_embeddings: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    candidate_k: int = 30
    use_metadata_filter: bool = True
    similarity_metric: str = "cosine"


class RerankerConfig(BaseModel):
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class GradingConfig(BaseModel):
    relevance_threshold: float = 0.45
    max_retries: int = 1


class LLMConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_tokens: int = 1024
    gemini_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )


class AppSettings(BaseModel):
    app: AppMeta = Field(default_factory=AppMeta)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    grading: GradingConfig = Field(default_factory=GradingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(config_path: str = "config.yaml") -> AppSettings:
    """Load configuration from YAML file, with environment variable overlays."""
    path = Path(config_path)
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    settings = AppSettings(**data)
    # Ensure environment variables overlay
    if os.getenv("LLM_PROVIDER"):
        settings.llm.provider = os.getenv("LLM_PROVIDER", "gemini")
    if os.getenv("GEMINI_API_KEY"):
        settings.llm.gemini_api_key = os.getenv("GEMINI_API_KEY")
    if os.getenv("OPENAI_API_KEY"):
        settings.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_BASE_URL"):
        settings.llm.openai_base_url = os.getenv("OPENAI_BASE_URL")

    return settings
