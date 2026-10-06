"""Unit tests for query filter extraction and metadata predicate matching."""

from vtu_rag.graph.llm_client import LLMClient
from vtu_rag.index.store import VectorStore
from vtu_rag.ingest.chunk import ChunkMetadata


def test_offline_filter_extraction_subject_and_module():
    client = LLMClient()
    # Test syllabus query with subject code and module
    extracted = client._offline_analyze_query("What topics are in Module 3 of 21CS51?")
    import json

    data = json.loads(extracted)
    assert data["subject"] == "21CS51"
    assert data["module"] == 3
    assert data["intent"] == "qa"


def test_offline_filter_extraction_pyq_intent():
    client = LLMClient()
    extracted = client._offline_analyze_query(
        "Show previous year questions on Dijkstra algorithm in Computer Networks 2023"
    )
    import json

    data = json.loads(extracted)
    assert data["intent"] == "pyq_lookup"
    assert data["subject"] == "21CS52"
    assert data["year"] == 2023
    assert data["doc_type"] == "pyq"


def test_vector_store_metadata_filtering():
    chunks = [
        ChunkMetadata(
            chunk_id="c1",
            doc_type="syllabus",
            subject_code="21CS51",
            subject_name="SEPM",
            module=1,
            page=1,
            source_file="f1.pdf",
            text="Chunk 1 content for testing metadata matching.",
        ),
        ChunkMetadata(
            chunk_id="c2",
            doc_type="syllabus",
            subject_code="21CS52",
            subject_name="CN",
            module=1,
            page=1,
            source_file="f2.pdf",
            text="Chunk 2 content for testing metadata matching.",
        ),
        ChunkMetadata(
            chunk_id="c3",
            doc_type="pyq",
            subject_code="21CS51",
            subject_name="SEPM",
            module=2,
            page=1,
            source_file="f3.pdf",
            text="Chunk 3 content for testing metadata matching.",
        ),
    ]

    import numpy as np

    class MockIndex:
        ntotal = 3

        def search(self, q, k):
            return np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]])

    store = VectorStore(index=MockIndex(), metadata=chunks)

    # Filter for 21CS51 syllabus
    res = store.search(
        np.zeros((1, 10)), top_k=5, filters={"subject_code": "21CS51", "doc_type": "syllabus"}
    )
    assert len(res) == 1
    assert res[0][0].chunk_id == "c1"

    # Filter for module 2
    res_mod = store.search(np.zeros((1, 10)), top_k=5, filters={"module": 2})
    assert len(res_mod) == 1
    assert res_mod[0][0].chunk_id == "c3"
