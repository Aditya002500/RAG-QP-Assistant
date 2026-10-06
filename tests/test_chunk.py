"""Unit tests for chunking logic and metadata schema validation."""

import pytest
from pydantic import ValidationError

from vtu_rag.ingest.chunk import Chunker, ChunkMetadata
from vtu_rag.ingest.parse import ParsedSection


def test_chunk_metadata_schema_valid():
    chunk = ChunkMetadata(
        chunk_id="cse22_5_21cs51_m1_syl_0001",
        doc_type="syllabus",
        scheme="2022",
        branch="CSE",
        semester=5,
        subject_code="21CS51",
        subject_name="Software Engineering",
        module=1,
        page=1,
        source_file="2022_CSE_5_21CS51_syl.pdf",
        extraction="text",
        text="Introduction to Software Engineering and Waterfall Process Model.",
    )
    assert chunk.subject_code == "21CS51"
    assert chunk.module == 1


def test_chunk_metadata_rejects_too_short():
    with pytest.raises(ValidationError):
        ChunkMetadata(
            chunk_id="short_chunk",
            doc_type="syllabus",
            subject_code="21CS51",
            subject_name="SE",
            page=1,
            source_file="test.pdf",
            text="Too short",  # < 30 chars
        )


def test_chunker_prefixes_headers():
    chunker = Chunker(chunk_size=500, chunk_overlap=80, min_chars=30)
    sections = [
        ParsedSection(
            doc_type="syllabus",
            scheme="2022",
            branch="CSE",
            semester=5,
            subject_code="21CS52",
            subject_name="Computer Networks",
            module=2,
            year=None,
            session=None,
            question_no=None,
            marks=None,
            page=2,
            source_file="2022_CSE_5_21CS52_syl.pdf",
            extraction="text",
            text="Principles of Application Layer Protocols, HTTP, DNS, and Socket Programming.",
        )
    ]
    chunks = chunker.chunk_sections(sections)
    assert len(chunks) == 1
    assert "Subject: Computer Networks (21CS52) | Module 2 | Syllabus" in chunks[0].text
    assert chunks[0].chunk_id.startswith("cse22_5_21cs52_m2_syl_")


def test_chunker_keeps_pyq_question_atomic():
    chunker = Chunker(chunk_size=500, chunk_overlap=80, min_chars=30)
    sections = [
        ParsedSection(
            doc_type="pyq",
            scheme="2022",
            branch="CSE",
            semester=5,
            subject_code="21CS54",
            subject_name="Artificial Intelligence & Machine Learning",
            module=3,
            year=2024,
            session="June / July 2024",
            question_no="5a",
            marks=10,
            page=1,
            source_file="2022_CSE_5_21CS54_pyq_2024.pdf",
            extraction="text",
            text="Q.5a (10 Marks): Explain Goal Stack Planning in Blocks World with an example start and goal state.",
        )
    ]
    chunks = chunker.chunk_sections(sections)
    assert len(chunks) == 1
    assert chunks[0].doc_type == "pyq"
    assert chunks[0].question_no == "5a"
    assert chunks[0].marks == 10
