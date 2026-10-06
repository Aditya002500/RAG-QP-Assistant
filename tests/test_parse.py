"""Unit tests for structure-aware parsing of VTU documents."""

from vtu_rag.ingest.parse import (
    DocumentMetadata,
    StructureParser,
    parse_filename_metadata,
)


def test_parse_filename_metadata_syllabus():
    meta = parse_filename_metadata("2022_CSE_5_21CS51_syl.pdf")
    assert meta.scheme == "2022"
    assert meta.branch == "CSE"
    assert meta.semester == 5
    assert meta.subject_code == "21CS51"
    assert meta.doc_type == "syllabus"
    assert meta.subject_name == "Software Engineering & Project Management"
    assert meta.year is None


def test_parse_filename_metadata_pyq():
    meta = parse_filename_metadata("2022_CSE_5_21CS52_pyq_2023.pdf")
    assert meta.scheme == "2022"
    assert meta.branch == "CSE"
    assert meta.semester == 5
    assert meta.subject_code == "21CS52"
    assert meta.doc_type == "pyq"
    assert meta.subject_name == "Computer Networks"
    assert meta.year == 2023


def test_parse_syllabus_modules():
    parser = StructureParser()
    sample_text = """VISVESVARAYA TECHNOLOGICAL UNIVERSITY
Course Title: Computer Networks
MODULE 1:
Introduction to Computer Networks, Packet Switching and Circuit Switching.
MODULE 2:
Application Layer, DNS, HTTP and Socket Programming.
TEXTBOOKS:
1. Kurose and Ross, Computer Networking.
"""
    pages_data = [
        {
            "source_file": "2022_CSE_5_21CS52_syl.pdf",
            "page": 1,
            "text": sample_text,
            "extraction": "text",
        }
    ]
    sections = parser.parse_syllabus(
        pages_data,
        DocumentMetadata("2022", "CSE", 5, "21CS52", "Computer Networks", "syllabus"),
    )
    assert len(sections) >= 2
    modules = [s.module for s in sections if s.module is not None]
    assert 1 in modules
    assert 2 in modules


def test_parse_pyq_questions():
    parser = StructureParser()
    sample_pyq = """VISVESVARAYA TECHNOLOGICAL UNIVERSITY
Fifth Semester B.E. Examination, Dec 2023 / Jan 2024
Subject: Database Management Systems (21CS53)
Module 1 | Q.1a (10 Marks)
Explain Three-Schema Architecture and Data Independence with diagrams.
Module 1 | Q.2a (10 Marks)
Describe Entity-Relationship model constructs and cardinality constraints.
"""
    pages_data = [
        {
            "source_file": "2022_CSE_5_21CS53_pyq_2024.pdf",
            "page": 1,
            "text": sample_pyq,
            "extraction": "text",
        }
    ]
    sections = parser.parse_pyq(
        pages_data,
        DocumentMetadata(
            "2022", "CSE", 5, "21CS53", "Database Management Systems", "pyq", year=2024
        ),
    )
    assert len(sections) == 2
    assert sections[0].module == 1
    assert sections[0].marks == 10
    assert "Three-Schema" in sections[0].text
