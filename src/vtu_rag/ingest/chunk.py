"""Chunking and metadata enrichment module for VTU RAG.
Implements Section 8 schema validation and Section 11 chunking rules.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from vtu_rag.config import AppSettings, load_config
from vtu_rag.ingest.extract import PDFExtractor
from vtu_rag.ingest.parse import ParsedSection, StructureParser

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_type: str  # syllabus | pyq
    scheme: str = "2022"
    branch: str = "CSE"
    semester: int = 5
    subject_code: str
    subject_name: str
    module: int | None = None
    year: int | None = None
    session: str | None = None
    question_no: str | None = None
    marks: int | None = None
    page: int
    source_file: str
    extraction: str = "text"  # text | ocr
    text: str = Field(min_length=30)


class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, min_chars: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars

    def _split_text_by_tokens(self, text: str, max_chars: int, overlap_chars: int) -> list[str]:
        """Simple, deterministic character/token-aware sliding window splitting for long sections."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # Try to break at a period or newline if possible
            if end < len(text):
                break_point = max(
                    text.rfind(". ", start, end),
                    text.rfind("\n", start, end),
                    text.rfind("; ", start, end),
                )
                if break_point != -1 and break_point > start + (max_chars // 2):
                    end = break_point + 1
            chunk_slice = text[start:end].strip()
            if len(chunk_slice) >= self.min_chars:
                chunks.append(chunk_slice)
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
        return chunks

    def chunk_sections(self, sections: list[ParsedSection]) -> list[ChunkMetadata]:
        chunks: list[ChunkMetadata] = []
        counter: dict[str, int] = defaultdict(int)

        # Approximate 4 characters per token
        max_chars = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        for sec in sections:
            # Header prefix to ground embeddings
            mod_label = f"Module {sec.module}" if sec.module else "General Overview"
            if sec.doc_type == "syllabus":
                prefix = f"Subject: {sec.subject_name} ({sec.subject_code}) | {mod_label} | Syllabus"
            else:
                session_str = sec.session or (f"Year {sec.year}" if sec.year else "PYQ")
                q_label = f" | Q.{sec.question_no}" if sec.question_no else ""
                prefix = f"Subject: {sec.subject_name} ({sec.subject_code}) | {mod_label}{q_label} | PYQ ({session_str})"

            if sec.doc_type == "pyq":
                # PYQ Rule: Never split an exam question across multiple chunks! Keep marks, module, question together.
                chunk_body = f"{prefix}\n{sec.text.strip()}"
                if len(chunk_body) < self.min_chars:
                    continue

                counter[sec.subject_code] += 1
                c_idx = counter[sec.subject_code]
                mod_key = f"m{sec.module}" if sec.module else "gen"
                chunk_id = f"{sec.branch.lower()}{sec.scheme[-2:]}_{sec.semester}_{sec.subject_code.lower()}_{mod_key}_pyq_{c_idx:04d}"

                chunk_obj = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_type="pyq",
                    scheme=sec.scheme,
                    branch=sec.branch,
                    semester=sec.semester,
                    subject_code=sec.subject_code,
                    subject_name=sec.subject_name,
                    module=sec.module,
                    year=sec.year,
                    session=sec.session,
                    question_no=sec.question_no,
                    marks=sec.marks,
                    page=sec.page,
                    source_file=sec.source_file,
                    extraction=sec.extraction,
                    text=chunk_body,
                )
                chunks.append(chunk_obj)
            else:
                # Syllabus Rule: Module parent unit. Split long modules into child chunks by topic/size.
                raw_body = sec.text.strip()
                sub_chunks = self._split_text_by_tokens(raw_body, max_chars, overlap_chars)

                for sub in sub_chunks:
                    chunk_body = f"{prefix}\n{sub}"
                    if len(chunk_body) < self.min_chars:
                        continue

                    counter[sec.subject_code] += 1
                    c_idx = counter[sec.subject_code]
                    mod_key = f"m{sec.module}" if sec.module else "gen"
                    chunk_id = f"{sec.branch.lower()}{sec.scheme[-2:]}_{sec.semester}_{sec.subject_code.lower()}_{mod_key}_syl_{c_idx:04d}"

                    chunk_obj = ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_type="syllabus",
                        scheme=sec.scheme,
                        branch=sec.branch,
                        semester=sec.semester,
                        subject_code=sec.subject_code,
                        subject_name=sec.subject_name,
                        module=sec.module,
                        year=None,
                        session=None,
                        question_no=None,
                        marks=None,
                        page=sec.page,
                        source_file=sec.source_file,
                        extraction=sec.extraction,
                        text=chunk_body,
                    )
                    chunks.append(chunk_obj)

        return chunks


def run_ingestion_pipeline(config: AppSettings | None = None) -> list[ChunkMetadata]:
    """Runs end-to-end ingestion pipeline: PDFs -> text/OCR extraction -> structure parse -> chunking -> JSONL."""
    if config is None:
        config = load_config()

    raw_dir = Path(config.paths.raw_data_dir)
    processed_dir = Path(config.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(config.paths.chunks_file)

    extractor = PDFExtractor(min_char_threshold=50)
    parser = StructureParser()
    chunker = Chunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_chars=config.chunking.min_chunk_chars,
    )

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    all_chunks: list[ChunkMetadata] = []

    for pdf_path in pdf_files:
        pages = extractor.extract_document(pdf_path)
        sections = parser.parse_document(pages)
        doc_chunks = chunker.chunk_sections(sections)
        all_chunks.extend(doc_chunks)

    # Sort chunks deterministically by chunk_id for idempotency
    all_chunks.sort(key=lambda c: c.chunk_id)

    # Save to JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(ch.model_dump(), ensure_ascii=False) + "\n" for ch in all_chunks)

    logger.info(
        "Ingestion complete: %d chunks written to %s (100%% validated)",
        len(all_chunks),
        out_file,
    )
    return all_chunks


if __name__ == "__main__":
    cfg = load_config()
    chunks = run_ingestion_pipeline(cfg)
    print(f"Successfully processed and generated {len(chunks)} chunks.")
    if chunks:
        print("Sample Chunk 1:")
        print(json.dumps(chunks[0].model_dump(), indent=2))
