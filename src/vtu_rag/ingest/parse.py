"""Structure-aware parser for VTU Syllabi and Previous-Year Question Papers (PYQs).
Complies with FR-3 and PRD Section 8 metadata requirements.
"""

import re
from dataclasses import dataclass
from typing import Any

SUBJECT_NAMES: dict[str, str] = {
    "21CS51": "Software Engineering & Project Management",
    "21CS52": "Computer Networks",
    "21CS53": "Database Management Systems",
    "21CS54": "Artificial Intelligence & Machine Learning",
    "21IS51": "Software Engineering & Project Management",
    "21IS52": "Computer Networks",
    "21IS53": "Database Management Systems",
}


@dataclass
class DocumentMetadata:
    scheme: str
    branch: str
    semester: int
    subject_code: str
    subject_name: str
    doc_type: str  # 'syllabus' or 'pyq'
    year: int | None = None
    session: str | None = None


@dataclass
class ParsedSection:
    doc_type: str
    scheme: str
    branch: str
    semester: int
    subject_code: str
    subject_name: str
    module: int | None
    year: int | None
    session: str | None
    question_no: str | None
    marks: int | None
    page: int
    source_file: str
    extraction: str
    text: str


def parse_filename_metadata(filename: str) -> DocumentMetadata:
    """Parse standard naming convention: <scheme>_<branch>_<sem>_<subjectcode>_<doctype>[_<year>].pdf"""
    stem = filename.replace(".pdf", "")
    parts = stem.split("_")
    scheme = parts[0] if len(parts) > 0 else "2022"
    branch = parts[1] if len(parts) > 1 else "CSE"
    sem = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 5
    subject_code = parts[3].upper() if len(parts) > 3 else "21CS51"
    doc_type = "syllabus" if "syl" in stem.lower() else "pyq"

    year = None
    if doc_type == "pyq":
        for part in reversed(parts):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                break

    subject_name = SUBJECT_NAMES.get(subject_code, f"Course {subject_code}")
    return DocumentMetadata(
        scheme=scheme,
        branch=branch,
        semester=sem,
        subject_code=subject_code,
        subject_name=subject_name,
        doc_type=doc_type,
        year=year,
    )


class StructureParser:
    """Parses extracted PDF pages into structured syllabus modules and PYQ questions."""

    def parse_syllabus(
        self, pages_data: list[dict[str, Any]], meta: DocumentMetadata
    ) -> list[ParsedSection]:
        sections: list[ParsedSection] = []

        # Combine text while tracking page numbers
        # Pattern to detect modules
        module_pattern = re.compile(
            r"(?:^|\n)\s*MODULE\s*([1-5])\s*[:\-]?\s*", re.IGNORECASE
        )
        textbook_pattern = re.compile(
            r"(?:^|\n)\s*(?:TEXTBOOKS?|REFERENCE BOOKS?)\s*[:\-]?\s*", re.IGNORECASE
        )

        for p in pages_data:
            text = p["text"]
            page_num = p["page"]
            source_file = p["source_file"]
            extraction = p["extraction"]

            # Split page content by Module headers
            splits = list(module_pattern.finditer(text))
            tb_split = list(textbook_pattern.finditer(text))

            if not splits and not tb_split:
                # General introductory/overview section
                clean_text = text.strip()
                if len(clean_text) >= 30:
                    sections.append(
                        ParsedSection(
                            doc_type="syllabus",
                            scheme=meta.scheme,
                            branch=meta.branch,
                            semester=meta.semester,
                            subject_code=meta.subject_code,
                            subject_name=meta.subject_name,
                            module=None,
                            year=None,
                            session=None,
                            question_no=None,
                            marks=None,
                            page=page_num,
                            source_file=source_file,
                            extraction=extraction,
                            text=clean_text,
                        )
                    )
                continue

            # Process module segments on this page
            indices = []
            for m in splits:
                indices.append((m.start(), int(m.group(1)), "module"))
            for tb in tb_split:
                indices.append((tb.start(), None, "textbooks"))

            indices.sort(key=lambda x: x[0])

            # Pre-module introductory text on the page
            first_idx = indices[0][0]
            if first_idx > 30:
                pre_text = text[:first_idx].strip()
                if len(pre_text) >= 30:
                    sections.append(
                        ParsedSection(
                            doc_type="syllabus",
                            scheme=meta.scheme,
                            branch=meta.branch,
                            semester=meta.semester,
                            subject_code=meta.subject_code,
                            subject_name=meta.subject_name,
                            module=None,
                            year=None,
                            session=None,
                            question_no=None,
                            marks=None,
                            page=page_num,
                            source_file=source_file,
                            extraction=extraction,
                            text=pre_text,
                        )
                    )

            # Segment between headers
            for i, (pos, mod_num, sec_type) in enumerate(indices):
                end_pos = indices[i + 1][0] if i + 1 < len(indices) else len(text)
                sec_text = text[pos:end_pos].strip()
                if len(sec_text) >= 30:
                    sections.append(
                        ParsedSection(
                            doc_type="syllabus",
                            scheme=meta.scheme,
                            branch=meta.branch,
                            semester=meta.semester,
                            subject_code=meta.subject_code,
                            subject_name=meta.subject_name,
                            module=mod_num,
                            year=None,
                            session=None,
                            question_no=None,
                            marks=None,
                            page=page_num,
                            source_file=source_file,
                            extraction=extraction,
                            text=sec_text,
                        )
                    )

        return sections

    def parse_pyq(
        self, pages_data: list[dict[str, Any]], meta: DocumentMetadata
    ) -> list[ParsedSection]:
        sections: list[ParsedSection] = []

        # Find session string from headers
        session = None
        for p in pages_data:
            session_match = re.search(
                r"(?:Examination,?\s*)([A-Za-z]+(?:\s*\/\s*[A-Za-z]+)?\s*\d{4})",
                p["text"],
            )
            if session_match:
                session = session_match.group(1).strip()
                break
        if not session and meta.year:
            session = f"Session {meta.year}"

        # Standard question regex:
        # e.g.: Module 1 | Q.1a (10 Marks)\n...
        # or Module 1\nQ.1a (10 Marks)
        # or Q.1a (10 Marks)
        q_header_pattern = re.compile(
            r"(?:(?:Module|MODULE)\s*([1-5])\s*\|?\s*)?Q\.?\s*(\d+[a-z]?)\s*(?:\((\d+)\s*Marks?\)|\[(\d+)\s*Marks?\])?",
            re.IGNORECASE,
        )

        for p in pages_data:
            text = p["text"]
            page_num = p["page"]
            source_file = p["source_file"]
            extraction = p["extraction"]

            matches = list(q_header_pattern.finditer(text))
            if not matches:
                # Check for line-by-line questions or OCR text
                lines = text.split("\n")
                current_mod = 1
                for line in lines:
                    line_clean = line.strip()
                    m_line = q_header_pattern.search(line_clean)
                    if m_line and len(line_clean) >= 30:
                        mod_val = (
                            int(m_line.group(1)) if m_line.group(1) else current_mod
                        )
                        q_no = m_line.group(2)
                        marks_val = (
                            int(m_line.group(3) or m_line.group(4))
                            if (m_line.group(3) or m_line.group(4))
                            else 10
                        )
                        sections.append(
                            ParsedSection(
                                doc_type="pyq",
                                scheme=meta.scheme,
                                branch=meta.branch,
                                semester=meta.semester,
                                subject_code=meta.subject_code,
                                subject_name=meta.subject_name,
                                module=mod_val,
                                year=meta.year,
                                session=session,
                                question_no=q_no,
                                marks=marks_val,
                                page=page_num,
                                source_file=source_file,
                                extraction=extraction,
                                text=line_clean,
                            )
                        )
                continue

            current_module: int | None = None
            for idx, m in enumerate(matches):
                if m.group(1):
                    current_module = int(m.group(1))

                q_no = m.group(2)
                marks_str = m.group(3) or m.group(4)
                marks = int(marks_str) if marks_str else 10

                # Question body extends until the next question header
                start_body = m.end()
                end_body = (
                    matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                )
                body = text[start_body:end_body].strip()
                full_q_text = f"Q.{q_no} ({marks} Marks): {body}" if body else m.group(0)

                # Determine module from question number if not explicit
                # VTU standard: Q1, Q2 -> Module 1; Q3, Q4 -> Module 2; Q5, Q6 -> Module 3; Q7, Q8 -> Module 4; Q9, Q10 -> Module 5
                if current_module is None:
                    q_int = re.search(r"\d+", q_no)
                    if q_int:
                        num = int(q_int.group(0))
                        current_module = (num + 1) // 2

                if len(full_q_text) >= 20:
                    sections.append(
                        ParsedSection(
                            doc_type="pyq",
                            scheme=meta.scheme,
                            branch=meta.branch,
                            semester=meta.semester,
                            subject_code=meta.subject_code,
                            subject_name=meta.subject_name,
                            module=current_module,
                            year=meta.year,
                            session=session,
                            question_no=q_no,
                            marks=marks,
                            page=page_num,
                            source_file=source_file,
                            extraction=extraction,
                            text=full_q_text,
                        )
                    )

        return sections

    def parse_document(
        self, pages_data: list[dict[str, Any]]
    ) -> list[ParsedSection]:
        if not pages_data:
            return []
        first_file = pages_data[0]["source_file"]
        meta = parse_filename_metadata(first_file)

        if meta.doc_type == "syllabus":
            return self.parse_syllabus(pages_data, meta)
        else:
            return self.parse_pyq(pages_data, meta)
