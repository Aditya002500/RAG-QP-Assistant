# Architecture & Design Decisions Log

This document records the architectural decisions, trade-offs, and rationale for the VTU Syllabus & Question-Paper Assistant RAG system, as specified in PRD Section 16.

---

### Decision 1: Scope, Branch & Semester Selection
- **Choice**: VTU 2022 Scheme, 5th Semester, Computer Science & Engineering (CSE).
- **Core Subjects**:
  1. `21CS51`: Software Engineering & Project Management (SEPM)
  2. `21CS52`: Computer Networks (CN)
  3. `21CS53`: Database Management Systems (DBMS)
  4. `21CS54`: Artificial Intelligence & Machine Learning (AIML)
- **Rationale**: 5th semester CSE contains well-structured 5-module syllabi with heavy student demand for PYQs. These subjects have distinct module boundaries and rich technical terminology that test retrieval accuracy, topic discrimination, and cross-module ambiguity.

---

### Decision 2: Ingestion & Text Extraction (PyMuPDF + Tesseract OCR)
- **Choice**: Primary extraction with `pymupdf` (fitz). If extracted text on a page has fewer than 50 characters, fall back to `pytesseract` OCR.
- **Rationale**: PyMuPDF is orders of magnitude faster than pypdf and provides accurate page offsets. VTU question papers from older semesters often contain scanned photocopy scans; OCR fallback ensures scanned pages are not discarded. We log OCR rate per document to monitor extraction fidelity.

---

### Decision 3: Structure-Aware Chunking
- **Choice**:
  - **Syllabus**: Module-level parent units with topic-level child chunks (~300–500 tokens). Every chunk retains metadata: `subject_code`, `subject_name`, `module`, `doc_type=syllabus`, `page`, etc.
  - **PYQ**: Question-level atomic chunks. A question (e.g., Q1a, Q1b with marks and module) is never split across multiple chunks.
  - **Header Prefixing**: Each chunk prepends a structured contextual header (e.g., `Subject: Computer Networks | Module 3 | Syllabus`) before embedding.
- **Rationale**: Retaining question integrity is crucial for students (a question makes no sense if sub-part 'b' is severed from sub-part 'a'). Prepending headers ensures the embedding space captures the subject and module context even for short topic phrases.

---

### Decision 4: Vector Store & Metadata Filtering Strategy
- **Choice**: `faiss-cpu` with L2-normalized vector inner product (exact cosine similarity).
- **Metadata Filtering**: Since FAISS flat indices do not natively support arbitrary metadata filtering queries, we implement candidate over-retrieval (retrieve top-30 candidates from FAISS, apply strict metadata predicate matching in Python, then return top-k). If hard pre-filtering by subject/module is requested, we also maintain partition/mask lookup.
- **Rationale**: `faiss-cpu` requires no running server or Docker container, runs in under 10ms locally, loads in under 1 second from disk, and candidate over-retrieval guarantees 100% filtering precision without introducing external database complexity.

---

### Decision 5: Embeddings & Reranker
- **Choice**: Default embedding model `BAAI/bge-small-en-v1.5` (384 dimensions) using BGE's recommended retrieval query prefix (`Represent this sentence for searching relevant passages: `). Compared experimentally against `sentence-transformers/all-MiniLM-L6-v2`.
- **Reranker**: Optional cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking top-20 retrieved candidates down to top-5.
- **Rationale**: Local sentence-transformers run with zero API costs, deterministic offline reproducibility, and low latency (<50ms on Apple Silicon / CPU).

---

### Decision 6: LangGraph Controlled Workflow
- **Choice**: Minimal 6-node state graph:
  1. `analyze_query`: Extracts structured metadata (subject, module, doc_type, year, intent).
  2. `retrieve`: Vector search with extracted/user filters + optional reranking.
  3. `grade`: Threshold score checking to classify retrieved context as `ok` or `weak`.
  4. `rewrite_query`: Rewrites colloquial student queries into VTU academic vocabulary; capped at at most 1 retry.
  5. `generate`: Grounded generation enforcing strict source citation syntax `[1]`, `[2]`. Refuses with `NOT_FOUND` if ungrounded.
  6. `not_found`: Deterministic, helpful refusal explaining what wasn't found and suggesting filter refinements.
- **Rationale**: Prevents uncontrolled agent looping, eliminates hallucination risk, and provides predictable latency.

---

### Decision 7: Rigorous Evaluation Protocol
- **Choice**: 50 hand-crafted questions with gold source annotations:
  - 18 Syllabus Lookup
  - 14 PYQ Lookup
  - 8 Cross-module / Ambiguous
  - 10 Unanswerable
  - Split: 20 Dev questions (used for configuration experiments) and 30 Test questions (strictly held out until final run).
- **Rationale**: Prevents data leakage and ensures reported Hit@k, MRR, Refusal Accuracy, and Latency metrics are authentic and reproducible.
