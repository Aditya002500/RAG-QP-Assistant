# PRD: VTU Syllabus & Question-Paper Assistant (RAG)

Author: Aditya Kumar | Version 1.0 | 6 October 2026 | Status: Ready to build

## 1. Overview

VTU Study Assistant is a retrieval-augmented generation (RAG) app. A VTU student asks a question in plain English, and the assistant answers only from the syllabus and previous-year question papers (PYQs) in its index, with citations (subject code, module, page, document type). When the sources do not contain the answer, it says so instead of guessing.

The project has two jobs: (1) be genuinely useful to students, and (2) work as a portfolio piece that proves RAG engineering skills: structured ingestion, a LangGraph flow, and a measured evaluation with real numbers.

## 2. Problem statement

- Syllabus and PYQ PDFs are scattered across scheme, branch, semester and subject. Answering 'which module covers X?' or 'how often was X asked?' means opening many files.
- Many old question papers are scanned images, so Ctrl+F does not work on them.
- Generic chatbots do not know scheme-specific module contents, invent topics, and cannot cite a page.

## 3. Goals and non-goals

### Goals

- G1: Answer syllabus questions from the index with correct citations.
- G2: Find previous-year questions by topic, module, subject or year.
- G3: Refuse to answer when the sources do not support an answer.
- G4: Measure quality with a hand-built evaluation set and report retrieval and answer metrics.
- G5: Run locally for free (free LLM tier or a local model) and be deployable as a public demo.
- G6: Finish an MVP in about one week.

### Non-goals

- No user accounts, payments or mobile app.
- No generating full model answers or answer keys for exam papers.
- No coverage of every branch. Start with one branch, one semester, 3 to 5 subjects.
- No model fine-tuning.
- No large-scale scraping.
- No claim of affiliation with VTU. The app is unofficial.

## 4. Users and user stories

- Primary user: a VTU undergraduate preparing for internals or semester exams.
- Secondary user: a recruiter or interviewer reading the repo. They need a clear README, results table, architecture diagram and a working demo.

User stories:

- US-1: As a student, I ask 'What topics are in Module 3 of \<subject>?' and get the topic list with a page citation.
- US-2: I ask 'Which PYQs were asked on \<topic>?' and see question text, marks, year and module.
- US-3: I filter by branch, semester and subject so answers come only from my subject.
- US-4: I open the sources behind an answer and read the exact passage.
- US-5: I get a clear 'not found in the documents' message instead of a made-up answer.
- US-6: As a developer, I run one command to evaluate the system and get a metrics table.
- US-7: I can change chunk size, embedding model, k and reranker from a config file without editing code.

## 5. Scope and priorities

| Priority | Item |
| --- | --- |
| Must (MVP) | Ingestion for 1 branch, 1 semester, 3 to 5 subjects. Metadata on every chunk. Vector index. Q&A with citations. PYQ lookup. Metadata filters. Streamlit UI. Evaluation harness with a 50-question set. README with results. |
| Must (for resume numbers) | Baseline vs improved experiments: chunking, embedding model and reranker, reported on the held-out test split. |
| Should | LangGraph rewrite-and-retry loop. OCR fallback for scanned pages. 'Most asked topics' view built from PYQs. GitHub Actions for lint and tests. |
| Could | Hybrid search (BM25 plus dense). Multi-turn memory. Docker image. Public deployment (Streamlit Community Cloud or Hugging Face Spaces). |
| Won't (v1) | More branches, auth, fine-tuning, answer generation for full papers. |

### Success metrics (targets, measured on the test split)

| Metric | Target |
| --- | --- |
| Retrieval hit@5 | 0.85 or higher |
| MRR | 0.60 or higher |
| Citation correctness (manual check of 20 answers) | 90% or higher |
| Refusal accuracy on unanswerable questions | 80% or higher |
| p50 end-to-end latency with a hosted LLM | under 5 seconds |
| Setup from a fresh clone | 3 commands or fewer |

If a target is missed, report the real number and the failure analysis. Do not adjust the test set to hit a target.

## 6. Functional requirements

| ID | Requirement | Priority | Acceptance criterion |
| --- | --- | --- | --- |
| FR-1 | Ingestion CLI that turns PDFs in data/raw into structured JSONL. Re-running is idempotent. | Must | Running twice gives identical output. |
| FR-2 | Text extraction with PyMuPDF. Pages with fewer than 50 characters fall back to OCR (Tesseract). | Must | OCR rate per file is logged. |
| FR-3 | Structure-aware parsing. Syllabus: split by subject and module. PYQ: split by question number (for example 1a, 1b), capturing marks, module and exam session. | Must | 10 random files spot-checked by hand and parsed correctly. |
| FR-4 | Every chunk carries the metadata schema in section 8. | Must | Schema validation passes for 100% of chunks. |
| FR-5 | Embed chunks and build a persisted vector index. A flag rebuilds it. | Must | Index loads in under 5 seconds. |
| FR-6 | Query analysis extracts filters (subject code or name, module, doc type, year) from the question. If nothing is found, search without filters. | Must | Correct filters on 8 of 10 sample questions. |
| FR-7 | Retrieve top-k (default 5) with metadata filtering. Optional reranker. | Must | k, filters and reranker are config-driven. |
| FR-8 | Relevance grading. If retrieval is weak, rewrite the query once and retry, otherwise return the not-found message. | Should | Unanswerable test questions are refused. |
| FR-9 | Generation grounded only in retrieved chunks, with numbered citations mapped to a source list. | Must | Every claim in 20 sampled answers is traceable to a cited chunk. |
| FR-10 | Streamlit chat UI with sidebar filters and expandable sources (page and passage). | Must | All three flows work end to end. |
| FR-11 | PYQ mode: 'find questions on X' returns question text, marks, year, module and source page. | Must | Returns 3 or more correct items for 5 sample topics. |
| FR-12 | Configuration through .env (secrets) and config.yaml (parameters). | Must | No parameter hardcoded in code. |
| FR-13 | Evaluation CLI runs the eval set and writes metrics JSON and a markdown table. Fixed seeds. | Must | Same input gives the same retrieval metrics. |
| FR-14 | Query log (JSONL): filters, retrieved chunk IDs, scores, latency. No personal data. | Should | Log written for every query. |
| FR-15 | Unit tests for parsing, chunking, filter extraction and metrics. | Must | pytest passes in CI. |

## 7. Non-functional requirements

- Cost: zero. Use a free LLM tier or a local model, and local embeddings.
- Latency: retrieval under 300 ms locally. p50 end-to-end under 5 seconds with a hosted LLM.
- Reproducibility: pinned dependencies, fixed random seeds, one command to rebuild the index.
- Privacy: no user accounts and no personal data stored. API keys only in .env, which is gitignored.
- Code quality: type hints, ruff for lint and format, pytest, small modules.
- Honesty: the UI shows an 'unofficial, verify with the VTU website' note.

## 8. Data

### Sources

- Syllabus PDFs for your scheme, branch and semester from the official VTU website.
- Previous-year question papers, at least 3 years per subject, from publicly downloadable sources.
- Pick YOUR branch and semester so you can judge answer quality yourself. Start with 3 to 5 subjects.

### Legal and hygiene

- Use only publicly available documents and check each site's terms before downloading.
- Do not commit PDFs to the repo. Add data/ to .gitignore and document where to get the files in docs/data\_sources.md.
- Keep file names consistent, for example \<scheme>*\<branch>*\<sem>*\<subjectcode>*\<doctype>\_\<year>.pdf.

### Chunk metadata schema

```yaml
chunk_id: ise22_5_21is51_m3_syl_0007
doc_type: syllabus        # syllabus | pyq
scheme: '2022'
branch: ISE
semester: 5
subject_code: 21IS51
subject_name: <name>
module: 3                 # null if not applicable
year: null                # exam year for PYQ
session: null             # for example Dec 2024 / Jan 2025
question_no: null         # for example 3b (PYQ only)
marks: null
page: 12
source_file: <file name>
extraction: text          # text | ocr
text: <chunk text>
```

Assumption to verify: VTU subjects are usually organised into modules (often five). Confirm from your actual PDFs and adjust the parser.

### Data quality checks

- Spot-check 10 percent of pages by hand after extraction.
- Log the OCR page rate per file. If OCR text looks bad, drop the file or fix it manually and note this in the README.
- Reject chunks under 30 characters.

## 9. Architecture

```text
OFFLINE
PDFs -> extract (PyMuPDF / OCR) -> clean -> structure parse -> chunk -> embed -> vector index + metadata

ONLINE (LangGraph)
question -> analyze_query -> retrieve -> grade --ok--> generate -> answer + citations
                              ^           |
                              |           +--weak--> rewrite_query (max 1 retry)
                              +-----------------------+   still weak --> not_found
```

### LangGraph state

- question, filters, rewritten\_question
- retrieved (list of chunks with scores)
- grade (ok or weak), retries (int)
- answer, citations, status (answered or not\_found)

### Nodes

| Node | Job |
| --- | --- |
| analyze\_query | LLM call with a JSON schema that extracts subject, module, doc\_type, year and intent (qa or pyq\_lookup). |
| retrieve | Vector search with metadata filter, then optional reranker. Returns top-k with scores. |
| grade | Score threshold first, optionally an LLM relevance check on the top chunks. Sets ok or weak. |
| rewrite\_query | LLM rewrites the question using the document vocabulary. Allowed once. |
| generate | Answers using only retrieved chunks and attaches numbered citations. |
| not\_found | Returns a fixed message and suggests which filter or phrasing to try. |

Keep the graph this small. Do not add agents or tools that the problem does not need.

## 10. Tech stack

| Layer | Choice | Why and alternatives |
| --- | --- | --- |
| Language | Python 3.11 or newer | Your main ML language. |
| PDF parsing | PyMuPDF (pymupdf) | Fast, gives page numbers. Alternative: pdfplumber. |
| OCR fallback | Tesseract through pytesseract | Free. Needs the Tesseract binary installed. |
| Orchestration | LangChain plus LangGraph | Matches your skills. Pin versions, because both change quickly. |
| Embeddings | sentence-transformers: BAAI/bge-small-en-v1.5 and all-MiniLM-L6-v2 | Free and local. Compare both in the experiments. |
| Vector store | FAISS (faiss-cpu) | Simple. Alternative: Chroma if you want built-in persistence and filtering. |
| Reranker | BAAI/bge-reranker-base or cross-encoder/ms-marco-MiniLM-L-6-v2 | Used as an on/off experiment. |
| LLM | Provider-agnostic through LangChain. A free-tier hosted model (for example a Gemini Flash-class model) or a local model through Ollama | Switch by config. Check current free-tier limits. |
| UI | Streamlit | Fastest path to a demo. |
| Evaluation | Custom scripts for retrieval metrics. Optional RAGAS for answer metrics. | Retrieval metrics must be your own and reproducible. |
| Quality | pytest, ruff, GitHub Actions | Lint and tests on every push. |
| Packaging | requirements.txt (pinned) or uv, optional Dockerfile | Reproducible setup. |

FAISS note: it has no native metadata filtering. Either filter in Python after retrieving a larger candidate set (for example top 30, then filter, then keep 5) or choose Chroma.

## 11. Chunking and retrieval design

### Chunking rules

- Syllabus: one parent unit per module. Split long modules into child chunks by topic or by size. Every child keeps the module and subject metadata.
- PYQ: one chunk per question (with its sub-parts), never split a question across chunks. Keep marks and module.
- Prefix each chunk with a short header such as 'Subject: \<name> | Module 3 | Syllabus' so the embedding captures context.

### Config defaults and experiment grid

| Parameter | Default | Grid to test |
| --- | --- | --- |
| chunk\_size (tokens) | 500 | 300, 500, 800 |
| chunk\_overlap | 80 | 0, 80 |
| embedding model | bge-small-en-v1.5 | bge-small, MiniLM |
| top\_k | 5 | 3, 5, 8 |
| metadata filter | on | on, off |
| reranker | off | off, on (rerank top 20 to 5) |

Do not run the full grid blindly. Change one factor at a time from the default and record each result.

### Retrieval notes

- Embed queries with the model's recommended query prefix if it has one (check the model card).
- Use cosine similarity with normalised vectors.
- Optional later: hybrid search with BM25 to catch exact subject codes and keywords.

## 12. Prompts and schemas

### Generation system prompt

```text
You are a study assistant for VTU students.
Answer ONLY using the numbered context passages below.
Cite every statement with the passage number, like [1] or [2][3].
If the context does not contain the answer, reply exactly: NOT_FOUND.
Do not use outside knowledge. Do not invent topics, marks or years.
Be concise. For lists of topics, use bullet points.
```

### Query analysis output (structured)

```yaml
intent: qa | pyq_lookup
subject: <code or name or null>
module: <int or null>
doc_type: syllabus | pyq | null
year: <int or null>
search_query: <cleaned query for retrieval>
```

### Behaviour rules

- If the model returns NOT\_FOUND or no passage is cited, show the not-found message.
- Check that each cited number exists in the retrieved list. If not, drop that citation and flag the answer.
- Not-found message: 'I could not find this in the indexed VTU documents. Try selecting the subject or module, or rephrase.'

## 13. Evaluation plan

This section is what turns the project into a resume bullet. Build it carefully.

### Evaluation set (50 questions, written by you)

| Type | Count | Example |
| --- | --- | --- |
| Syllabus lookup | 18 | Which module covers \<topic> in \<subject>? |
| PYQ lookup | 14 | In which years was a question on \<topic> asked? |
| Cross-module or ambiguous | 8 | Compare \<topic A> and \<topic B> coverage. |
| Unanswerable | 10 | A topic or subject not in the index. |

Write the questions yourself, or generate drafts with an LLM and verify every gold label by hand. Each item stores the gold source.

```yaml
id: q012
question: Which module covers <topic> in <subject>?
type: syllabus_lookup
answerable: true
gold:
  - subject_code: 21IS51
    module: 3
    doc_type: syllabus
split: test
```

### Splits

- Dev split: 20 questions (use for choosing the configuration).
- Test split: 30 questions (touch only for the final report). Keep a similar mix of types in both.

### Metrics

- Retrieval hit@k (k = 1, 3, 5): at least one gold (subject\_code, module, doc\_type) appears in the top-k chunks.
- MRR: reciprocal rank of the first gold hit.
- Refusal accuracy: share of unanswerable questions that return not-found, plus false-refusal rate on answerable ones.
- Citation correctness: manual check of 20 answers, are all cited passages real and supportive?
- Faithfulness: manual 0/1 on the same 20 answers (any unsupported claim fails).
- Latency: p50 and p95, split into retrieval and generation.

### Protocol

1. Run the baseline (defaults, no reranker) on dev and test. Save the numbers.
2. Change one factor at a time on dev only. Keep a table of all runs, not just the best.
3. Pick the final configuration using dev.
4. Run once on test and report that.
5. Do a failure analysis: read every test miss and tag the cause (bad OCR, bad chunk boundary, wrong filter, vocabulary mismatch, ambiguous question).

### Results table to put in the README

| Config | hit@1 | hit@3 | hit@5 | MRR | Refusal acc. | p50 latency |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline |  |  |  |  |  |  |
| + chunk 300 / overlap 80 |  |  |  |  |  |  |
| + MiniLM embeddings |  |  |  |  |  |  |
| + reranker |  |  |  |  |  |  |
| Final |  |  |  |  |  |  |

Integrity rules: never tune on the test split, never delete bad runs, and report the numbers you actually got.

## 14. Repository structure

```text
vtu-rag/
  README.md
  requirements.txt
  .env.example
  config.yaml
  .gitignore               # data/, .env, index/
  docs/
    PRD.md
    data_sources.md
    decisions.md           # why each choice was made
    architecture.png
  data/                    # not committed
    raw/  processed/  index/
  src/vtu_rag/
    config.py
    ingest/  extract.py  parse.py  chunk.py
    index/   embed.py  store.py
    graph/   state.py  nodes.py  build.py  prompts.py
    retrieval/  retriever.py  rerank.py
    app/     streamlit_app.py
    eval/    run_eval.py  metrics.py
  eval/
    eval_set.yaml
    results/               # metrics JSON per run
  tests/
    test_parse.py  test_chunk.py  test_filters.py  test_metrics.py
  .github/workflows/ci.yml
```

## 15. Milestones (about 7 days)

| Day | Work | Done when |
| --- | --- | --- |
| 1 | Repo, environment, choose subjects, collect PDFs, extraction prototype | Text extracted for every page, OCR rate logged |
| 2 | Structure parsing, chunking, metadata, index build | Index built, 10 manual spot-checks pass |
| 3 | Retriever, 50-question eval set, eval harness, baseline numbers | Baseline results table committed |
| 4 | LangGraph flow, prompts, citations, not-found handling | 10 sample questions answered with citations, 3 unanswerable refused |
| 5 | Streamlit UI, filters, PYQ mode | All user stories US-1 to US-5 work |
| 6 | Experiments, failure analysis, choose final config on dev, run test | Full results table filled |
| 7 | README, architecture diagram, demo GIF, tests, CI, deployment, resume numbers | Fresh clone runs in 3 commands |

## 16. Instructions for your AI coding agent

Paste sections 1 to 15 into the agent as context, then give it these rules:

- Work in the order of the milestones. After each day, run the tests and commit.
- Keep the graph to the nodes listed in section 9. Do not add frameworks, databases or agents that are not in this PRD.
- Pin dependency versions. LangChain and LangGraph APIs change often, so check the current documentation before writing code against them.
- Never hardcode keys or paths. Read secrets from .env and parameters from config.yaml.
- Write unit tests for parsing, chunking, filter extraction and metrics before moving on.
- Log every design decision and its reason in docs/decisions.md.
- Never touch the test split while tuning. Evaluation code must read the split field.

Understand-it-yourself checklist. Because an agent writes the code, you must be able to explain each of these in an interview:

- Why this chunk size, and what changed when you varied it.
- How embeddings, cosine similarity and top-k retrieval work.
- Why metadata filtering helps, and how FAISS filtering differs from Chroma.
- What each LangGraph node does and why there is only one retry.
- How hit@k and MRR are computed, and why the test split is kept separate.
- Three real failures from your failure analysis and what you would fix.

## 17. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Scanned PYQs give poor OCR | Log OCR rate, try higher DPI, drop or hand-fix the worst files, document this honestly. |
| Question numbering differs between papers | Write the parser with tests, spot-check by hand, fall back to page-level chunks for odd files. |
| LLM hallucination | Strict prompt, NOT\_FOUND rule, citation validation, refusal metric. |
| Free-tier rate limits | Cache LLM calls during evaluation, keep a local-model fallback. |
| Overfitting to the eval set | Dev/test split, one-factor-at-a-time changes, keep all runs. |
| FAISS has no native filtering | Over-retrieve then filter, or switch to Chroma. |
| Library API changes | Pin versions, read current docs. |
| Scope creep | Hold to one branch and 3 to 5 subjects until the MVP is done. |

## 18. Legal, ethics and privacy

- Documents are used from public sources. Do not redistribute the PDFs in the repo.
- The app states that it is unofficial and that students must verify against the official VTU documents.
- No user data is stored. Logs hold only query text, filters and retrieval IDs. Add a note in the README.
- Do not present generated answers to exam questions as official solutions.

## 19. Portfolio and resume deliverables

- README: problem, demo GIF, architecture diagram, setup in 3 commands, results table, failure analysis, limitations.
- A public demo link if deployment works, otherwise a short screen recording.
- Repo link added to the resume project entry.

Resume bullet template (fill with real numbers only):

- Built a RAG assistant with LangChain and LangGraph over VTU syllabi and past papers, returning answers with subject, module and page citations.
- Evaluated on a 30-question held-out test set; improved hit@5 from \<baseline> to \<final> by tuning chunking and adding a reranker.

## 20. Definition of done

- [ ] Fresh clone runs the app in 3 commands or fewer.
- [ ] Index built from at least 3 subjects with 3 or more years of PYQs each.
- [ ] 50-question eval set committed with dev and test splits.
- [ ] Results table filled with real numbers for baseline and the final configuration.
- [ ] Unanswerable questions are refused and every answer shows citations.
- [ ] Tests and CI pass.
- [ ] README, architecture diagram and demo are complete.
- [ ] Resume project entry updated with the repo link and true numbers.

## 21. Open decisions for you

- Which branch, semester and subjects to index first.
- LLM provider: hosted free tier or local model through Ollama.
- Vector store: FAISS (default) or Chroma.
- Deployment target: Streamlit Community Cloud, Hugging Face Spaces, or local demo only.

## Appendix A. Sample queries to test

- What topics are covered in Module 2 of \<subject>?
- Show previous-year questions on \<topic> for \<subject>.
- How many marks did \<topic> carry in recent papers?
- Which module should I study for \<topic>?
- (Unanswerable) What is the syllabus of \<subject not indexed>?
- (Unanswerable) Give me the answer key for the Dec 2024 paper.
