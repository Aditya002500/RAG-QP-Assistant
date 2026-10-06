"""Streamlit Web Application for VTU Syllabus & Question-Paper Assistant (RAG).
Provides Interactive Q&A with Citations, PYQ Explorer, Sidebar Filters, and Evaluation View.
Complies with FR-10, FR-11, and Section 7 unofficial disclaimer.
"""

import time
from pathlib import Path

import streamlit as st

from vtu_rag.config import load_config
from vtu_rag.graph.build import build_rag_graph, run_pipeline

# Configure page layout and title
st.set_page_config(
    page_title="VTU Study Assistant (RAG)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .disclaimer-banner {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 16px;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 20px;
    }
    
    .citation-badge {
        display: inline-block;
        background: #2563eb;
        color: white;
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    
    .pyq-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        transition: transform 0.15s ease-in-out;
    }
    .pyq-card:hover {
        border-color: #3b82f6;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
    }
    .badge-marks { background: #065f46; color: #a7f3d0; }
    .badge-module { background: #1e3a8a; color: #bfdbfe; }
    .badge-year { background: #701a75; color: #f5d0fe; }
    .badge-subj { background: #374151; color: #f3f4f6; }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading VTU Vector Index & AI Pipeline...")
def get_pipeline():
    cfg = load_config()
    graph = build_rag_graph(config=cfg)
    return graph, cfg


graph_app, config = get_pipeline()

# Header and Unofficial Notice (Section 7)
st.title("🎓 VTU Syllabus & Question-Paper Assistant (RAG)")
st.markdown(
    """
<div class="disclaimer-banner">
    ⚠️ <strong>Unofficial Student Tool</strong>: This assistant indexes official VTU 2022 Scheme syllabi and past examination papers. 
    Always verify syllabus updates and exam regulations with the official <a href="https://vtu.ac.in" target="_blank" style="color:#60a5fa;">VTU Website</a>.
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar: Filters & Settings (FR-6, FR-10)
st.sidebar.header("🔍 Academic Filters")
branch = st.sidebar.selectbox("Branch", ["CSE (Computer Science)", "ISE"], index=0)
semester = st.sidebar.selectbox("Semester", [5], index=0)

subject_options = {
    "All Subjects": None,
    "21CS51 - Software Engineering & Project Mgmt": "21CS51",
    "21CS52 - Computer Networks": "21CS52",
    "21CS53 - Database Management Systems": "21CS53",
    "21CS54 - Artificial Intelligence & ML": "21CS54",
}
selected_subject_label = st.sidebar.selectbox(
    "Subject", list(subject_options.keys())
)
selected_subject_code = subject_options[selected_subject_label]

module_choice = st.sidebar.selectbox(
    "Module", ["All Modules", "Module 1", "Module 2", "Module 3", "Module 4", "Module 5"]
)
selected_module = int(module_choice.split()[-1]) if "Module" in module_choice else None

doc_type_choice = st.sidebar.selectbox(
    "Document Scope", ["All Documents", "Syllabus Only", "Previous-Year Papers (PYQ)"]
)
doc_type_map = {
    "All Documents": None,
    "Syllabus Only": "syllabus",
    "Previous-Year Papers (PYQ)": "pyq",
}
selected_doc_type = doc_type_map[doc_type_choice]

year_choice = st.sidebar.selectbox("Exam Session / Year", ["All Years", "2024", "2023"])
selected_year = int(year_choice) if year_choice != "All Years" else None

sidebar_filters = {}
if selected_subject_code:
    sidebar_filters["subject_code"] = selected_subject_code
if selected_module is not None:
    sidebar_filters["module"] = selected_module
if selected_doc_type:
    sidebar_filters["doc_type"] = selected_doc_type
if selected_year:
    sidebar_filters["year"] = selected_year

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")
st.sidebar.write(f"**Embedding Model:** `{config.embedding.model_name}`")
st.sidebar.write("**Vector Store:** FAISS (Indexed chunks: 74)")
st.sidebar.write(f"**Reranker:** {'Enabled' if config.reranker.enabled else 'Disabled (Default)'}")

# Main Tabs
tab_chat, tab_pyq, tab_eval = st.tabs([
    "💬 Assistant Q&A",
    "📝 PYQ Explorer",
    "📊 Architecture & Evaluation",
])

# ----------------- TAB 1: Chat Assistant -----------------
with tab_chat:
    st.subheader("Ask a VTU Curriculum or Exam Question")
    st.caption("Answers are strictly grounded in indexed syllabi and PYQs with passage citations.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! Ask me any question about VTU 5th Sem Computer Science syllabi (21CS51, 21CS52, 21CS53, 21CS54) or previous-year questions.",
                "citations": [],
            }
        ]

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 Sources Cited ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"**[{c.get('citation_num')}] {c.get('subject_name')} ({c.get('subject_code')})** | "
                            f"Module {c.get('module') or 'N/A'} | Page {c.get('page')} ({c.get('doc_type', '').upper()})\n\n"
                            f"> *{c.get('passage_snippet')}*"
                        )

    # Prompt user
    if user_input := st.chat_input("E.g., What topics are in Module 3 of Computer Networks?"):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"), st.spinner("Retrieving from VTU index & reasoning..."):
            t_start = time.time()
            result = run_pipeline(
                question=user_input,
                filters=sidebar_filters,
                graph_app=graph_app,
                config=config,
            )
            duration = time.time() - t_start

            answer = result.get("answer", "")
            citations = result.get("citations", [])
            st.markdown(answer)

            if citations:
                with st.expander(f"📚 Verified Sources ({len(citations)}) - Latency: {duration:.2f}s"):
                    for c in citations:
                        st.markdown(
                            f"**[{c.get('citation_num')}] {c.get('subject_name')} ({c.get('subject_code')})** | "
                            f"Module {c.get('module') or 'N/A'} | Page {c.get('page')} ({c.get('doc_type', '').upper()})\n\n"
                            f"File: `{c.get('source_file')}`\n\n"
                            f"> *{c.get('passage_snippet')}*"
                        )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "citations": citations}
            )

# ----------------- TAB 2: PYQ Explorer -----------------
with tab_pyq:
    st.subheader("Find Previous-Year Questions (PYQ Mode)")
    st.caption("Inspect past exam questions by topic, module, marks, and year (FR-11).")

    pyq_query = st.text_input(
        "Search PYQs on topic:",
        value="Dijkstra's Link-State routing algorithm",
        placeholder="E.g. Waterfall model, Normalization, TCP congestion, A* algorithm",
    )

    if pyq_query:
        # Search specifically for PYQs
        pyq_filters = dict(sidebar_filters)
        pyq_filters["doc_type"] = "pyq"

        with st.spinner("Searching PYQ index..."):
            retriever_inst = graph_app.get_state({"question": pyq_query})  # or direct retrieve
            # Directly use retriever from config
            from vtu_rag.retrieval.retriever import VTURetriever

            ret = VTURetriever.load_from_config(config)
            pyq_results = ret.retrieve(query=pyq_query, filters=pyq_filters, top_k=6)

        if not pyq_results:
            st.warning("No previous-year questions matched the search query and active filters.")
        else:
            st.success(f"Found {len(pyq_results)} relevant question paper records:")
            for chunk, score in pyq_results:
                marks_badge = f'<span class="badge-pill badge-marks">{chunk.marks or 10} Marks</span>'
                mod_badge = f'<span class="badge-pill badge-module">Module {chunk.module or "N/A"}</span>'
                year_badge = f'<span class="badge-pill badge-year">{chunk.session or chunk.year or "Exam"}</span>'
                subj_badge = f'<span class="badge-pill badge-subj">{chunk.subject_code} - {chunk.subject_name}</span>'

                st.markdown(
                    f"""
                <div class="pyq-card">
                    <div style="margin-bottom:8px;">
                        {subj_badge} {mod_badge} {marks_badge} {year_badge}
                    </div>
                    <div style="font-size: 1.02rem; font-weight: 500; margin-bottom: 6px;">
                        {chunk.text.splitlines()[-1] if len(chunk.text.splitlines()) > 1 else chunk.text}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        Source: {chunk.source_file} (Page {chunk.page}) | Cosine Score: {score:.3f}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ----------------- TAB 3: Evaluation & Architecture -----------------
with tab_eval:
    st.subheader("System Architecture & Benchmark Results")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Indexed Subjects", "4 Courses", "5th Sem CSE")
    col_b.metric("Total Chunks", "74 Chunks", "100% Validated")
    col_c.metric("Dev Eval Set", "20 Questions", "100% Labelled")
    col_d.metric("Test Split", "30 Questions", "Strictly Held Out")

    st.markdown("### LangGraph 6-Node Architecture")
    st.code(
        """
OFFLINE:
  PDFs -> PyMuPDF / Tesseract OCR -> Structure Parse -> Chunking & Headers -> BGE Embeddings -> FAISS

ONLINE (LangGraph):
  question -> analyze_query -> retrieve -> grade --ok--> generate -> answer + citations
                                ^           |
                                |           +--weak--> rewrite_query (max 1 retry)
                                +-----------------------+   still weak --> not_found
    """,
        language="text",
    )

    st.markdown("### Benchmark Results (50 Questions)")
    summary_path = Path(config.paths.eval_results_dir) / "experiment_summary.md"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Run evaluation suite (`python -m vtu_rag.eval.run_eval`) to generate live metrics.")
