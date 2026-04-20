"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend:
  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Design System — injected CSS
# ---------------------------------------------------------------------------

DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e0e0e !important;
}

.stApp {
    background-color: #0e0e0e;
    color: #e8e2d9;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    line-height: 1.6;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #141414; }
::-webkit-scrollbar-thumb { background: #2e2e2e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3e3e3e; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #1e1e1e !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}

/* sidebar headers */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6b6560 !important;
    margin-bottom: 0.75rem !important;
    margin-top: 1.5rem !important;
}

/* sidebar captions */
[data-testid="stSidebar"] .stCaption {
    color: #7a746d !important;
    font-size: 12px !important;
}

/* sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #1e1e1e !important;
    margin: 1.25rem 0 !important;
}

/* ── Main content area ────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    max-width: 100% !important;
}

/* ── Typography ───────────────────────────────────────────────────────── */
h1 {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #e8e2d9 !important;
    letter-spacing: -0.02em;
}
h2 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #d4cec6 !important;
    letter-spacing: -0.01em;
}
h3 { font-size: 13px !important; font-weight: 500 !important; color: #c4beb6 !important; }
p, li { color: #c8c2b9 !important; }

.stCaption p, .stCaption {
    color: #6b6560 !important;
    font-size: 12px !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    background-color: #c9a96e !important;
    color: #0e0e0e !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
    font-family: inherit !important;
}
.stButton > button:hover {
    background-color: #d4b47a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(201, 169, 110, 0.25) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background-color: #252525 !important;
    color: #444 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background-color: #141414 !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 8px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #c9a96e !important;
}
[data-testid="stFileUploaderDropzone"] p {
    color: #6b6560 !important;
    font-size: 12px !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background-color: #1e1e1e !important;
    border-radius: 3px;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #c9a96e, #d4b47a) !important;
    border-radius: 3px;
}

/* ── Select boxes ─────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 6px !important;
    color: #c8c2b9 !important;
    font-size: 13px !important;
    transition: border-color 0.15s;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #333 !important;
}
[data-testid="stSelectbox"] label {
    color: #6b6560 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Chat input ───────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background-color: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #c9a96e !important;
    box-shadow: 0 0 0 3px rgba(201, 169, 110, 0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #e8e2d9 !important;
    font-family: inherit !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4a4540 !important; }
[data-testid="stChatInputSubmitButton"] button {
    background-color: #c9a96e !important;
    border-radius: 7px !important;
}

/* ── Chat messages ────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #141414 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    border: 1px solid #1e1e1e !important;
}

/* avatar */
[data-testid="chatAvatarIcon-user"] {
    background-color: #c9a96e !important;
    color: #0e0e0e !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background-color: #1e1e1e !important;
    color: #c9a96e !important;
    border: 1px solid #2a2a2a !important;
}

/* ── Expanders ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #141414 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #8a8480 !important;
    font-weight: 500 !important;
    padding: 0.6rem 0.75rem !important;
}
[data-testid="stExpander"] summary:hover { color: #c8c2b9 !important; }
[data-testid="stExpander"] > div > div {
    padding: 0.5rem 0.75rem 0.75rem !important;
}

/* ── Alerts ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    border-left-width: 3px !important;
}
.stSuccess { background-color: #0d1f16 !important; border-color: #2e6b49 !important; color: #7bcba0 !important; }
.stWarning { background-color: #1f1a0d !important; border-color: #6b4e2e !important; color: #c9a96e !important; }
.stError   { background-color: #1f0d0d !important; border-color: #6b2e2e !important; color: #c97070 !important; }
.stInfo    { background-color: #0d141f !important; border-color: #2e4e6b !important; color: #70a0c9 !important; }

/* ── Metrics ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricValue"] {
    color: #c9a96e !important;
    font-size: 24px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #6b6560 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: #c9a96e !important; }

/* ── Scrollable containers ────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    background-color: #0c0c0c !important;
}

/* ── Column gap ───────────────────────────────────────────────────────── */
[data-testid="column"] {
    padding: 0 0.5rem !important;
}

/* ── App header ───────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.25rem;
}
.app-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #c9a96e 0%, #a07840 100%);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.app-title {
    font-size: 18px;
    font-weight: 700;
    color: #e8e2d9;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 12px;
    color: #5a5550;
    margin: 0;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 1rem;
}

/* ── Metadata tags ────────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 4px;
}
.tag-topic  { background: #1a1a2e; color: #7090c9; border: 1px solid #2a2a4e; }
.tag-diff-beginner     { background: #0d1f0f; color: #6bb56d; border: 1px solid #1e3a1e; }
.tag-diff-intermediate { background: #1f1a0d; color: #c9a96e; border: 1px solid #3a2e0d; }
.tag-diff-advanced     { background: #1f0d15; color: #c97095; border: 1px solid #3a0d1e; }
.tag-type   { background: #151515; color: #8a8480; border: 1px solid #252525; }

/* ── Source citation ──────────────────────────────────────────────────── */
.source-cite {
    font-size: 11px;
    color: #5a5550;
    padding: 2px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.source-cite::before {
    content: "›";
    color: #c9a96e;
    font-weight: 600;
}

/* ── Confidence bar ───────────────────────────────────────────────────── */
.confidence-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
}
.confidence-label { font-size: 11px; color: #5a5550; white-space: nowrap; }
.confidence-bar {
    flex: 1;
    height: 3px;
    background: #1e1e1e;
    border-radius: 2px;
    overflow: hidden;
    max-width: 80px;
}
.confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #6b6b6b, #c9a96e);
    transition: width 0.4s ease;
}

/* ── Doc card ─────────────────────────────────────────────────────────── */
.doc-card {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 7px;
    padding: 8px 10px;
    margin-bottom: 6px;
    cursor: default;
    transition: border-color 0.15s;
}
.doc-card:hover { border-color: #2a2a2a; }
.doc-name { font-size: 12px; font-weight: 500; color: #c8c2b9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { font-size: 11px; color: #5a5550; margin-top: 2px; }

/* ── Section label ────────────────────────────────────────────────────── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a4540;
    margin-bottom: 8px;
    margin-top: 16px;
}

/* ── Empty state ──────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #3a3530;
}
.empty-state-icon { font-size: 28px; margin-bottom: 0.5rem; }
.empty-state-text { font-size: 13px; color: #4a4540; }

/* ── Panel subheader ──────────────────────────────────────────────────── */
.panel-header {
    font-size: 13px;
    font-weight: 600;
    color: #c4beb6;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1a1a1a;
}

/* ── Hallucination guard banner ───────────────────────────────────────── */
.hg-banner {
    background: #1a1209;
    border: 1px solid #3a2e12;
    border-radius: 7px;
    padding: 8px 12px;
    font-size: 12px;
    color: #a08040;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-top: 6px;
}
.hg-icon { flex-shrink: 0; margin-top: 1px; }
</style>
"""


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diff_tag(difficulty: str) -> str:
    cls = {
        "beginner": "tag-diff-beginner",
        "intermediate": "tag-diff-intermediate",
        "advanced": "tag-diff-advanced",
    }.get((difficulty or "").lower(), "tag-type")
    return f'<span class="tag {cls}">{difficulty or "?"}</span>'


def _topic_tag(topic: str) -> str:
    return f'<span class="tag tag-topic">{topic or "?"}</span>'


def _type_tag(t: str) -> str:
    return f'<span class="tag tag-type">{t or "?"}</span>'


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------

def render_ingestion_panel(store: VectorStoreManager, chunker: DocumentChunker) -> None:
    st.sidebar.markdown(
        '<div class="section-label">Corpus Ingestion</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_paths = []
            for uf in uploaded_files:
                dest = Path(tmp_dir) / uf.name
                dest.write_bytes(uf.read())
                file_paths.append(dest)

            all_chunks = []
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            total_files = len(file_paths)

            for i, fp in enumerate(file_paths):
                status_text.markdown(
                    f'<p style="font-size:11px;color:#6b6560;margin:0">Processing {fp.name}…</p>',
                    unsafe_allow_html=True,
                )
                try:
                    all_chunks.extend(chunker.chunk_file(fp, {}))
                except Exception as e:
                    st.sidebar.error(f"Failed to chunk {fp.name}: {e}")
                progress_bar.progress((i + 1) / total_files)

            status_text.empty()
            progress_bar.empty()

            with st.sidebar.spinner("Embedding…"):
                result = store.ingest(all_chunks)
                st.session_state["last_ingestion_result"] = result
                st.session_state["ingested_documents"] = store.list_documents()

        if result.errors:
            for error in result.errors:
                st.sidebar.error(error)
        elif result.ingested > 0:
            msg = f"**{result.ingested}** chunks added from {total_files} file{'s' if total_files > 1 else ''}"
            if result.skipped > 0:
                msg += f" · {result.skipped} duplicates skipped"
            st.sidebar.success(msg)
        elif result.skipped > 0:
            st.sidebar.warning(f"{result.skipped} duplicates skipped — already ingested")

    st.sidebar.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr style="border-color:#1a1a1a;margin:0">', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="section-label" style="margin-top:1rem">Ingested Documents</div>',
        unsafe_allow_html=True,
    )

    docs = st.session_state.get("ingested_documents") or store.list_documents()
    if not docs:
        st.sidebar.markdown(
            '<div class="empty-state" style="padding:1.5rem 0.5rem">'
            '<div class="empty-state-icon">📂</div>'
            '<div class="empty-state-text">No documents yet.<br>Upload PDFs or Markdown files above.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for doc in docs:
            st.sidebar.markdown(
                f'<div class="doc-card">'
                f'<div class="doc-name">{doc["source"]}</div>'
                f'<div class="doc-meta">{doc["topic"]} · {doc["chunk_count"]} chunks</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_corpus_stats(store: VectorStoreManager) -> None:
    stats = store.get_collection_stats()
    st.sidebar.markdown('<hr style="border-color:#1a1a1a;margin:0.5rem 0">', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="section-label">Corpus Stats</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    if stats.get("topics"):
        topics_html = "".join(_topic_tag(t) for t in stats["topics"])
        st.sidebar.markdown(topics_html, unsafe_allow_html=True)
    if stats.get("bonus_topics_present"):
        st.sidebar.success("Bonus topics present")
    else:
        st.sidebar.warning("No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------

def render_document_viewer(store: VectorStoreManager) -> None:
    st.markdown(
        '<div class="panel-header">📄 Document Viewer</div>',
        unsafe_allow_html=True,
    )

    docs = st.session_state.get("ingested_documents") or store.list_documents()
    if not docs:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">📑</div>'
            '<div class="empty-state-text">Ingest documents from the sidebar<br>to browse their contents here.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    sources = [d["source"] for d in docs]
    selected = st.selectbox(
        "Document",
        options=sources,
        key="doc_selector",
        label_visibility="collapsed",
    )
    st.session_state["selected_document"] = selected

    if selected:
        chunks = store.get_document_chunks(selected)
        st.markdown(
            f'<div style="font-size:11px;color:#4a4540;margin-bottom:0.75rem;margin-top:0.25rem">'
            f"{len(chunks)} chunks · <span style='color:#7a7470'>{selected}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.container(height=520):
            for chunk in chunks:
                m = chunk.metadata
                label_html = (
                    _topic_tag(m.topic)
                    + _diff_tag(m.difficulty)
                    + _type_tag(m.type)
                )
                with st.expander(f"{m.topic}  ·  {m.difficulty}  ·  {m.type}"):
                    st.markdown(label_html, unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="font-size:13px;color:#c8c2b9;margin-top:0.5rem;line-height:1.65">'
                        f"{chunk.chunk_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------

def _render_message_meta(message: dict) -> None:
    """Render source citations, confidence, and query info for an assistant message."""
    if message.get("no_context_found"):
        st.markdown(
            '<div class="hg-banner">'
            '<span class="hg-icon">⚠</span>'
            "No relevant content found in the corpus. "
            "Try rephrasing with a specific deep learning term."
            "</div>",
            unsafe_allow_html=True,
        )

    if message.get("sources"):
        sources_html = "".join(
            f'<div class="source-cite">{s}</div>' for s in message["sources"]
        )
        st.markdown(
            f'<div style="margin-top:8px">{sources_html}</div>',
            unsafe_allow_html=True,
        )

    meta_parts = []
    if message.get("confidence") is not None:
        conf = message["confidence"]
        fill_pct = int(conf * 100)
        meta_parts.append(
            f'<div class="confidence-row">'
            f'<span class="confidence-label">Confidence</span>'
            f'<div class="confidence-bar"><div class="confidence-fill" style="width:{fill_pct}%"></div></div>'
            f'<span class="confidence-label">{conf:.0%}</span>'
            f"</div>"
        )
    if message.get("retries", 0) > 0:
        meta_parts.append(
            f'<span style="font-size:11px;color:#4a4540">↻ {message["retries"]} retrieval retries</span>'
        )

    if meta_parts:
        st.markdown(
            '<div style="margin-top:6px;display:flex;flex-direction:column;gap:4px">'
            + "".join(meta_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    rq = message.get("rewritten_query")
    oq = message.get("original_query")
    if rq and rq != oq:
        with st.expander("Query rewritten to"):
            st.markdown(
                f'<span style="font-size:13px;color:#8a8480;font-style:italic">"{rq}"</span>',
                unsafe_allow_html=True,
            )


def render_chat_interface(graph) -> None:
    st.markdown(
        '<div class="panel-header">💬 Interview Prep Chat</div>',
        unsafe_allow_html=True,
    )

    # Compact filter row
    col_topic, col_diff = st.columns(2)
    with col_topic:
        raw_topic = st.selectbox(
            "Topic",
            ["(all)", "ANN", "CNN", "RNN", "LSTM", "Seq2Seq", "Autoencoder"],
            key="topic_sel",
        )
        st.session_state["topic_filter"] = None if raw_topic == "(all)" else raw_topic
    with col_diff:
        raw_diff = st.selectbox(
            "Difficulty",
            ["(all)", "beginner", "intermediate", "advanced"],
            key="diff_sel",
        )
        st.session_state["difficulty_filter"] = None if raw_diff == "(all)" else raw_diff

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # Chat history
    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="empty-state" style="padding:3rem 1rem">'
                '<div class="empty-state-icon">🧠</div>'
                '<div class="empty-state-text">'
                "Ask a question about deep learning concepts.<br>"
                '<span style="color:#3a3530">Questions are answered from your ingested documents only.</span>'
                "</div></div>",
                unsafe_allow_html=True,
            )
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    _render_message_meta(message)

    # Input
    query = st.chat_input("Ask about a deep learning topic…")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        topic = st.session_state.get("topic_filter")
        diff = st.session_state.get("difficulty_filter")

        with st.spinner("Thinking…"):
            from langchain_core.messages import HumanMessage as HM

            graph_input = {
                "messages": [HM(content=query)],
                "topic_filter": topic,
                "difficulty_filter": diff,
            }
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            response = None

            with chat_container:
                with st.chat_message("assistant"):
                    try:
                        def generate_stream():
                            nonlocal response
                            for event in graph.stream(graph_input, config=config):
                                for node_name, state in event.items():
                                    if state and "final_response" in state:
                                        response = state["final_response"]
                                        if response and response.answer:
                                            yield response.answer

                        st.write_stream(generate_stream)
                    except Exception:
                        try:
                            result = graph.invoke(graph_input, config=config)
                            response = result.get("final_response")
                            if response:
                                st.markdown(response.answer)
                        except Exception as e:
                            st.error(f"Something went wrong: {e}")

                    if response:
                        _render_message_meta(
                            {
                                "no_context_found": response.no_context_found,
                                "sources": response.sources,
                                "confidence": response.confidence,
                                "rewritten_query": response.rewritten_query,
                                "original_query": query,
                                "retries": getattr(response, "retries", 0),
                            }
                        )

        if response:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "no_context_found": response.no_context_found,
                    "rewritten_query": response.rewritten_query,
                    "original_query": query,
                    "retries": getattr(response, "retries", 0),
                }
            )
        else:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "I was unable to process your query. Please try again.",
                    "sources": [],
                    "confidence": 0.0,
                    "no_context_found": True,
                    "rewritten_query": "",
                    "original_query": query,
                }
            )
        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject design system
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        f'<div class="app-header">'
        f'<div class="app-logo">🧠</div>'
        f'<h1 class="app-title">{settings.app_title}</h1>'
        f"</div>"
        f'<p class="app-subtitle">RAG-powered interview preparation · LangGraph · ChromaDB · Groq</p>',
        unsafe_allow_html=True,
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar brand mark
    st.sidebar.markdown(
        '<div style="padding:0 0 0.5rem 0;border-bottom:1px solid #1a1a1a;margin-bottom:0.25rem">'
        '<span style="font-size:11px;font-weight:700;letter-spacing:0.12em;color:#3a3530;text-transform:uppercase">Documents</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
