"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
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
                status_text.text(f"Processing file {i+1} of {total_files}: {fp.name}")
                try:
                    all_chunks.extend(chunker.chunk_file(fp, {}))
                except Exception as e:
                    st.sidebar.error(f"Failed to chunk {fp.name}: {e}")
                progress_bar.progress((i + 1) / total_files)
            
            status_text.empty()
            progress_bar.empty()

            with st.sidebar.spinner("Embedding and updating vector store..."):
                result = store.ingest(all_chunks)
                st.session_state["last_ingestion_result"] = result
                st.session_state["ingested_documents"] = store.list_documents()

        if result.errors:
            for error in result.errors:
                st.sidebar.error(error)
        else:
            if result.ingested > 0:
                msg = f"✅ {result.ingested} chunks ingested from {total_files} documents"
                if result.skipped > 0:
                    msg += f"\n\n⏭️ {result.skipped} duplicates skipped"
                st.sidebar.success(msg)
            elif result.skipped > 0:
                st.sidebar.warning(f"⏭️ {result.skipped} duplicates skipped")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Ingested Documents")
    docs = st.session_state.get("ingested_documents") or store.list_documents()
    if not docs:
        st.sidebar.caption("No documents ingested yet.")
    for doc in docs:
        st.sidebar.caption(f"**{doc['source']}** — {doc['topic']} ({doc['chunk_count']} chunks)")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.write(f"Topics: {', '.join(stats['topics'])}")
    if stats.get("bonus_topics_present"):
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    docs = st.session_state.get("ingested_documents") or store.list_documents()
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return
    sources = [d["source"] for d in docs]
    selected = st.selectbox("Select document", options=sources, key="doc_selector")
    st.session_state["selected_document"] = selected
    if selected:
        chunks = store.get_document_chunks(selected)
        st.caption(f"{len(chunks)} chunks from **{selected}**")
        with st.container(height=500):
            for chunk in chunks:
                with st.expander(f"[{chunk.metadata.topic} | {chunk.metadata.difficulty} | {chunk.metadata.type}]"):
                    st.write(chunk.chunk_text)


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        st.session_state["topic_filter"] = st.selectbox(
            "Topic filter", ["(all)", "ANN", "CNN", "RNN", "LSTM", "Seq2Seq", "Autoencoder"],
            key="topic_sel"
        ) or None
    with col_diff:
        st.session_state["difficulty_filter"] = st.selectbox(
            "Difficulty", ["(all)", "beginner", "intermediate", "advanced"],
            key="diff_sel"
        ) or None

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    if message.get("no_context_found"):
                        st.warning("⚠️ No relevant content found in the corpus. Try rephrasing with a specific deep learning term.")
                    
                    if message.get("sources"):
                        for source in message["sources"]:
                            st.caption(source)
                    
                    if message.get("confidence") is not None:
                        st.caption(f"Confidence: {message['confidence']:.2f}")
                    
                    if message.get("rewritten_query") and message["rewritten_query"] != message.get("original_query"):
                        with st.expander("🔍 Query rewritten to"):
                            st.write(message["rewritten_query"])

    query = st.chat_input("Ask about a deep learning topic...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)

        topic = st.session_state.get("topic_filter")
        diff = st.session_state.get("difficulty_filter")
        topic = None if topic == "(all)" else topic
        diff = None if diff == "(all)" else diff
        with st.spinner("Thinking..."):
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
                        if response.no_context_found:
                            st.warning("⚠️ No relevant content found in the corpus. Try rephrasing with a specific deep learning term.")
                        if response.sources:
                            for source in response.sources:
                                st.caption(source)
                        if response.confidence is not None:
                            st.caption(f"Confidence: {response.confidence:.2f}")
                        if response.rewritten_query and response.rewritten_query != query:
                            with st.expander("🔍 Query rewritten to"):
                                st.write(response.rewritten_query)

        if response:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "confidence": response.confidence,
                "no_context_found": response.no_context_found,
                "rewritten_query": response.rewritten_query,
                "original_query": query,
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I was unable to process your query. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "no_context_found": True,
                "rewritten_query": "",
                "original_query": query,
            })
        st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
