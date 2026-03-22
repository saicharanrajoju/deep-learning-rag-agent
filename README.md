# Deep Learning RAG Interview Prep Agent

A RAG-powered interview preparation agent built with LangChain, LangGraph, and ChromaDB. Upload deep learning study material, chat with it to generate technical interview questions, and get evaluated on your answers.

---

## Team

| Name | Role | Owned |
|---|---|---|
| **Sai Charan Rajoju** | Pipeline Engineer | `config.py`, `store.py`, `nodes.py`, `graph.py` |
| **Snehal Teja Adidam** | Corpus Architect | `data/corpus/` — study material chunks |
| **Girivarshini** | UX Lead | `src/rag_agent/ui/app.py` — Streamlit interface |
| **Kishore** | Prompt Engineer | `src/rag_agent/agent/prompts.py` — LLM prompts |
| **Durgi Sai Jashwanth Kumar** | QA Lead | `tests/` — integration tests and demo script |

---

## What We Built

A fully working RAG agent that:

- Ingests deep learning study material (PDF and Markdown) into a ChromaDB vector store
- Detects and skips duplicate documents on re-upload using content hashing
- Rewrites natural language queries into keyword-dense search terms for better retrieval
- Retrieves the most relevant chunks using cosine similarity search
- Generates grounded answers with source citations using a LangGraph pipeline
- Fires a hallucination guard when no relevant context is found — refusing to answer from general knowledge
- Generates technical interview questions from ingested material
- Evaluates student answers with a score and structured feedback

---

## How Each Role Contributed

### Sai Charan Rajoju — Pipeline Engineer
Built the backbone of the system. Implemented the full ChromaDB integration in `store.py` including initialisation, content-hash-based duplicate detection, batch ingestion, cosine similarity retrieval with threshold filtering, and corpus inspection methods. Set up the `LLMFactory` and `EmbeddingFactory` in `config.py` for swappable LLM providers (Groq, Ollama, LM Studio) and local sentence-transformer embeddings. Wired together the three-node LangGraph pipeline — query rewriting, retrieval, and generation — in `nodes.py` and `graph.py` with a `MemorySaver` checkpointer for multi-turn conversation.

### Snehal Teja Adidam — Corpus Architect
Authored all study material in `data/corpus/`. Drafted intermediate-level content for ANN, CNN, and RNN covering forward propagation, backpropagation, activation functions, convolution operations, pooling, feature maps, hidden state, BPTT, and the vanishing gradient problem. Located and sourced landmark papers: Rumelhart et al. (1986) for backprop, LeCun et al. (1998) for LeNet, and Hochreiter & Schmidhuber (1997) for LSTM. Ensured every chunk follows the agreed metadata schema (`topic`, `difficulty`, `type`, `source`, `related_topics`, `is_bonus`) and is atomic enough to anchor a single interview question.

### Girivarshini — UX Lead
Built the three-panel Streamlit interface in `app.py`. Panel 1 handles multi-file upload with a per-file progress bar and structured ingestion result display (chunks added, duplicates skipped, errors). Panel 2 is a document viewer with a selectable dropdown and chunk-level expanders showing metadata. Panel 3 is the chat interface with topic and difficulty filters, inline source citations, confidence scores, a rewritten query expander, and a hallucination guard warning. Implemented streaming responses via `graph.stream` with a `graph.invoke` fallback. Wrapped all persistent resources in `@st.cache_resource` to prevent re-initialisation on every click.

### Kishore — Prompt Engineer
Designed and hardened all four prompts in `prompts.py`. The system prompt defines strict rules preventing the LLM from using general knowledge and enforces citation format `[topic | difficulty | source]`. The query rewrite prompt converts conversational queries into 10-word keyword-dense search strings while preserving the original intent. The question generation prompt produces open-ended interview questions with a validated JSON schema (`question`, `difficulty`, `model_answer`, `concepts_tested`). The answer evaluation prompt scores student answers 0–10 with strict rubric anchors and returns structured feedback (`score`, `feedback`, `missing_concepts`, `correct_concepts`). Documented failure modes and mitigations above every prompt constant.

### Durgi Sai Jashwanth Kumar — QA Lead
Wrote five integration tests in `test_vectorstore.py` covering the full pipeline: single chunk ingestion, duplicate detection (same chunk ingested twice skips on second run), retrieval with source citation, hallucination guard (off-topic query returns empty list), and deterministic chunk ID generation. All tests run against isolated ChromaDB instances using `tmp_path` fixtures — no shared state between tests. Wrote the 60-second demo script in `docs/architecture.md` covering all five demo beats in order.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Orchestration | LangChain + LangGraph |
| UI | Streamlit |
| Package Manager | UV |

---

## Getting Started

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure environment
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 3. Run the app
```bash
uv run streamlit run src/rag_agent/ui/app.py
```

### 4. Run the tests
```bash
uv run pytest tests/ -v
```

The app will be available at **http://localhost:8501**.

---

## LLM Provider Options

### Groq (Recommended — free, fast)
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```
Get a free API key at [console.groq.com](https://console.groq.com).

### Ollama (Local, no API key)
```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```
Start with `ollama serve` before running the app.

### LM Studio (Local GUI)
```
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
```

---

## Project Structure

```
deep-learning-rag-agent/
├── data/corpus/            ← Study material (.md and .pdf)
├── docs/
│   ├── checklist.md        ← Phase 1 & 2 build checklist
│   └── architecture.md     ← QA test results and demo script
├── examples/
│   └── sample_chunk.json   ← Canonical chunk schema reference
├── src/rag_agent/
│   ├── config.py           ← LLMFactory, EmbeddingFactory, Settings
│   ├── corpus/chunker.py   ← PDF and Markdown chunker
│   ├── vectorstore/store.py← ChromaDB integration
│   ├── agent/
│   │   ├── state.py        ← Data models
│   │   ├── prompts.py      ← All LLM prompt templates
│   │   ├── nodes.py        ← LangGraph node functions
│   │   └── graph.py        ← Graph assembly
│   └── ui/app.py           ← Streamlit application
├── tests/test_vectorstore.py
├── .env.example
└── pyproject.toml
```

---

## Common Issues

**`ModuleNotFoundError: No module named 'rag_agent'`**
Run from the project root using `uv run`, not `python` directly.

**`chromadb.errors.NotEnoughElementsException`**
Ingest more documents or reduce `RETRIEVAL_K` in `.env`.

**`ollama: connection refused`**
Start Ollama with `ollama serve` in a separate terminal.

**Streamlit loses state on every click**
All persistent objects are wrapped in `@st.cache_resource` — restart the app if state appears corrupted.
