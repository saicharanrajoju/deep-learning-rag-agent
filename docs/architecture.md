# Deep Learning RAG Agent Architecture & QA

## Section 1: Integration Test Results

| # | Test | Description | Expected | Status |
|---|------|-------------|----------|--------|
| 1 | test_same_content_produces_same_id | Same source+text always produces the same 16-char hash ID | id1 == id2 | ✅ PASS |
| 2 | test_ingestion_skips_duplicate | Ingesting same chunk twice → skipped=1 on second run | result.skipped == 1 | ✅ PASS |
| 3 | test_relevant_query_returns_results | Semantic query returns matching chunks above threshold | score > 0.3 | ✅ PASS |
| 4 | test_irrelevant_query_returns_empty | Off-topic query returns empty list (hallucination guard) | results == [] | ✅ PASS |
| 5 | test_topic_filter_restricts_results | topic_filter='LSTM' returns only LSTM chunks | all topics == LSTM | ✅ PASS |

All 5 tests pass. Run with: `uv run pytest tests/ -v`

---

## Section 2: 60-Second Demo Script

**Beat 1 — Upload documents (10s)**
- **Action:** Upload `ann_intermediate.md` and `cnn_intermediate.md` via the sidebar.
- **Point out:** Progress bar advances per file, final count shows "X chunks ingested from 2 documents."

**Beat 2 — Duplicate detection (10s)**
- **Action:** Upload `ann_intermediate.md` again and click Ingest.
- **Point out:** Message shows "X duplicates skipped, 0 new chunks" — a content hash ensures identical files are never stored twice.

**Beat 3 — Normal query with citation (15s)**
- **Action:** Type "Explain the vanishing gradient problem in ANNs."
- **Point out:** Response cites the source `[ANN | intermediate | ann_intermediate.md]`, confidence score is visible, and the rewritten query expander shows what the system exactly searched for.

**Beat 4 — Hallucination guard (10s)**
- **Action:** Type "What is the capital of France?"
- **Point out:** Yellow warning fires ("No relevant content found in the corpus") — the system refuses to answer from general knowledge.

**Beat 5 — Interview question generation (15s)**
- **Action:** Type "Generate an intermediate interview question about CNN pooling."
- **Point out:** Structured answer containing a question, model answer, and source citation — ready to use in a real interview.
