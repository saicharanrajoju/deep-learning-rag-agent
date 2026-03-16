"""
test_vectorstore.py
===================
Integration tests for VectorStoreManager.

Run with: uv run pytest tests/ -v
"""

from __future__ import annotations

import pytest
from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager
from rag_agent.config import Settings

@pytest.fixture
def store(tmp_path):
    settings = Settings(
        CHROMA_DB_PATH=str(tmp_path / "test_chroma"),
        CHROMA_COLLECTION_NAME="test_collection",
        SIMILARITY_THRESHOLD=0.4
    )
    return VectorStoreManager(settings=settings)

@pytest.fixture
def sample_chunk():
    metadata = ChunkMetadata(
        topic="ANN",
        difficulty="intermediate",
        type="concept_explanation",
        source="ann_intermediate.md",
        related_topics=["backpropagation"],
        is_bonus=False,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("ann_intermediate.md", "Backpropagation is the algorithm..."),
        chunk_text=(
            "Backpropagation is the algorithm used to train neural networks by "
            "computing the gradient of the loss function with respect to each weight using "
            "the chain rule of calculus. The algorithm works in two passes: a forward pass "
            "to compute predictions and loss, and a backward pass to propagate gradients "
            "from the output layer to the input layer."
        ),
        metadata=metadata,
    )

def test_ingest_single_chunk(store, sample_chunk):
    result = store.ingest([sample_chunk])
    assert result.ingested == 1
    assert result.skipped == 0
    assert store.check_duplicate(sample_chunk.chunk_id) is True

def test_duplicate_detection(store, sample_chunk):
    result1 = store.ingest([sample_chunk])
    assert result1.ingested == 1
    assert result1.skipped == 0
    
    result2 = store.ingest([sample_chunk])
    assert result2.ingested == 0
    assert result2.skipped == 1
    
    stats = store.get_collection_stats()
    assert stats["total_chunks"] == 1

def test_retrieval_returns_citation(store, sample_chunk):
    store.ingest([sample_chunk])
    results = store.query("what is backpropagation")
    assert len(results) >= 1
    assert results[0].score > 0.0
    assert results[0].metadata.source == "ann_intermediate.md"
    assert "ann_intermediate.md" in results[0].to_citation()

def test_hallucination_guard(store):
    results = store.query("capital of france geography")
    assert results == []

def test_generate_chunk_id_deterministic():
    id1 = VectorStoreManager.generate_chunk_id("file.md", "some text")
    id2 = VectorStoreManager.generate_chunk_id("file.md", "some text")
    assert id1 == id2
    assert len(id1) == 16
    
    id3 = VectorStoreManager.generate_chunk_id("file.md", "different text")
    assert id1 != id3
