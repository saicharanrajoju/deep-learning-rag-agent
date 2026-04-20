"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        import chromadb
        db_path = Path(self._settings.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"ChromaDB initialised: '{self._settings.chroma_collection_name}' "
            f"with {self._collection.count()} items"
        )
        self._build_bm25()

    def _build_bm25(self) -> None:
        """
        Build an in-memory BM25 index of all current chunk documents.
        Used for the sparse keyword half of our Hybrid Search strategy.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return

        all_data = self._collection.get(include=["documents", "metadatas"])
        self._bm25_docs = all_data.get("documents", [])
        self._bm25_ids = all_data.get("ids", [])
        self._bm25_metadata = all_data.get("metadatas", [])
        
        if self._bm25_docs:
            tokenized_corpus = [doc.lower().split() for doc in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        return len(result["ids"]) > 0

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    logger.debug(f"Duplicate skipped: {chunk.chunk_id}")
                    continue
                embedding = self._embeddings.embed_documents([chunk.chunk_text])[0]
                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()]
                )
                result.ingested += 1
                if chunk.metadata.source not in result.document_ids:
                    result.document_ids.append(chunk.metadata.source)
            except Exception as e:
                result.errors.append(f"Chunk {chunk.chunk_id}: {e}")
                logger.error(f"Ingest failed for {chunk.chunk_id}: {e}")
        logger.info(
            f"Ingestion done: {result.ingested} ingested, "
            f"{result.skipped} skipped, {len(result.errors)} errors"
        )
        if result.ingested > 0 or result.deleted > 0 if hasattr(result, 'deleted') else False:
            self._build_bm25()
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k
        where_filter = None
        if topic_filter and difficulty_filter:
            where_filter = {"$and": [{"topic": topic_filter},
                                     {"difficulty": difficulty_filter}]}
        elif topic_filter:
            where_filter = {"topic": topic_filter}
        elif difficulty_filter:
            where_filter = {"difficulty": difficulty_filter}
        query_embedding = self._embeddings.embed_query(query_text)
        
        # 1. Dense Retrieval (ChromaDB)
        dense_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        candidate_pool = {}  # chunk_id -> chunk dict
        
        if dense_results and dense_results["ids"] and dense_results["ids"][0]:
            for i, chunk_id in enumerate(dense_results["ids"][0]):
                score = 1 - dense_results["distances"][0][i]
                metadata = ChunkMetadata.from_dict(dense_results["metadatas"][0][i])
                candidate_pool[chunk_id] = {
                    "id": chunk_id,
                    "text": dense_results["documents"][0][i],
                    "metadata": metadata,
                    "dense_score": score
                }

        # 2. Sparse Retrieval (BM25)
        if hasattr(self, "_bm25") and self._bm25 is not None and query_text:
            tokenized_query = query_text.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            import numpy as np
            # Get top k indices
            top_k_indices = np.argsort(bm25_scores)[-k:][::-1]
            for idx in top_k_indices:
                score = bm25_scores[idx]
                if score > 0:
                    chunk_id = self._bm25_ids[idx]
                    metadata = ChunkMetadata.from_dict(self._bm25_metadata[idx])
                    
                    # Apply where_filter locally
                    if topic_filter and metadata.topic != topic_filter:
                        continue
                    if difficulty_filter and metadata.difficulty != difficulty_filter:
                        continue
                        
                    if chunk_id not in candidate_pool:
                        candidate_pool[chunk_id] = {
                            "id": chunk_id,
                            "text": self._bm25_docs[idx],
                            "metadata": metadata,
                            "dense_score": 0.0
                        }

        if not candidate_pool:
            return []

        # 3. Cross-Encoder Re-Ranking (FlashRank)
        candidates_for_rerank = [{"id": v["id"], "text": v["text"]} for v in candidate_pool.values()]
        
        try:
            from flashrank import Ranker, RerankRequest
            ranker = Ranker()
            rerank_request = RerankRequest(query=query_text, passages=candidates_for_rerank)
            reranked_results = ranker.rerank(rerank_request)
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to dense scores.")
            reranked_results = sorted(
                [{"id": v["id"], "score": v["dense_score"]} for v in candidate_pool.values()],
                key=lambda x: x["score"],
                reverse=True
            )

        # 4. Final selection and formatting
        final_chunks = []
        # Return top K from reranked
        top_k = self._settings.rerank_top_k
        for res in reranked_results[:top_k]:
            chunk_id = str(res["id"])
            score = float(res.get("score", 0.0))
            if score < self._settings.similarity_threshold:
                continue
            
            chunk_data = candidate_pool[chunk_id]
            final_chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                chunk_text=chunk_data["text"],
                metadata=chunk_data["metadata"],
                score=score,
            ))
            
        return final_chunks

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        all_data = self._collection.get(include=["metadatas"])
        docs = {}
        for meta in all_data["metadatas"]:
            source = meta.get("source", "unknown")
            if source not in docs:
                docs[source] = {"source": source, "topic": meta.get("topic", ""), "chunk_count": 0}
            docs[source]["chunk_count"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        result = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )
        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            metadata = ChunkMetadata.from_dict(result["metadatas"][i])
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=result["documents"][i],
                metadata=metadata,
            ))
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        all_data = self._collection.get(include=["metadatas"])
        topics = list({m.get("topic", "") for m in all_data["metadatas"]})
        sources = list({m.get("source", "") for m in all_data["metadatas"]})
        bonus = any(m.get("is_bonus", "false") == "true" for m in all_data["metadatas"])
        return {
            "total_chunks": len(all_data["ids"]),
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        before = self._collection.count()
        self._collection.delete(where={"source": source})
        after = self._collection.count()
        deleted = before - after
        logger.info(f"Deleted {deleted} chunks for source: {source}")
        if deleted > 0:
            self._build_bm25()
        return deleted
