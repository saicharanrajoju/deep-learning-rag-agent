import json
from pathlib import Path
from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager, VectorStoreManager as VSM

sample_path = Path("examples/sample_chunk.json")
data = json.loads(sample_path.read_text())

chunk = DocumentChunk(
    chunk_id=VSM.generate_chunk_id(data["metadata"]["source"], data["chunk_text"]),
    chunk_text=data["chunk_text"],
    metadata=ChunkMetadata(**{**data["metadata"],
                              "related_topics": data["metadata"]["related_topics"]})
)

store = VectorStoreManager()
result = store.ingest([chunk])
print(f"Run 1 — ingested: {result.ingested}, skipped: {result.skipped}")

result2 = store.ingest([chunk])
print(f"Run 2 — ingested: {result2.ingested}, skipped: {result2.skipped}")

hits = store.query("what is a neural network")
for h in hits:
    print(f"Score: {h.score:.3f} | {h.to_citation()}")
    print(h.chunk_text[:120], "...")
