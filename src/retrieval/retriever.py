from typing import List, Dict
from qdrant_client import QdrantClient
from src.ingestion.embedder import embed_query
from src.core.config import settings
from src.core.logger import logger


async def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """
    Embed query and retrieve top-k similar chunks from Qdrant.
    Returns list of {text, source, chunk_id, score}
    Filters out results below similarity_threshold.
    """
    top_k = top_k or settings.top_k
    query_vector = await embed_query(query)

    client = QdrantClient(url=settings.qdrant_url)
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=settings.similarity_threshold,
    )

    chunks = []
    for r in results:
        chunks.append({
            "text": r.payload["text"],
            "source": r.payload["source"],
            "chunk_id": r.payload["chunk_id"],
            "chunk_index": r.payload.get("chunk_index", 0),
            "score": r.score,
        })

    logger.info("retrieved", query=query[:60], results=len(chunks))
    return chunks
