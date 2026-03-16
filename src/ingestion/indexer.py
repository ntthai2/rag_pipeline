from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff
)
from src.ingestion.embedder import embed_texts
from src.core.config import settings
from src.core.logger import logger
import uuid


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        logger.info("collection_created", name=settings.qdrant_collection)
    else:
        logger.info("collection_exists", name=settings.qdrant_collection)


async def index_chunks(chunks: List[Dict], batch_size: int = 32):
    """Embed chunks in batches and store in Qdrant."""
    client = get_qdrant_client()
    ensure_collection(client)

    total = len(chunks)
    logger.info("indexing_start", total_chunks=total)

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        embeddings = await embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[j],
                payload={
                    "text": batch[j]["text"],
                    "source": batch[j]["source"],
                    "chunk_id": batch[j]["chunk_id"],
                    "chunk_index": batch[j]["chunk_index"],
                },
            )
            for j in range(len(batch))
        ]

        client.upsert(collection_name=settings.qdrant_collection, points=points)
        logger.info("indexed_batch", batch=f"{i}-{i+len(batch)}", total=total)

    logger.info("indexing_complete", total_indexed=total)
    return total
