from typing import List
import httpx
from src.core.config import settings
from src.core.logger import logger


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using Jina v3 via Infinity server."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.embedding_url}/embeddings",
            json={"model": settings.embedding_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings


async def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    results = await embed_texts([query])
    return results[0]
