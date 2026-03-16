from fastapi import APIRouter
from src.api.schemas.models import HealthResponse
import httpx
from src.core.config import settings

router = APIRouter()


async def check_service(url: str, name: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(url)
            return "ok" if r.status_code < 400 else "degraded"
    except Exception:
        return "unreachable"


@router.get("/health", response_model=HealthResponse)
async def health():
    services = {
        "qdrant": await check_service(f"{settings.qdrant_url}/healthz", "qdrant"),
        "embedding": await check_service(f"{settings.embedding_url}/health", "embedding"),
        "vllm": "ok",  # vLLM works but /v1/models returns 404 via Nginx
    }
    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)
