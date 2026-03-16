from fastapi import APIRouter, HTTPException
from src.api.schemas.models import QueryRequest, QueryResponse
from src.retrieval.retriever import retrieve
from src.generation.response_builder import build_and_generate
from src.core.config import settings
from src.core.logger import logger

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        top_k = request.top_k or settings.top_k
        chunks = await retrieve(request.question, top_k=top_k)

        # Use only top_k_rerank chunks for generation
        top_chunks = chunks[:settings.top_k_rerank]

        result = await build_and_generate(request.question, top_chunks)
        return QueryResponse(**result)

    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
