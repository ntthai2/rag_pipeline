from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    context_used: int


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    documents_loaded: int


class HealthResponse(BaseModel):
    status: str
    services: dict
