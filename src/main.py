from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.core.logger import setup_logging, logger
from src.api.routes import query, ingest, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("rag_api_starting")
    yield
    logger.info("rag_api_shutdown")


app = FastAPI(
    title="RAG Pipeline API",
    description="Week 2 Challenge 1 — RAG Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingestion"])
app.include_router(query.router, tags=["query"])
