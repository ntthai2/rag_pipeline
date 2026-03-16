from fastapi import APIRouter, HTTPException
from src.api.schemas.models import IngestResponse
from src.ingestion.document_loader import load_documents
from src.ingestion.splitter import chunk_documents
from src.ingestion.indexer import index_chunks
from src.core.logger import logger

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(data_dir: str = "data/raw"):
    try:
        docs = load_documents(data_dir)
        if not docs:
            raise HTTPException(status_code=400, detail=f"No PDFs found in {data_dir}")

        chunks = chunk_documents(docs)
        total_indexed = await index_chunks(chunks)

        logger.info("ingest_complete", docs=len(docs), chunks=total_indexed)
        return IngestResponse(
            status="success",
            chunks_indexed=total_indexed,
            documents_loaded=len(docs),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ingest_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
