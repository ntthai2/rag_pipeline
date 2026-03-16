"""
CLI script to ingest documents.
Usage: python scripts/ingest.py [--data-dir data/raw]
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import load_documents
from src.ingestion.splitter import chunk_documents
from src.ingestion.indexer import index_chunks
from src.core.logger import setup_logging, logger


async def main(data_dir: str):
    setup_logging()
    logger.info("ingest_script_start", data_dir=data_dir)

    docs = load_documents(data_dir)
    if not docs:
        logger.error("no_documents_found", data_dir=data_dir)
        print(f"No PDFs found in {data_dir}. Put your PDF files there and try again.")
        return

    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc['source']} ({doc['page_count']} pages)")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    total = await index_chunks(chunks)
    print(f"Successfully indexed {total} chunks into Qdrant ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    asyncio.run(main(args.data_dir))
