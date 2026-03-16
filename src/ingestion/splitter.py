from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import settings
from src.core.logger import logger


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    """
    Split documents into chunks with metadata.
    Returns list of {text, source, chunk_id, page_count}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}__chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(splits),
            })

    logger.info("chunking_done", total_chunks=len(chunks))
    return chunks
