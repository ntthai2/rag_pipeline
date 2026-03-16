from pathlib import Path
from typing import List, Dict
from docling.document_converter import DocumentConverter
from src.core.logger import logger


def load_documents(data_dir: str = "data/raw") -> List[Dict]:
    """
    Load all PDFs from data_dir using Docling.
    Returns list of {text, source, page_count}
    """
    converter = DocumentConverter()
    docs = []
    pdf_files = list(Path(data_dir).glob("**/*.pdf"))

    if not pdf_files:
        logger.warning("no_pdfs_found", data_dir=data_dir)
        return docs

    logger.info("loading_documents", count=len(pdf_files))

    for pdf_path in pdf_files:
        try:
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
            docs.append({
                "text": text,
                "source": pdf_path.name,
                "path": str(pdf_path),
                "page_count": len(result.document.pages) if result.document.pages else 0,
            })
            logger.info("loaded_doc", source=pdf_path.name, chars=len(text))
        except Exception as e:
            logger.error("failed_to_load", source=str(pdf_path), error=str(e))

    return docs
