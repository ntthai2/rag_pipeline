from typing import List, Dict, Tuple
import yaml
from pathlib import Path
from src.generation.llm_client import generate
from src.core.logger import logger


def load_prompts() -> Dict:
    prompt_path = Path("config/prompts.yaml")
    with open(prompt_path) as f:
        return yaml.safe_load(f)


def build_context(chunks: List[Dict]) -> Tuple[str, List[str]]:
    """Build context string and deduplicated source list from chunks."""
    context_parts = []
    sources = []

    for i, chunk in enumerate(chunks):
        source = chunk["source"]
        context_parts.append(f"[{i+1}] (Source: {source})\n{chunk['text']}")
        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


async def build_and_generate(query: str, chunks: List[Dict]) -> Dict:
    """
    Build prompt from retrieved chunks and generate answer.
    Returns {answer, sources, context_used}
    """
    if not chunks:
        return {
            "answer": "I don't have enough information to answer this question based on the available documents.",
            "sources": [],
            "context_used": 0,
        }

    prompts = load_prompts()
    context, sources = build_context(chunks)

    messages = [
        {"role": "system", "content": prompts["rag_system"]},
        {
            "role": "user",
            "content": prompts["rag_user"].format(
                context=context,
                question=query,
            ),
        },
    ]

    answer = await generate(messages)
    logger.info("generated_answer", query=query[:60], sources=sources, chunks_used=len(chunks))

    return {
        "answer": answer,
        "sources": sources,
        "context_used": len(chunks),
    }
