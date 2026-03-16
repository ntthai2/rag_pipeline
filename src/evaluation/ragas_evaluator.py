import json
import asyncio
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from src.retrieval.retriever import retrieve
from src.generation.response_builder import build_and_generate
from src.core.config import settings
from src.core.logger import logger


async def run_single(question: str, ground_truth: str) -> dict:
    chunks = await retrieve(question, top_k=settings.top_k)
    top_chunks = chunks[:settings.top_k_rerank]
    result = await build_and_generate(question, top_chunks)
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": [c["text"] for c in top_chunks],
        "ground_truth": ground_truth,
    }


async def run_evaluation(dataset_path: str = "eval/dataset.json") -> dict:
    with open(dataset_path) as f:
        eval_data = json.load(f)

    logger.info("eval_start", samples=len(eval_data))

    rows = []
    for item in eval_data:
        row = await run_single(item["question"], item["ground_truth"])
        rows.append(row)
        logger.info("eval_sample_done", question=item["question"][:60])

    dataset = Dataset.from_list(rows)

    llm = LangchainLLMWrapper(ChatOpenAI(
        base_url="http://localhost/v1",
        api_key="secret-key-change-me",
        model="/models/Qwen2.5-1.5B-Instruct-AWQ",
        temperature=0.1,
        max_retries=5,
        max_tokens=512,
    ))

    scores = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=llm,
    )

    def extract_score(val):
        if isinstance(val, list):
            valid = [v for v in val if v is not None and str(v) != 'nan']
            return round(sum(valid) / len(valid), 4) if valid else 0.0
        try:
            f = float(val)
            return 0.0 if f != f else round(f, 4)
        except:
            return 0.0

    results = {
        "faithfulness": extract_score(scores["faithfulness"]),
        "answer_relevancy": "N/A - requires embedding API",
        "num_samples": len(rows),
    }

    results_dir = Path("eval/results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "latest.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("eval_complete", faithfulness=results["faithfulness"], num_samples=len(rows))
    return results


if __name__ == "__main__":
    asyncio.run(run_evaluation())