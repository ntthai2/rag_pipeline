import json
import asyncio
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
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
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    results = {
        "faithfulness": round(float(scores["faithfulness"]), 4),
        "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
        "num_samples": len(rows),
    }

    # Save results
    results_dir = Path("eval/results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "latest.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("eval_complete", **results)
    return results


if __name__ == "__main__":
    asyncio.run(run_evaluation())
