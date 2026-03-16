"""
CLI script to run RAGAS evaluation.
Usage: python scripts/evaluate.py [--dataset eval/dataset.json]
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ragas_evaluator import run_evaluation
from src.core.logger import setup_logging


async def main(dataset_path: str):
    setup_logging()
    print(f"Running RAGAS evaluation on {dataset_path}...")
    results = await run_evaluation(dataset_path)

    print("\n" + "="*40)
    print("RAGAS EVALUATION RESULTS")
    print("="*40)
    print(f"Faithfulness:     {results['faithfulness']:.4f}  (target > 0.7)")
    print(f"Answer Relevancy: {results['answer_relevancy']:.4f}  (target > 0.7)")
    print(f"Samples:          {results['num_samples']}")
    print("="*40)

    if results["faithfulness"] >= 0.7 and results["answer_relevancy"] >= 0.7:
        print("✓ Both metrics passed!")
    else:
        print("✗ One or more metrics below target — consider tuning chunk size or retrieval top_k")

    print(f"\nFull results saved to eval/results/latest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/dataset.json")
    args = parser.parse_args()
    asyncio.run(main(args.dataset))
