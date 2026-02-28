#!/usr/bin/env python3
"""
Main runner for the multi-model reasoning inference pipeline.

Usage:
    python runner.py --models gemini deepseek --limit 100 --workers 4
    python runner.py --models gemini --limit 10 --workers 2 --checkpoint-every 5
"""

import argparse
import csv
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt import prompt_template, format_result
from schema_extractor import get_schema
from base_model import BaseReasoningModel

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("runner")

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
INPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.csv")

FIELDNAMES = [
    "question_id",
    "schema_id",
    "nl_question",
    "sql_level",
    "nl_level",
    "explanation",
    "sql_answer",
    "think",
]


def read_questions(limit: int) -> list[dict]:
    """Read questions.csv and return up to `limit` rows as dicts."""
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)
    logger.info(f"Loaded {len(rows)} questions from {INPUT_CSV}")
    return rows


def output_path(model_name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"questions_{model_name}.csv",
    )


def load_checkpoint(model_name: str) -> dict[str, dict]:
    """Load existing checkpoint file if it exists. Returns dict keyed by question_id."""
    path = output_path(model_name)
    if not os.path.exists(path):
        return {}
    existing = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["question_id"]] = row
    logger.info(f"Loaded checkpoint for {model_name}: {len(existing)} rows")
    return existing


def save_checkpoint(model_name: str, rows: list[dict]):
    """Save all rows to the output CSV."""
    path = output_path(model_name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved checkpoint for {model_name}: {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def create_model(name: str) -> BaseReasoningModel:
    if name == "gemini":
        from gemini_model import GeminiModel
        return GeminiModel()
    elif name == "deepseek":
        from deepseek_model import DeepSeekModel
        return DeepSeekModel()
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Process a single question
# ---------------------------------------------------------------------------
def process_question(
    model: BaseReasoningModel,
    row: dict,
) -> dict:
    """Process one question: extract schema, build prompt, call model, parse result."""
    qid = row["question_id"]
    schema_id = row["schema_id"]
    question = row["nl_question"]

    logger.info(f"[{model.name}] Processing q{qid} (schema={schema_id})")
    start = time.time()

    # 1. Get schema
    schema = get_schema(schema_id)

    # 2. Build prompt
    prompt = prompt_template.format(schema=schema, question=question)

    # 3. Call model
    think_raw, sql = model.generate(prompt)

    # 4. Format the think part
    think_formatted = format_result(think_raw)

    elapsed = time.time() - start
    logger.info(f"[{model.name}] Done q{qid} in {elapsed:.1f}s")

    # 5. Build result row
    result = dict(row)
    result["sql_answer"] = sql
    result["think"] = think_formatted
    return result


# ---------------------------------------------------------------------------
# Run one model across all questions
# ---------------------------------------------------------------------------
def run_model(
    model_name: str,
    questions: list[dict],
    workers: int,
    checkpoint_every: int,
):
    """Run inference for a single model on all questions."""
    logger.info(f"=== Starting model: {model_name} ===")

    # Load checkpoint
    existing = load_checkpoint(model_name)

    # Build result list preserving order, pre-filling from checkpoint
    results = []
    pending_indices = []

    for i, row in enumerate(questions):
        qid = row["question_id"]
        if qid in existing and existing[qid].get("sql_answer", "").strip():
            results.append(existing[qid])
            logger.info(f"[{model_name}] Skipping q{qid} (already answered)")
        else:
            results.append(None)  # placeholder
            pending_indices.append(i)

    logger.info(
        f"[{model_name}] {len(pending_indices)} pending, "
        f"{len(results) - len(pending_indices)} already done"
    )

    if not pending_indices:
        save_checkpoint(model_name, results)
        return

    # Create model instance
    model = create_model(model_name)

    # Lock for checkpointing
    lock = threading.Lock()
    completed_since_checkpoint = 0

    def _process(idx):
        nonlocal completed_since_checkpoint
        row = questions[idx]
        try:
            result = process_question(model, row)
            with lock:
                results[idx] = result
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= checkpoint_every:
                    # Save checkpoint with what we have so far
                    safe_results = [r for r in results if r is not None]
                    save_checkpoint(model_name, safe_results)
                    completed_since_checkpoint = 0
        except Exception as e:
            logger.error(f"[{model_name}] Error on q{row['question_id']}: {e}")
            # Keep row as-is (no answer)
            with lock:
                result = dict(row)
                result.setdefault("sql_answer", "")
                result.setdefault("think", "")
                results[idx] = result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, idx): idx for idx in pending_indices}
        for future in as_completed(futures):
            future.result()  # re-raise exceptions if any

    # Final save
    final_results = [r for r in results if r is not None]
    save_checkpoint(model_name, final_results)
    logger.info(f"=== Finished model: {model_name} ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-model reasoning inference for Text-to-SQL"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gemini", "deepseek"],
        default=["gemini", "deepseek"],
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Max number of questions to process",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads per model (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N completed questions (default: 10)",
    )
    args = parser.parse_args()

    logger.info(
        f"Config: models={args.models}, limit={args.limit}, "
        f"workers={args.workers}, checkpoint_every={args.checkpoint_every}"
    )

    # Read questions
    questions = read_questions(args.limit)
    if not questions:
        logger.error("No questions loaded. Exiting.")
        sys.exit(1)

    # Run models in parallel (each model in its own thread)
    threads = []
    for model_name in args.models:
        t = threading.Thread(
            target=run_model,
            args=(model_name, questions, args.workers, args.checkpoint_every),
            name=f"model-{model_name}",
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    logger.info("All models finished.")


if __name__ == "__main__":
    main()
