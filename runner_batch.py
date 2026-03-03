#!/usr/bin/env python3
"""
Batch runner for the multi-model reasoning inference pipeline.

Uses the Batch APIs of Gemini, Claude, and OpenAI for asynchronous processing
at 50% cost with higher rate limits. Suitable for large-scale inference where
immediate response is not required.

Usage:
    # Submit a new batch
    python runner_batch.py submit --models gemini claude openai --limit 2000

    # Check status of a submitted batch
    python runner_batch.py status --models gemini

    # Retrieve results when batch is complete
    python runner_batch.py retrieve --models gemini

    # Submit and wait until done (all-in-one)
    python runner_batch.py run --models gemini --limit 2000 --poll-interval 120
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time

# Reuse the EXACT same prompt/format/schema logic from the existing codebase
from prompt import prompt_template, format_result
from schema_extractor import get_schema
from batch_providers import create_batch_provider, BaseBatchProvider

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("runner_batch")

# ---------------------------------------------------------------------------
# CSV helpers (SAME as runner.py — NOT modified)
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

BATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_jobs")


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
        f"questions_batch_{model_name}.csv",
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
# Response parsing (SAME logic as gemini_model.py / deepseek_model.py)
# ---------------------------------------------------------------------------
def _extract_sql(text: str) -> str:
    match = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    logger.warning("Could not extract SQL from response, returning raw text")
    return text.strip()


def _extract_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Batch job metadata persistence
# ---------------------------------------------------------------------------
def _ensure_batch_dir():
    os.makedirs(BATCH_DIR, exist_ok=True)


def _batch_meta_path(model_name: str) -> str:
    return os.path.join(BATCH_DIR, f"{model_name}_batch.json")


def save_batch_meta(model_name: str, meta: dict):
    """Save batch job metadata (batch_id, custom_id ordering, etc.)."""
    _ensure_batch_dir()
    path = _batch_meta_path(model_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved batch metadata -> {path}")


def load_batch_meta(model_name: str) -> dict:
    """Load batch job metadata."""
    path = _batch_meta_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No batch metadata found for '{model_name}'. "
            f"Did you run 'submit' first? Expected: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core batch operations
# ---------------------------------------------------------------------------
def cmd_submit(args):
    """Submit batch jobs for the specified models."""
    questions = read_questions(args.limit)
    if not questions:
        logger.error("No questions loaded. Exiting.")
        sys.exit(1)

    for model_name in args.models:
        logger.info(f"=== Submitting batch for: {model_name} ===")

        # Load checkpoint to skip already-done questions
        existing = load_checkpoint(model_name)

        # Build requests for questions that need processing
        batch_requests = []
        # Keep a full ordered list mapping index -> question for later retrieval
        ordered_questions = []

        for i, row in enumerate(questions):
            qid = row["question_id"]
            ordered_questions.append(row)

            if (
                qid in existing
                and existing[qid].get("sql_answer", "").strip()
                and existing[qid].get("think", "").strip()
            ):
                logger.info(f"[{model_name}] Skipping q{qid} (already complete)")
                continue

            # Build prompt (SAME logic as runner.py process_question)
            schema = get_schema(row["schema_id"])
            prompt = prompt_template.format(schema=schema, question=row["nl_question"])

            batch_requests.append({
                "custom_id": f"q{qid}",
                "prompt": prompt,
            })

        if not batch_requests:
            logger.info(f"[{model_name}] All questions already complete. Nothing to submit.")
            continue

        logger.info(f"[{model_name}] Submitting {len(batch_requests)} requests ...")

        # Create provider and submit
        provider = create_batch_provider(model_name)

        # For Gemini inline batch, custom_id mapping uses index order
        # We store the original custom_ids in order for inline-indexed results
        custom_id_order = [r["custom_id"] for r in batch_requests]

        batch_id = provider.submit(batch_requests)

        # Persist metadata
        meta = {
            "batch_id": batch_id,
            "model_name": model_name,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_requests": len(batch_requests),
            "custom_id_order": custom_id_order,
            "limit": args.limit,
        }
        save_batch_meta(model_name, meta)
        logger.info(f"[{model_name}] Batch submitted: {batch_id}")


def cmd_status(args):
    """Check the status of submitted batch jobs."""
    for model_name in args.models:
        try:
            meta = load_batch_meta(model_name)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        provider = create_batch_provider(model_name)
        status = provider.poll(meta["batch_id"])

        print(f"[{model_name}] Batch ID: {meta['batch_id']}")
        print(f"[{model_name}] Submitted at: {meta['submitted_at']}")
        print(f"[{model_name}] Total requests: {meta['total_requests']}")
        print(f"[{model_name}] Status: {status}")
        print()


def cmd_retrieve(args):
    """Retrieve results from completed batch jobs and save to CSV."""
    questions = read_questions(args.limit)
    if not questions:
        logger.error("No questions loaded. Exiting.")
        sys.exit(1)

    for model_name in args.models:
        try:
            meta = load_batch_meta(model_name)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        provider = create_batch_provider(model_name)

        # Verify completion
        status = provider.poll(meta["batch_id"])
        if status != "completed":
            logger.warning(
                f"[{model_name}] Batch not yet complete (status: {status}). "
                f"Run 'status' to check, or 'run' to wait."
            )
            continue

        logger.info(f"[{model_name}] Retrieving results ...")
        raw_results = provider.get_results(meta["batch_id"])

        # Build custom_id -> index mapping for Gemini (inline uses numeric indices)
        custom_id_order = meta.get("custom_id_order", [])
        idx_to_custom_id = {str(i): cid for i, cid in enumerate(custom_id_order)}

        # Normalize keys: Gemini inline returns "0", "1", etc.
        # We map them back to "q{qid}" format
        normalized_results = {}
        for key, value in raw_results.items():
            if key in idx_to_custom_id:
                normalized_results[idx_to_custom_id[key]] = value
            else:
                normalized_results[key] = value

        # Load existing checkpoint
        existing = load_checkpoint(model_name)

        # Build ordered results list from original questions
        limit = meta.get("limit", args.limit)
        result_rows = []
        fail_count = 0

        for row in questions[:limit]:
            qid = row["question_id"]
            custom_id = f"q{qid}"

            # Check if already in checkpoint
            if (
                qid in existing
                and existing[qid].get("sql_answer", "").strip()
                and existing[qid].get("think", "").strip()
            ):
                result_rows.append(existing[qid])
                continue

            # Try to get from batch results
            if custom_id in normalized_results:
                response_text = normalized_results[custom_id]

                # Parse response (SAME logic as gemini_model.py / deepseek_model.py)
                think_text = format_result(_extract_think(response_text))
                sql = _extract_sql(response_text)

                result = dict(row)
                result["sql_answer"] = sql
                result["think"] = think_text

                if not sql.strip() or not think_text.strip():
                    fail_count += 1
                    missing = []
                    if not sql.strip():
                        missing.append("SQL")
                    if not think_text.strip():
                        missing.append("reasoning")
                    logger.warning(
                        f"[{model_name}] q{qid}: missing {', '.join(missing)}"
                    )

                result_rows.append(result)
            else:
                # Not in results (maybe skipped or errored)
                result = dict(row)
                result.setdefault("sql_answer", "")
                result.setdefault("think", "")
                result_rows.append(result)
                fail_count += 1
                logger.warning(f"[{model_name}] q{qid}: not found in batch results")

        # Save to CSV (SAME format as runner.py)
        save_checkpoint(model_name, result_rows)

        success_count = len(result_rows) - fail_count
        logger.info(
            f"[{model_name}] Done! {success_count} succeeded, {fail_count} failed/incomplete "
            f"out of {len(result_rows)} total."
        )


def cmd_run(args):
    """Submit, wait, and retrieve — all in one."""
    # Step 1: Submit
    cmd_submit(args)

    # Step 2: Wait for all models
    for model_name in args.models:
        try:
            meta = load_batch_meta(model_name)
        except FileNotFoundError:
            continue

        provider = create_batch_provider(model_name)
        final_status = provider.wait_for_completion(
            meta["batch_id"],
            poll_interval=args.poll_interval,
        )

        if final_status != "completed":
            logger.error(
                f"[{model_name}] Batch ended with status: {final_status}. "
                f"Results may be incomplete."
            )

    # Step 3: Retrieve
    cmd_retrieve(args)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch API runner for multi-model Text-to-SQL inference"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Common arguments
    def add_common_args(sub):
        sub.add_argument(
            "--models",
            nargs="+",
            choices=["gemini", "claude", "openai"],
            default=["gemini"],
            help="Models to run (default: gemini)",
        )
        sub.add_argument(
            "--limit",
            type=int,
            required=True,
            help="Max number of questions to process",
        )

    # --- submit ---
    sub_submit = subparsers.add_parser("submit", help="Submit batch jobs")
    add_common_args(sub_submit)
    sub_submit.set_defaults(func=cmd_submit)

    # --- status ---
    sub_status = subparsers.add_parser("status", help="Check batch job status")
    sub_status.add_argument(
        "--models",
        nargs="+",
        choices=["gemini", "claude", "openai"],
        default=["gemini"],
    )
    sub_status.add_argument("--limit", type=int, default=9999)
    sub_status.set_defaults(func=cmd_status)

    # --- retrieve ---
    sub_retrieve = subparsers.add_parser("retrieve", help="Retrieve batch results")
    add_common_args(sub_retrieve)
    sub_retrieve.set_defaults(func=cmd_retrieve)

    # --- run ---
    sub_run = subparsers.add_parser("run", help="Submit, wait, and retrieve (all-in-one)")
    add_common_args(sub_run)
    sub_run.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between status checks (default: 120)",
    )
    sub_run.set_defaults(func=cmd_run)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
