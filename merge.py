#!/usr/bin/env python3
"""
Merge multiple questions_*.csv files by majority-vote on SQL execution results.

For each question_id:
  1. Collect all SQL answers from every questions_*.csv file.
  2. Execute each SQL against the corresponding SQLite database (with timeout).
  3. Group results that are equivalent (ignoring column names and row order,
     with numeric tolerance -- inspired by Spider/BIRD evaluation).
  4. Pick the largest group (majority vote); within that group prefer entries
     that have reasoning (think column non-empty).
  5. Write the winning SQL + think into questions_merged.csv.

Usage:
    python merge.py                          # defaults: timeout=30s
    python merge.py --timeout 60
    python merge.py --files questions_gemini.csv questions_deepseek.csv
"""

import argparse
import csv
import glob
import logging
import math
import multiprocessing
import os
import sqlite3
import sys
import traceback

csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DIR = os.path.join(BASE_DIR, "sqlite")
TOLERANCE = 1e-2

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


# ---------------------------------------------------------------------------
# Safe SQL execution in a child process (avoids segfault in main process)
# ---------------------------------------------------------------------------

def _exec_sql_worker(db_path: str, sql: str, queue: multiprocessing.Queue):
    """Run in a child process. Puts (rows, col_names) or an error string."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        queue.put(("ok", rows, col_names))
    except Exception as e:
        queue.put(("error", str(e)))


def execute_sql(db_path: str, sql: str, timeout: int = 30):
    """
    Execute SQL in a separate process with a timeout.

    Returns:
        (rows, col_names) on success, or None on error/timeout.
    """
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_sql_worker, args=(db_path, sql, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        logger.warning(f"SQL execution timed out after {timeout}s")
        return None

    if queue.empty():
        logger.warning("SQL execution returned no result")
        return None

    result = queue.get_nowait()
    if result[0] == "ok":
        return result[1], result[2]  # rows, col_names
    else:
        logger.warning(f"SQL execution error: {result[1]}")
        return None


# ---------------------------------------------------------------------------
# Result comparison (inspired by eval_example.py / Spider / BIRD)
# ---------------------------------------------------------------------------

def _normalize_value(v):
    """Normalize a value for comparison: try float, otherwise normalize string."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    # Normalize string: strip, collapse whitespace, normalize comma separators
    import re as _re
    s = str(v).strip()
    s = _re.sub(r'\s+', ' ', s)          # collapse multiple spaces
    s = _re.sub(r'\s*,\s*', ', ', s)      # normalize ", " / "," / " , " → ", "
    return s


def _row_signature(row):
    """Create a sortable signature for a row (for order-independent comparison)."""
    normed = [_normalize_value(v) for v in row]
    # Sort key: (is_none, str_repr) so Nones go last
    return tuple(
        (0 if v is not None else 1, "" if v is None else str(v))
        for v in normed
    )


def _vectors_match(v1, v2, tolerance=TOLERANCE) -> bool:
    """Check if two value vectors match (with numeric tolerance, order-sensitive)."""
    if len(v1) != len(v2):
        return False
    for a, b in zip(v1, v2):
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, abs_tol=tolerance):
                return False
        elif isinstance(a, float) or isinstance(b, float):
            return False
        else:
            if str(a) != str(b):
                return False
    return True


def results_equivalent(rows_a, rows_b) -> bool:
    """
    Compare two SQL result sets, ignoring:
      - Column names (compare values only)
      - Row order (sorted comparison)
      - Extra columns (subset check: smaller column set must be contained
        in the larger one, matched by column-vector values)
      - Numeric tolerance (1e-2)
    """
    if rows_a is None or rows_b is None:
        return rows_a is rows_b  # both None → equivalent

    if len(rows_a) != len(rows_b):
        return False

    if len(rows_a) == 0:
        return True

    # Normalize values
    norm_a = [tuple(_normalize_value(v) for v in row) for row in rows_a]
    norm_b = [tuple(_normalize_value(v) for v in row) for row in rows_b]

    ncols_a = len(norm_a[0]) if norm_a else 0
    ncols_b = len(norm_b[0]) if norm_b else 0

    if ncols_a == 0 and ncols_b == 0:
        return True

    if ncols_a == ncols_b:
        # Fast path: same column count → sort rows and compare directly
        sorted_a = sorted(norm_a, key=_row_signature)
        sorted_b = sorted(norm_b, key=_row_signature)
        for ra, rb in zip(sorted_a, sorted_b):
            if not _vectors_match(list(ra), list(rb)):
                return False
        return True

    # Different column counts → column-vector subset matching.
    # The result with fewer columns must have ALL its column-vectors
    # present in the result with more columns.
    # (Transpose approach, similar to eval_example.py)
    if ncols_a <= ncols_b:
        smaller, larger = norm_a, norm_b
    else:
        smaller, larger = norm_b, norm_a

    nrows = len(smaller)
    ncols_small = len(smaller[0])
    ncols_large = len(larger[0])

    # Sort both by the same row ordering (use the smaller's row signature)
    # We need a consistent row order for column-vector comparison.
    # Sort rows of both tables by a shared key.
    sorted_small = sorted(smaller, key=_row_signature)
    sorted_large = sorted(larger, key=_row_signature)

    # Extract column vectors (transposed)
    small_cols = [[sorted_small[r][c] for r in range(nrows)] for c in range(ncols_small)]
    large_cols = [[sorted_large[r][c] for r in range(nrows)] for c in range(ncols_large)]

    # Check every column in smaller has a match in larger
    used = set()
    for scol in small_cols:
        matched = False
        for j, lcol in enumerate(large_cols):
            if j in used:
                continue
            if _vectors_match(scol, lcol):
                used.add(j)
                matched = True
                break
        if not matched:
            return False

    return True


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def discover_csv_files(explicit_files=None):
    """Find all questions_*.csv files (or use explicit list)."""
    if explicit_files:
        return explicit_files
    pattern = os.path.join(BASE_DIR, "questions_*.csv")
    files = sorted(glob.glob(pattern))
    # Also include base questions.csv if it has sql_answer
    base = os.path.join(BASE_DIR, "questions.csv")
    if base not in files and os.path.exists(base):
        files.insert(0, base)
    return files


def read_all_csvs(file_paths):
    """
    Read all CSV files and return a dict:
        question_id -> list of {row_dict, source_file}
    """
    merged = {}
    for path in file_paths:
        source = os.path.basename(path)
        logger.info(f"Reading {source}")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row["question_id"]
                if qid not in merged:
                    merged[qid] = {"base_row": row, "candidates": []}
                sql = row.get("sql_answer", "").strip()
                think = row.get("think", "").strip()
                if sql:
                    merged[qid]["candidates"].append({
                        "sql": sql,
                        "think": think,
                        "source": source,
                    })
    return merged


# ---------------------------------------------------------------------------
# Majority vote logic
# ---------------------------------------------------------------------------

def pick_best(candidates, schema_id, timeout):
    """
    Execute each candidate SQL against the SQLite DB,
    group by equivalent results, and pick the majority winner.
    Prefer candidates with reasoning (think) within the winning group.

    Returns:
        (winner_candidate, group_size, total_candidates)
    """
    db_path = os.path.join(SQLITE_DIR, f"{schema_id}.sqlite")
    if not os.path.exists(db_path):
        logger.warning(f"DB not found: {db_path}, using first candidate")
        return candidates[0], 0, len(candidates)

    # Execute all candidates and store results
    exec_results = []  # list of (candidate, rows_or_none)
    for cand in candidates:
        result = execute_sql(db_path, cand["sql"], timeout=timeout)
        if result is not None:
            rows, _ = result
            exec_results.append((cand, rows))
        else:
            exec_results.append((cand, None))

    # Group by equivalent results
    # groups[i] = list of (candidate, rows) that are all equivalent
    groups = []
    for cand, rows in exec_results:
        placed = False
        for group in groups:
            _, rep_rows = group[0]
            if results_equivalent(rows, rep_rows):
                group.append((cand, rows))
                placed = True
                break
        if not placed:
            groups.append([(cand, rows)])

    if not groups:
        return (candidates[0], 0, len(candidates)) if candidates else (None, 0, 0)

    # Filter out groups where execution failed (rows is None)
    valid_groups = [g for g in groups if g[0][1] is not None]
    if not valid_groups:
        # All failed — fall back to first candidate
        return candidates[0], 0, len(candidates)

    # Sort groups: largest first; break ties by preferring groups with reasoning
    def group_sort_key(group):
        size = len(group)
        has_reasoning = sum(1 for c, _ in group if c["think"])
        return (size, has_reasoning)

    valid_groups.sort(key=group_sort_key, reverse=True)
    winning_group = valid_groups[0]
    group_size = len(winning_group)

    # Within winning group, prefer candidate with reasoning
    with_reasoning = [(c, r) for c, r in winning_group if c["think"]]
    if with_reasoning:
        return with_reasoning[0][0], group_size, len(candidates)
    return winning_group[0][0], group_size, len(candidates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge questions_*.csv files by majority-vote on SQL execution results"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit CSV files to merge (default: auto-discover questions_*.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="SQL execution timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: questions_merged.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(BASE_DIR, "questions_merged.csv")

    # 1. Discover and read CSV files
    csv_files = discover_csv_files(args.files)
    if not csv_files:
        logger.error("No CSV files found.")
        sys.exit(1)
    logger.info(f"Files to merge: {[os.path.basename(f) for f in csv_files]}")

    merged = read_all_csvs(csv_files)
    logger.info(f"Total questions: {len(merged)}")

    # 2. Process each question
    output_rows = []
    low_agreement = []  # questions with agreement=1 among multiple candidates
    sorted_qids = sorted(merged.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for qid in sorted_qids:
        entry = merged[qid]
        base_row = entry["base_row"]
        candidates = entry["candidates"]
        schema_id = base_row["schema_id"]

        if not candidates:
            logger.info(f"q{qid}: no SQL candidates, keeping original")
            result_row = dict(base_row)
            result_row.setdefault("sql_answer", "")
            result_row.setdefault("think", "")
            output_rows.append(result_row)
            continue

        if len(candidates) == 1:
            winner = candidates[0]
            logger.info(
                f"q{qid}: single candidate from {winner['source']}"
            )
        else:
            logger.info(
                f"q{qid}: {len(candidates)} candidates from "
                f"{[c['source'] for c in candidates]}, running majority vote..."
            )
            winner, group_size, total = pick_best(candidates, schema_id, args.timeout)
            logger.info(
                f"q{qid}: winner from {winner['source']}, "
                f"agreement {group_size}/{total}"
            )
            if group_size <= 1 and total > 1:
                low_agreement.append((qid, schema_id, winner["source"], group_size, total))
                logger.warning(
                    f"q{qid}: LOW AGREEMENT -- agreement {group_size}/{total}"
                )

        result_row = dict(base_row)
        result_row["sql_answer"] = winner["sql"]
        result_row["think"] = winner["think"]
        output_rows.append(result_row)

    # 3. Write output
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(f"Merged {len(output_rows)} questions -> {output_path}")

    # 4. Summary of low-agreement questions → CSV + per-level stats
    low_agreement_csv = os.path.join(BASE_DIR, "low_agreement.csv")
    if low_agreement:
        # Collect per-source SQL for each low-agreement question
        source_names = sorted({
            os.path.splitext(c["source"])[0]   # strip .csv → e.g. "questions_gemini"
            for entry in merged.values()
            for c in entry["candidates"]
        })

        la_rows = []
        level_counts: dict[str, int] = {}
        for qid, sid, picked_src, gs, total in low_agreement:
            entry = merged[qid]
            base_row = entry["base_row"]
            sql_level = base_row.get("sql_level", "")
            level_counts[sql_level] = level_counts.get(sql_level, 0) + 1

            row = {
                "question_id": qid,
                "schema_id": sid,
                "sql_level": sql_level,
                "agreement": f"{gs}/{total}",
            }
            # One column per source file with its SQL (empty if no candidate)
            src_map = {
                os.path.splitext(c["source"])[0]: c["sql"]
                for c in entry["candidates"]
            }
            for src in source_names:
                row[src] = src_map.get(src, "")
            la_rows.append(row)

        csv_fieldnames = ["question_id", "schema_id", "sql_level", "agreement"] + source_names
        with open(low_agreement_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(la_rows)

        logger.warning(
            f"{len(low_agreement)} low-agreement questions written -> {low_agreement_csv}"
        )

        # Print per-level breakdown to console
        print("\n=== Low-agreement questions by sql_level (sorted by count) ===")
        for level, count in sorted(level_counts.items(), key=lambda x: -x[1]):
            label = level if level else "(empty)"
            print(f"  {label}: {count}")
        print()
    else:
        logger.info("All multi-candidate questions had consensus (agreement >= 2).")
        if os.path.exists(low_agreement_csv):
            os.remove(low_agreement_csv)


if __name__ == "__main__":
    main()
