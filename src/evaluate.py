"""
evaluate.py — Step 8 of the pipeline: measure how well the assistant works.

Part 1 (this file, for now): RETRIEVAL evaluation. Free, no API key needed.
  For every answerable question in data/eval_questions.csv, we run retrieve()
  and check whether a correct section appears in the results.

  We report:
    - Hit rate @1: the correct section is the FIRST result
    - Hit rate @3: the correct section is somewhere in the top 3  <- main metric
    - Hit rate @5: in the top 5 (that's how many chunks answer.py gives Claude)

Part 2 (coming next): ANSWER evaluation with Claude, i.e. answer correctness
  and how often the assistant correctly says "I don't know".

Run it from the project root with:
    python src/evaluate.py                  # evaluate the 300-word chunks
    python src/evaluate.py --sizes 300 150  # compare chunk sizes side by side
(Each size must be built first with chunk.py --size N and index.py --size N.)
"""

import argparse
import csv
from pathlib import Path

from config import DEFAULT_CHUNK_SIZE
from retrieve import retrieve

QUESTIONS_PATH = Path("data/eval_questions.csv")
RESULTS_DIR = Path("results")

# We retrieve this many chunks once, then check the top 1, 3 and 5 of them.
MAX_K = 5


def load_questions():
    """Read the test questions. expected_sections becomes a list, e.g. ['40', '26']."""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["expected_sections"] = [s for s in row["expected_sections"].split(";") if s]
    return rows


def first_hit_rank(retrieved_sections, expected_sections):
    """
    Return the position (1, 2, 3, ...) of the first retrieved chunk that comes
    from a correct section, or None if no correct section was retrieved.
    """
    for rank, section in enumerate(retrieved_sections, start=1):
        if section in expected_sections:
            return rank
    return None


def evaluate_retrieval(questions, chunk_size):
    """Run retrieval for every answerable question and record where the hit was."""
    results = []
    for q in questions:
        if q["type"] != "answerable":
            continue  # unanswerable questions have no "correct section" to find

        chunks = retrieve(q["question"], top_k=MAX_K, chunk_size=chunk_size)
        retrieved = [c["section_number"] for c in chunks]
        rank = first_hit_rank(retrieved, q["expected_sections"])

        results.append({
            "id": q["id"],
            "question": q["question"],
            "expected_sections": ";".join(q["expected_sections"]),
            "retrieved_sections": ";".join(retrieved),
            "hit_rank": rank if rank else "",          # empty = not found in top 5
            "hit_at_1": rank is not None and rank <= 1,
            "hit_at_3": rank is not None and rank <= 3,
            "hit_at_5": rank is not None and rank <= 5,
            "top_score": chunks[0]["score"],
        })
    return results


def summarise(results):
    """Turn the per-question results into hit rates (percentages)."""
    n = len(results)
    return {
        f"hit_at_{k}": sum(r[f"hit_at_{k}"] for r in results) / n
        for k in (1, 3, 5)
    } | {"n_questions": n}


def save_results(results, filename):
    """Save the per-question results as a CSV so we can inspect them later."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return path


def print_misses(results):
    """Show the questions that did NOT get a correct section in the top 3."""
    misses = [r for r in results if not r["hit_at_3"]]
    if not misses:
        print("  No misses in the top 3.")
        return
    print(f"  Missed in top 3 ({len(misses)}):")
    for r in misses:
        where = f"found at rank {r['hit_rank']}" if r["hit_rank"] else "not in top 5"
        print(f"    [{r['id']}] {r['question']}")
        print(f"         expected {r['expected_sections']}, got {r['retrieved_sections']} ({where})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[DEFAULT_CHUNK_SIZE],
                        help="chunk sizes to evaluate, e.g. --sizes 300 150")
    sizes = parser.parse_args().sizes

    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")

    summaries = {}
    for size in sizes:
        results = evaluate_retrieval(questions, size)
        summaries[size] = summarise(results)
        path = save_results(results, f"retrieval_results_{size}.csv")

        print(f"\n=== {size}-word chunks ===")
        print_misses(results)
        print(f"  Details saved to {path}")

    # A small comparison table: one row per chunk size.
    n = summaries[sizes[0]]["n_questions"]
    print(f"\n=== Retrieval hit rates ({n} answerable questions) ===")
    print(f"{'Chunk size':>10} | {'@1':>5} | {'@3 (main)':>9} | {'@5':>5}")
    print("-" * 40)
    for size, s in summaries.items():
        print(f"{size:>10} | {s['hit_at_1']:>5.0%} | {s['hit_at_3']:>9.0%} | {s['hit_at_5']:>5.0%}")


if __name__ == "__main__":
    main()
