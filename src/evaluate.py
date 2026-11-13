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
    python src/evaluate.py
"""

import csv
from pathlib import Path

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


def evaluate_retrieval(questions):
    """Run retrieval for every answerable question and record where the hit was."""
    results = []
    for q in questions:
        if q["type"] != "answerable":
            continue  # unanswerable questions have no "correct section" to find

        chunks = retrieve(q["question"], top_k=MAX_K)
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


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions. Evaluating retrieval ...")

    results = evaluate_retrieval(questions)
    summary = summarise(results)
    path = save_results(results, "retrieval_results.csv")

    print(f"\n=== Retrieval results ({summary['n_questions']} answerable questions) ===")
    print(f"  Hit rate @1: {summary['hit_at_1']:.0%}")
    print(f"  Hit rate @3: {summary['hit_at_3']:.0%}   <- main metric")
    print(f"  Hit rate @5: {summary['hit_at_5']:.0%}")

    # Show the questions that did NOT get a correct section in the top 3,
    # so we can look at why.
    misses = [r for r in results if not r["hit_at_3"]]
    if misses:
        print(f"\nMissed in top 3 ({len(misses)}):")
        for r in misses:
            where = f"found at rank {r['hit_rank']}" if r["hit_rank"] else "not in top 5"
            print(f"  [{r['id']}] {r['question']}")
            print(f"       expected {r['expected_sections']}, got {r['retrieved_sections']} ({where})")
    else:
        print("\nNo misses in the top 3.")

    print(f"\nPer-question details saved to {path}")


if __name__ == "__main__":
    main()
