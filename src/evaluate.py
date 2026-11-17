"""
evaluate.py — Step 8 of the pipeline: measure how well the assistant works.

Part 1: RETRIEVAL evaluation. Free, no API key needed.
  For every answerable question in data/eval_questions.csv, we run retrieve()
  and check whether a correct section appears in the results.

  We report:
    - Hit rate @1: the correct section is the FIRST result
    - Hit rate @3: the correct section is somewhere in the top 3  <- main metric
    - Hit rate @5: in the top 5 (that's how many chunks answer.py gives Claude)

Part 2: ANSWER evaluation with Claude. Costs money (roughly $3 for all
  64 questions), so it only runs when you add --answers.
  For EVERY question (answerable and unanswerable) we run answer() and check:
    - "I don't know" accuracy: did it say "I don't know" exactly when it should?
    - Citation accuracy: did it cite at least one correct section?
    - Correctness: a second Claude call (the "judge") compares the answer with
      the expected answer and says correct / partial / wrong.
  The judge can make mistakes too, so check about 10 of its verdicts by hand:
  fill in the empty "my_check" column in results/answer_results_<size>.csv.

Run it from the project root with:
    python src/evaluate.py                  # retrieval, 300-word chunks
    python src/evaluate.py --sizes 300 150  # compare chunk sizes side by side
    python src/evaluate.py --answers --limit 5   # cheap trial: 5 questions
    python src/evaluate.py --answers             # full answer evaluation
(Each size must be built first with chunk.py --size N and index.py --size N.)
"""

import argparse
import csv
import re
from pathlib import Path

import anthropic

from answer import MODEL, answer
from config import DEFAULT_CHUNK_SIZE
from retrieve import retrieve

QUESTIONS_PATH = Path("data/eval_questions.csv")
RESULTS_DIR = Path("results")

# We retrieve this many chunks once, then check the top 1, 3 and 5 of them.
MAX_K = 5

# Instructions for the judge. It only compares facts: style and citations
# are checked separately, so it should ignore them.
JUDGE_PROMPT = """You are grading an assistant's answer to a question about the Kenya Data Protection Act, 2019.

Compare the ASSISTANT ANSWER with the REFERENCE ANSWER, which is correct.
- correct: it contains the key facts of the reference answer and nothing that contradicts it.
- partial: it gets the main idea right but misses an important fact, or adds a small error.
- wrong: it contradicts the reference, misses the main point, or says "I don't know".
Ignore writing style, length, and section citations. Extra correct detail is fine.

QUESTION: {question}

REFERENCE ANSWER: {expected}

ASSISTANT ANSWER: {answer}

Reply with exactly one word: correct, partial, or wrong."""


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


# ---------------------------------------------------------------------------
# Part 2: answer evaluation (calls Claude, costs money)
# ---------------------------------------------------------------------------

def judge(client, question, expected, answer_text):
    """Ask Claude to grade one answer. Returns "correct", "partial" or "wrong"."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, expected=expected, answer=answer_text)}],
        # Grading against a given reference is easy, so low effort is enough.
        output_config={"effort": "low"},
    )
    text = "".join(b.text for b in response.content if b.type == "text").lower()
    # Take the first verdict word we find. (\b stops "incorrect" counting as "correct".)
    match = re.search(r"\b(correct|partial|wrong)\b", text)
    return match.group(1) if match else "unclear"


def evaluate_answers(questions, chunk_size):
    """Answer every question with Claude and grade the result."""
    client = anthropic.Anthropic()
    results = []
    for i, q in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {q['question'][:70]}")
        should_idk = q["type"] == "unanswerable"

        try:
            r = answer(q["question"], chunk_size=chunk_size)
        except anthropic.APIError as e:
            # Don't lose the whole run because of one failed call.
            print(f"     API error, skipping: {e}")
            continue

        # Citation check: only makes sense for answerable questions.
        cites_correct = ""
        if not should_idk:
            cites_correct = any(s in q["expected_sections"] for s in r["cited_sections"])

        # Correctness:
        #  - unanswerable question: correct only if it said "I don't know"
        #  - answerable but it said "I don't know": wrong (no need to ask the judge)
        #  - otherwise: ask the judge
        if should_idk:
            verdict = "correct" if r["said_idk"] else "wrong"
        elif r["said_idk"]:
            verdict = "wrong"
        else:
            verdict = judge(client, q["question"], q["expected_answer"], r["answer"])

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected_sections": ";".join(q["expected_sections"]),
            "cited_sections": ";".join(r["cited_sections"]),
            "retrieved_sections": ";".join(c["section_number"] for c in r["retrieved"]),
            "said_idk": r["said_idk"],
            "idk_correct": r["said_idk"] == should_idk,
            "cites_correct": cites_correct,
            "bad_citations": ";".join(r["bad_citations"]),
            "verdict": verdict,
            "my_check": "",  # for YOU: write agree / disagree for ~10 rows
            "expected_answer": q["expected_answer"],
            "answer": r["answer"],
        })
    return results


def summarise_answers(results):
    """Print the answer-quality numbers."""
    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    def pct(part, whole):
        return f"{len(part)}/{len(whole)} ({len(part) / len(whole):.0%})" if whole else "-"

    print(f"\n=== Answer quality ({len(results)} questions) ===")
    if answerable:
        print(f"  Answerable questions ({len(answerable)}):")
        for v in ("correct", "partial", "wrong", "unclear"):
            hits = [r for r in answerable if r["verdict"] == v]
            if hits:
                print(f"    {v:<8} {pct(hits, answerable)}")
        print(f"    cited a correct section:  "
              f"{pct([r for r in answerable if r['cites_correct']], answerable)}")
        print(f"    wrongly said \"I don't know\": "
              f"{pct([r for r in answerable if r['said_idk']], answerable)}")
    if unanswerable:
        print(f"  Unanswerable questions ({len(unanswerable)}):")
        print(f"    correctly said \"I don't know\": "
              f"{pct([r for r in unanswerable if r['said_idk']], unanswerable)}")
    print(f"  Overall \"I don't know\" accuracy: "
          f"{pct([r for r in results if r['idk_correct']], results)}")
    bad = [r for r in results if r["bad_citations"]]
    print(f"  Answers citing a section that wasn't retrieved: {len(bad)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[DEFAULT_CHUNK_SIZE],
                        help="chunk sizes to evaluate, e.g. --sizes 300 150")
    parser.add_argument("--answers", action="store_true",
                        help="also evaluate answers with Claude (costs money)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N questions for the answer test")
    args = parser.parse_args()
    sizes = args.sizes

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

    # Part 2 only runs when asked, and only on the first chunk size.
    if args.answers:
        size = sizes[0]
        chosen = questions
        if args.limit:
            # Take a few of each type, so a trial run tests both kinds.
            half = max(1, args.limit // 2)
            chosen = ([q for q in questions if q["type"] == "answerable"][:args.limit - half]
                      + [q for q in questions if q["type"] == "unanswerable"][:half])
        print(f"\n=== Answer evaluation: {len(chosen)} questions, {size}-word chunks ===")
        results = evaluate_answers(chosen, size)
        if not results:
            print("No answers were evaluated (every call failed).")
            return
        suffix = "_trial" if args.limit else ""
        path = save_results(results, f"answer_results_{size}{suffix}.csv")
        summarise_answers(results)
        print(f"  Details saved to {path}")


if __name__ == "__main__":
    main()
