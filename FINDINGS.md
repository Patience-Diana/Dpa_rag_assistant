# Findings

What I measured, what worked, what didn't, and what I would do next.

## Test set

- **64 questions** in [`data/eval_questions.csv`](data/eval_questions.csv):
  - **54 answerable** questions, each with the correct section number(s) and a short expected answer.
  - **10 unanswerable** questions that the Act does not cover. The assistant should say "I don't know" to these.
- Some questions are deliberate traps. For example:
  - The Act does not define a child's age, so the assistant should not assume 18.
  - The Act's 30-day limit applies to data portability, not to access requests.
- **Caveat:** the questions were written while reading the Act. Their wording is therefore probably closer to the Act's wording than a real user's would be, and the scores below may be **optimistic**.

## 1. Retrieval: does search find the right section?

A question counts as a hit if a chunk from a correct section appears in the top *k* results.

| Chunk size | Chunks | Hit @1 | **Hit @3** | Hit @5 |
|---|---|---|---|---|
| **300 words** (overlap 50) | 83 | 80% | **96%** | 98% |
| 150 words (overlap 25) | 119 | 80% | 94% | 98% |

Embedding model: `BAAI/bge-small-en-v1.5`. Vector store: ChromaDB, using cosine similarity.

**Result:** 300-word chunks were slightly better, by **one question out of 54**. That difference is too small to say one size is clearly better. I kept 300 words as the default.

**Why smaller chunks didn't help:**
- Most sections of the Act are already shorter than 150 words, so they are identical in both settings.
- Long sections get cut into several similar pieces. Those pieces can fill several of the top slots by themselves and push the correct section down. For example, for *"On what legal grounds can personal data be processed?"* the 150-word index returned sections 45, 45, 34, 4, and then the correct section 30 in 5th place.

### Where retrieval fails: definitions (Section 2)

| Question | Expected | 300 words | 150 words |
|---|---|---|---|
| What is the difference between a data controller and a data processor? | 2 | not in top 5 | rank 5 |
| What counts as sensitive personal data? | 2 | ✓ | not in top 5 |
| On what legal grounds can personal data be processed? | 30 | rank 4 | rank 5 |

Section 2 holds **all the definitions of the Act** in one long section (4 chunks at 300 words, 7 at 150). Each chunk mixes many unrelated definitions, so no single chunk matches a question about one definition very strongly. Other sections that simply *use* the defined term (for example section 18 on registering data controllers) win instead.

**Possible fix (not done, out of time):** split Section 2 into one small chunk per definition, labelled for example `Section 2: Interpretation — "data processor"`.

## 2. Search scores can't tell when to say "I don't know"

- I checked whether a similarity-score cut-off could detect off-topic questions. It can't.
- Unanswerable questions got top scores of about **0.57–0.66**, while a correctly answered question scored **0.647**. The ranges overlap, so no cut-off separates them.
- So the decision is left to Claude. Claude reads the retrieved excerpts and is told to say "I don't know" when they don't contain the answer.

## 3. Answer quality (Claude)

> ⏳ **Not run yet.** The evaluation code is ready (`python src/evaluate.py --answers`) but needs an Anthropic API key.

When run, it measures:

| Metric | Result |
|---|---|
| Answerable questions answered correctly (LLM judge) | pending |
| Answers citing at least one correct section | pending |
| Unanswerable questions where it correctly said "I don't know" | pending |
| Answerable questions where it wrongly said "I don't know" | pending |
| Judge verdicts I checked by hand / agreed with | pending |

## What I would do next

1. Split Section 2 into one chunk per definition, and re-measure.
2. Rewrite part of the test set in everyday language (how a member of the public would ask), to get a less optimistic score.
3. Add the ODPC's public guidance notes as a second source.
