# CLAUDE.md — Kenya DPA RAG Assistant

## Project
A RAG (retrieval-augmented generation) assistant that answers questions about the
**Kenya Data Protection Act, 2019**, using only the text of the Act (plus public guidance
from the Office of the Data Protection Commissioner (ODPC), if added).

- Every answer must cite the section(s) it came from.
- The assistant must say **"I don't know"** when the documents don't cover the question.
- This is a public portfolio project on GitHub, so **public documents only**.

## Pipeline
1. Collect public PDFs → `data/raw/`
2. Extract and clean text (`src/ingest.py`)
3. Split into overlapping chunks (~300 words), keeping section number + title on each chunk (`src/chunk.py`)
4. Embed chunks with sentence-transformers (`src/index.py`)
5. Store in ChromaDB, locally (`src/index.py`)
6. Retrieve the top chunks for a question (`src/retrieve.py`)
7. Send chunks + question to Claude via the Anthropic API: answer only from the chunks, cite sections, say "I don't know" if the answer isn't there (`src/answer.py`)
8. Evaluate on 50+ test questions (`src/evaluate.py`)
9. Simple Streamlit app (`app.py`)

Other files: `README.md`, `FINDINGS.md`, `requirements.txt`, `.gitignore`, `.env` (never committed).

## 5-day timeline
- **Day 1 (Fri):** setup, collect documents, `ingest.py` + `chunk.py`. Done when clean chunks print with section numbers.
- **Day 2 (Sat):** embeddings, vector store, retrieval. Done when a question returns the right section in the top 3.
- **Day 3 (Sun):** answer generation with Claude, citations, "I don't know" handling.
- **Day 4 (Mon):** 50 test questions with correct sections; measure retrieval + answer quality; compare at least 2 chunk sizes.
- **Day 5 (Tue):** Streamlit app, README with results, push to GitHub.

## Final deliverables
Public GitHub repo with working code, README (problem, how it works, how to run, results),
an evaluation table (top-3 retrieval hit rate, answer correctness, correct "I don't know" rate),
and a short demo for interviews.

## Rules for Claude
- The user is a learner: explain each step in simple terms, keep code simple and well commented, work in small steps.
- Don't write code until the user approves the plan for that day.
- After each step, explain what was done in simple terms and ask the user to confirm before moving on.
- **Ask before installing any package.**
- **Never put API keys in code or commit them.** Use `.env`, and make sure `.env` is in `.gitignore`.
- Keep `data/raw/` out of git if the files are large.
- **Never use documents from the user's employer. Public sources only.**
- After each finished piece of work, remind the user to commit and push.
- If something is too big for the time available, say what to cut instead of silently growing the scope.
