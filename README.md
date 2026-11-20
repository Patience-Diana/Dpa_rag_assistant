# Kenya Data Protection Act RAG Assistant

A question-answering assistant for the **Kenya Data Protection Act, 2019**.
It answers **only from the text of the Act**, **cites the section(s)** each answer came from,
and says **"I don't know"** when the Act doesn't cover the question.

## The problem

The Data Protection Act is long, and most people never read it. General AI chatbots can
answer questions about it, but they may mix in other countries' laws or invent details, and
they rarely show where an answer came from.

This project uses **retrieval-augmented generation (RAG)** to fix that:
1. The assistant first **finds** the relevant sections of the Act.
2. It then **answers using only those sections**, with citations, so every answer can be checked.

## How it works

```
 Act PDF ──► ingest.py ──► chunk.py ──► index.py ──► ChromaDB
            (clean text)  (300-word     (embeddings)   (local vector
                           chunks with                   database)
                           section labels)                   │
                                                             ▼
 Question ──────────────────────────────────────────► retrieve.py
                                                     (top 5 chunks)
                                                             │
                                                             ▼
                                                        answer.py ──► Answer + citations
                                                     (Claude, answers      or "I don't know"
                                                      only from chunks)
```

| Step | File | What it does |
|---|---|---|
| 1. Ingest | [`src/ingest.py`](src/ingest.py) | Extracts text from the PDF and removes page headers, page numbers and broken characters. |
| 2. Chunk | [`src/chunk.py`](src/chunk.py) | Splits the Act into its sections, then into overlapping ~300-word chunks. Each chunk keeps its **section number, title and Part**. |
| 3. Index | [`src/index.py`](src/index.py) | Turns each chunk into an embedding with `BAAI/bge-small-en-v1.5` and stores it in a local **ChromaDB** database. |
| 4. Retrieve | [`src/retrieve.py`](src/retrieve.py) | Embeds the question and finds the most similar chunks (cosine similarity). |
| 5. Answer | [`src/answer.py`](src/answer.py) | Sends the chunks and the question to **Claude** with strict rules: use only the excerpts, cite a section after every statement, and reply "I don't know" if the answer isn't there. |
| 6. Evaluate | [`src/evaluate.py`](src/evaluate.py) | Measures retrieval and answer quality on 64 test questions. |
| 7. App | [`app.py`](app.py) | A simple Streamlit web app. |

**Design choices:**
- **Section labels on every chunk.** The label (for example `Section 40: Right of rectification and erasure`) is part of the chunk's text, so both the search and Claude know where each chunk came from. That is what makes the citations possible.
- **Claude decides "I don't know", not a score cut-off.** Search similarity scores for off-topic questions overlapped with scores for real ones, so no cut-off could separate them (see [FINDINGS.md](FINDINGS.md)).
- **Citation check.** The code checks that every section Claude cites was actually among the retrieved chunks, and flags it if not.

## Results

Evaluated on **64 questions** ([`data/eval_questions.csv`](data/eval_questions.csv)): 54 answerable questions and 10 that the Act doesn't cover.

**Retrieval:** is a correct section in the top *k* results? (54 answerable questions)

| Chunk size | Hit @1 | **Hit @3** | Hit @5 |
|---|---|---|---|
| **300 words** (default) | 80% | **96%** | 98% |
| 150 words | 80% | 94% | 98% |

**Answers (Claude):** ⏳ *pending, will be added once the answer evaluation has been run.*

| Metric | Result |
|---|---|
| Answer correctness (LLM judge, spot-checked by hand) | pending |
| Correct "I don't know" rate (unanswerable questions) | pending |
| Answers citing a correct section | pending |

The full analysis is in **[FINDINGS.md](FINDINGS.md)**: why smaller chunks didn't help, why definition questions are the weak spot, and the limits of this test set.

## How to run it

You need **Python 3.10+**. An **Anthropic API key** is optional: without one, the app and scripts still find the matching sections of the Act, but Claude won't write answers.

**1. Install**
```bash
git clone https://github.com/Patience-Diana/Dpa_rag_assistant.git
cd Dpa_rag_assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API key (optional)**
```bash
cp .env.example .env
# then open .env and replace your-key-here with your key
```
`.env` is listed in `.gitignore`, so the key is never committed.

**3. Get the Act**

Download the PDF from the link under [Sources](#sources) and save it as
`data/raw/data_protection_act_2019.pdf`.

**4. Build the search index** (run once)
```bash
python src/ingest.py     # PDF -> data/processed/dpa_clean.txt
python src/chunk.py      # text -> data/processed/chunks_300.json
python src/index.py      # chunks -> chroma_db/
```
The cleaned text and chunks are already included in the repo, so you can skip straight to `index.py` if you like.

**5. Use it**
```bash
streamlit run app.py                                # web app at http://localhost:8501
python src/answer.py "Can I ask a company to delete my data?"   # command line
python src/retrieve.py "Can I ask a company to delete my data?" # search only, no key needed
```

**6. Evaluate**
```bash
python src/evaluate.py                    # retrieval (free)
python src/chunk.py --size 150 && python src/index.py --size 150
python src/evaluate.py --sizes 300 150    # compare chunk sizes
python src/evaluate.py --answers          # answer quality (uses the API, about $3)
```

## Limitations

- **One document only.** It knows the Act itself, not the regulations made under it, ODPC guidance or court decisions.
- **The Act as revised to 31 December 2022.** Later amendments are not included.
- **Definition questions are weaker.** Section 2 holds all the definitions in one long section, and retrieval sometimes misses it (see [FINDINGS.md](FINDINGS.md)).
- **The test questions were written by the author** while reading the Act, so the scores are probably somewhat optimistic.
- **Not legal advice.**

## Built with

Python · [sentence-transformers](https://www.sbert.net/) (`BAAI/bge-small-en-v1.5`) · [ChromaDB](https://www.trychroma.com/) · [Anthropic Claude API](https://docs.anthropic.com/) · [Streamlit](https://streamlit.io/) · pypdf

## Sources

Public documents only:
- **Data Protection Act, 2019 (Act No. 24 of 2019, Cap. 411C)**, revised to 31 December 2022.
  Published by Kenya Law and Laws.Africa: https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31
  (The PDF itself is not stored in this repo. Download it from the link above into `data/raw/`.)
