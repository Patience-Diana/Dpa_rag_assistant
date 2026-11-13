"""
retrieve.py — Step 6 of the pipeline: find the chunks that best match a question.

How it works:
  1. Turn the question into an embedding with the SAME model used in index.py
  2. Ask ChromaDB for the chunks whose embeddings are closest to it
  3. Return those chunks with their section labels and a similarity score

Other files (answer.py, evaluate.py, app.py) will import retrieve() from here.

Run it from the project root:
    python src/retrieve.py                          # type questions one by one
    python src/retrieve.py "Can I delete my data?"  # ask a single question
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

# The bge model works better when a short instruction is put in front of
# QUESTIONS (not in front of the chunks). This is recommended by its authors.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# How many chunks to return by default.
DEFAULT_TOP_K = 3

# Loading the model takes a few seconds, so we load it (and open the
# database) only once, the first time retrieve() is called, then reuse them.
_model = None
_collection = None


def _load():
    """Load the embedding model and open the ChromaDB collection (once)."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def retrieve(question, top_k=DEFAULT_TOP_K):
    """
    Return the top_k chunks most similar to the question.

    Each result is a dictionary like:
      {"chunk_id": "s26-c1", "section_number": "26",
       "section_title": "Rights of a data subject", "part": "Part IV – ...",
       "text": "Section 26: Rights of a data subject\n...", "score": 0.78}

    score is the cosine similarity: closer to 1 means a closer match.
    """
    model, collection = _load()

    # Embed the question the same way the chunks were embedded.
    query_embedding = model.encode(QUERY_INSTRUCTION + question,
                                   normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=[query_embedding],
                               n_results=top_k)

    # ChromaDB returns lists inside lists (one inner list per question).
    # We asked one question, so we take item [0] of each.
    chunks = []
    for chunk_id, text, meta, distance in zip(results["ids"][0],
                                              results["documents"][0],
                                              results["metadatas"][0],
                                              results["distances"][0]):
        chunks.append({
            "chunk_id": chunk_id,
            "section_number": meta["section_number"],
            "section_title": meta["section_title"],
            "part": meta["part"],
            "text": text,
            # ChromaDB gives a cosine DISTANCE (0 = identical).
            # Similarity = 1 - distance, which is easier to read.
            "score": round(1 - distance, 3),
        })
    return chunks


def print_results(question, chunks):
    """Print the results in a short, readable way."""
    print(f"\nQ: {question}")
    for rank, c in enumerate(chunks, start=1):
        # Show the first 120 characters of the chunk, without the label line.
        preview = c["text"].split("\n", 1)[-1][:120]
        print(f"  {rank}. [score {c['score']:.3f}] Section {c['section_number']}: "
              f"{c['section_title']}")
        print(f"       {preview}...")


def main():
    # If a question was typed after the script name, answer just that one.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print_results(question, retrieve(question))
        return

    # Otherwise, keep asking for questions until the user types nothing.
    print("Type a question about the Kenya Data Protection Act "
          "(press Enter on an empty line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        print_results(question, retrieve(question))


if __name__ == "__main__":
    main()
