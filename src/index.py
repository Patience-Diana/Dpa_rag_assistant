"""
index.py — Steps 4 and 5 of the pipeline: embed the chunks and store them.

What it does:
  1. Loads the chunks from data/processed/chunks.json (made by chunk.py)
  2. Turns each chunk's text into an embedding: a list of 384 numbers that
     captures its meaning. Texts with similar meaning get similar numbers.
  3. Saves every chunk (text + embedding + labels) into a local ChromaDB
     database in the chroma_db/ folder

It rebuilds the database from scratch every time, so running it twice
never creates duplicates.

Run it from the project root with:
    python src/index.py
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME, EMBEDDING_MODEL


def load_chunks():
    """Read the list of chunks that chunk.py saved."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts):
    """Turn a list of texts into a list of embeddings (vectors)."""
    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(texts)} chunks ...")
    # normalize_embeddings=True scales every vector to the same length.
    # Then comparing two vectors measures only their direction (meaning),
    # not their size. This is called "cosine similarity".
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True)
    return embeddings.tolist()  # ChromaDB wants plain Python lists


def build_collection(chunks, embeddings):
    """Create a fresh ChromaDB collection and add all chunks to it."""
    # PersistentClient saves the database to disk, so it is still there
    # after the script ends.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete the old collection if there is one, so we start clean.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    # "hnsw:space": "cosine" tells ChromaDB to compare vectors with cosine
    # distance, which matches the normalised embeddings we made above.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add everything in one go. Each list lines up by position:
    # ids[0], documents[0], embeddings[0] and metadatas[0] are the same chunk.
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "section_number": c["section_number"],
                "section_title": c["section_title"],
                "part": c["part"],
                "chunk_index": c["chunk_index"],
                "chunk_count": c["chunk_count"],
            }
            for c in chunks
        ],
    )
    return collection


def main():
    chunks = load_chunks()
    embeddings = embed_texts([c["text"] for c in chunks])
    collection = build_collection(chunks, embeddings)

    print(f"\nDone. Stored {collection.count()} chunks in '{COLLECTION_NAME}' "
          f"(folder: {CHROMA_DIR}/).")
    print(f"Each embedding has {len(embeddings[0])} numbers. "
          f"First 5 numbers of chunk 1: {[round(x, 3) for x in embeddings[0][:5]]}")


if __name__ == "__main__":
    main()
