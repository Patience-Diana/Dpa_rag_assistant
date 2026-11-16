"""
index.py — Steps 4 and 5 of the pipeline: embed the chunks and store them.

What it does:
  1. Loads the chunks from data/processed/chunks_<size>.json (made by chunk.py)
  2. Turns each chunk's text into an embedding: a list of 384 numbers that
     captures its meaning. Texts with similar meaning get similar numbers.
  3. Saves every chunk (text + embedding + labels) into a local ChromaDB
     database in the chroma_db/ folder

Each chunk size gets its own collection (e.g. "dpa_chunks_300"), and that
collection is rebuilt from scratch every time, so running it twice never
creates duplicates.

Run it from the project root with:
    python src/index.py               # index the 300-word chunks (default)
    python src/index.py --size 150    # index the 150-word chunks
"""

import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import (CHROMA_DIR, DEFAULT_CHUNK_SIZE, EMBEDDING_MODEL,
                    chunks_path, collection_name)


def load_chunks(chunk_size):
    """Read the list of chunks that chunk.py saved for this chunk size."""
    with open(chunks_path(chunk_size), encoding="utf-8") as f:
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


def build_collection(chunks, embeddings, name):
    """Create a fresh ChromaDB collection called `name` and add all chunks to it."""
    # PersistentClient saves the database to disk, so it is still there
    # after the script ends.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete the old collection if there is one, so we start clean.
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)

    # "hnsw:space": "cosine" tells ChromaDB to compare vectors with cosine
    # distance, which matches the normalised embeddings we made above.
    collection = client.create_collection(
        name=name,
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
    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB.")
    parser.add_argument("--size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="which chunk size to index (default 300)")
    size = parser.parse_args().size
    name = collection_name(size)

    chunks = load_chunks(size)
    embeddings = embed_texts([c["text"] for c in chunks])
    collection = build_collection(chunks, embeddings, name)

    print(f"\nDone. Stored {collection.count()} chunks in '{name}' "
          f"(folder: {CHROMA_DIR}/).")
    print(f"Each embedding has {len(embeddings[0])} numbers. "
          f"First 5 numbers of chunk 1: {[round(x, 3) for x in embeddings[0][:5]]}")


if __name__ == "__main__":
    main()
