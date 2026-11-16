"""
config.py — settings shared by several files.

Keeping them in ONE place means index.py (which builds the database) and
retrieve.py (which searches it) can never disagree, for example by using
two different embedding models.
"""

from pathlib import Path

# The embedding model (downloaded automatically the first time, ~130 MB).
# bge-small reads up to ~380 words per text, so our 300-word chunks fit.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Where ChromaDB saves its files (this folder is in .gitignore).
CHROMA_DIR = "chroma_db"

# The chunk size (in words) used unless we ask for a different one.
# On Day 4 we compare this with a smaller size, e.g. 150.
DEFAULT_CHUNK_SIZE = 300


def default_overlap(chunk_size):
    """Overlap = about 1/6 of the chunk size (300 -> 50 words, 150 -> 25 words)."""
    return chunk_size // 6


def chunks_path(chunk_size):
    """Each chunk size gets its own file, e.g. data/processed/chunks_300.json."""
    return Path(f"data/processed/chunks_{chunk_size}.json")


def collection_name(chunk_size):
    """
    Each chunk size gets its own ChromaDB collection, e.g. "dpa_chunks_300".
    (A collection is like a table that holds the chunks.) Keeping them separate
    lets us compare chunk sizes without rebuilding the database each time.
    """
    return f"dpa_chunks_{chunk_size}"
