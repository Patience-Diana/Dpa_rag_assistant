"""
config.py — settings shared by several files.

Keeping them in ONE place means index.py (which builds the database) and
retrieve.py (which searches it) can never disagree, for example by using
two different embedding models.
"""

from pathlib import Path

# The chunks made by chunk.py.
CHUNKS_PATH = Path("data/processed/chunks.json")

# The embedding model (downloaded automatically the first time, ~130 MB).
# bge-small reads up to ~380 words per text, so our 300-word chunks fit.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Where ChromaDB saves its files (this folder is in .gitignore).
CHROMA_DIR = "chroma_db"

# A ChromaDB "collection" is like a table that holds our chunks.
COLLECTION_NAME = "dpa_chunks"
