"""Helper to upload local chunks to a Qdrant collection.

Usage:
  Set `QDRANT_URL`, `QDRANT_API_KEY` (if needed) and `QDRANT_COLLECTION` env vars, then:

    python -m src.qdrant_upload

This will encode all `data/processed/chunks.json` texts with the project's embedder
and upsert them into the collection.
"""
import os
import json
from pathlib import Path

from src.qdrant_store import connect_client, ensure_collection, upload_chunks
from src.retreival import load_embedder


def main():
    url = os.environ.get("QDRANT_URL")
    if not url:
        raise RuntimeError("Set QDRANT_URL env var to your Qdrant endpoint.")

    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("QDRANT_COLLECTION", "wiki")

    chunks_path = Path("data/processed/chunks.json")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embed_fn = load_embedder()

    # Determine vector size by encoding a single text
    sample = chunks[0]["text"] if chunks else ""
    vec = embed_fn(sample)
    try:
        dim = int(vec.shape[-1])
    except Exception:
        # fallback
        dim = len(vec[0]) if hasattr(vec, "__len__") and hasattr(vec[0], "__len__") else 1536

    client = connect_client(url, api_key)
    ensure_collection(client, collection, vector_size=dim)
    upload_chunks(client, collection, chunks, embed_fn)


if __name__ == "__main__":
    main()
