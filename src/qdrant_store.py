from typing import List
import os
import math
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct


def connect_client(url: str, api_key: str | None = None, prefer_grpc: bool = False) -> QdrantClient:
    """Create a Qdrant client connected to the given URL.

    Example QDRANT_URL: "https://xyz-123.qdrant.cloud"
    """
    if url.startswith("http"):
        return QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
    # allow host:port shorthand
    return QdrantClient(host=url, api_key=api_key, prefer_grpc=prefer_grpc)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        params = VectorParams(size=vector_size, distance=distance)
        client.recreate_collection(collection_name=collection_name, vectors_config=params)


def collection_count(client: QdrantClient, collection_name: str) -> int:
    try:
        stats = client.get_collection(collection_name=collection_name)
        return stats.vectors_count or 0
    except Exception:
        return 0


def upload_chunks(client: QdrantClient, collection_name: str, chunks: List[dict], embed_fn, batch_size: int = 64):
    """Encode chunk texts and upload them to Qdrant as points with payload {"id","text"}.

    Points use the chunk 'id' as the point id to preserve mapping.
    """
    # compute vectors in batches
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]
        vecs = embed_fn(texts)
        points = []
        for c, v in zip(batch, vecs):
            pid = int(c.get("id", start))
            payload = {"text": c.get("text", ""), "chunk_id": pid}
            points.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))
        client.upsert(collection_name=collection_name, points=points)


def search(client: QdrantClient, collection_name: str, query: str, embed_fn, chunks: List[dict], top_k: int = 3):
    """Search Qdrant and return results in the same format as the FAISS-based `search`.
    If payload text is present it is used; otherwise the local `chunks` list is used to map ids.
    """
    qvec = embed_fn(query)
    # qvec may be (1, dim) or (dim,) depending on embedder
    try:
        qv = qvec[0].tolist()
    except Exception:
        # assume 1-d
        qv = qvec.tolist()

    hits = client.search(collection_name=collection_name, query_vector=qv, limit=top_k)
    results = []
    for hit in hits:
        score = float(hit.score) if hasattr(hit, "score") else 0.0
        payload = getattr(hit, "payload", {}) or {}
        text = payload.get("text")
        chunk_id = payload.get("chunk_id")
        if text is None and chunk_id is not None:
            # fallback to local chunks
            match = next((c for c in chunks if int(c.get("id", -1)) == int(chunk_id)), None)
            text = match.get("text") if match else ""
        results.append({"score": score, "text": text or "", "id": int(chunk_id) if chunk_id is not None else -1})
    return results
