import sys
import os
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Ensure src/ is on the path
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.retreival import load_embedder
from src.llm_intergration import ask_llm
from src.qdrant_store import (
    connect_client as qdrant_connect,
    search as qdrant_search,
    collection_count,
)

app = Flask(__name__)

# ─── Qdrant Configuration ────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "wiki")

if not QDRANT_URL:
    raise RuntimeError("QDRANT_URL environment variable is not set. Check your .env file.")

# Eagerly initialise clients once at startup (no per-request overhead)
q_client = qdrant_connect(QDRANT_URL, QDRANT_API_KEY)
embed_fn = load_embedder()

# Load local chunks file for chunk explorer (optional – non-fatal if missing)
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        CHUNKS: list[dict] = json.load(f)
except FileNotFoundError:
    CHUNKS = []


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/stats")
def stats():
    """Return live pipeline statistics for the KPI cards."""
    try:
        vec_count = collection_count(q_client, QDRANT_COLLECTION)
    except Exception:
        vec_count = 0

    return jsonify({
        "collection": QDRANT_COLLECTION,
        "vectors": vec_count,
        "chunks": len(CHUNKS),
        "model": "llama-3.1-8b-instant",
        "embedding": "all-MiniLM-L6-v2",
    })


@app.route("/api/chunks")
def chunks_data():
    """Return chunk list for the document explorer."""
    page = int(request.args.get("page", 0))
    size = int(request.args.get("size", 10))
    start = page * size
    end = start + size
    subset = CHUNKS[start:end]
    return jsonify({
        "total": len(CHUNKS),
        "page": page,
        "chunks": subset,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """RAG query endpoint – retrieve context from Qdrant then call the LLM."""
    payload = request.get_json(force=True)
    question = (payload.get("question") or "").strip()
    top_k = int(payload.get("top_k", 3))

    if not question:
        return jsonify({"error": "Question must not be empty."}), 400

    t_start = time.perf_counter()

    try:
        retrieved = qdrant_search(
            q_client, QDRANT_COLLECTION, question, embed_fn, CHUNKS, top_k=top_k
        )
    except Exception as exc:
        return jsonify({"error": f"Qdrant search failed: {exc}"}), 500

    if not retrieved:
        return jsonify({
            "answer": "I couldn't find relevant context for your question in the knowledge base.",
            "sources": [],
            "latency": round(time.perf_counter() - t_start, 3),
        })

    try:
        answer = ask_llm(question, retrieved)
    except Exception as exc:
        return jsonify({"error": f"LLM call failed: {exc}"}), 500

    latency = round(time.perf_counter() - t_start, 3)

    return jsonify({
        "answer": answer,
        "sources": retrieved,
        "latency": latency,
    })


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
