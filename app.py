import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.wikipedia_chatbot import load_pipeline, TOP_K
from src.retreival import search, load_embedder
from src.llm_intergration import ask_llm
import os
import json
from src.qdrant_store import connect_client as qdrant_connect, search as qdrant_search, collection_count

st.set_page_config(
    page_title="Wikipedia RAG Chatbot",
    page_icon="🧠",
    layout="wide",
)

st.title("Wikipedia RAG Chatbot Dashboard")
st.markdown(
    "Use the local RAG pipeline to ask questions about the stored Wikipedia article. "
    "The dashboard loads the FAISS index and chunk store, retrieves relevant context, "
    "and sends the grounded prompt to the LLM."
)

@st.cache_resource
def load_pipeline_resource():
    return load_pipeline()

# Decide whether to use Qdrant (managed) or local FAISS index
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "wiki")

use_qdrant = False
q_client = None
index = None
chunks = []
embed_fn = None

if QDRANT_URL:
    try:
        chunks_path = Path("data/processed/chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embed_fn = load_embedder()
        q_client = qdrant_connect(QDRANT_URL, QDRANT_API_KEY)
        use_qdrant = True
    except Exception as exc:
        st.error("Unable to initialize Qdrant pipeline — check QDRANT_URL and chunks file.")
        st.error(str(exc))
        st.stop()
else:
    pipeline = None
    try:
        pipeline = load_pipeline_resource()
    except Exception as exc:
        st.error("Unable to load the pipeline. Make sure the index and chunks are built first.")
        st.error(str(exc))
        st.stop()

    index, chunks, embed_fn = pipeline

with st.sidebar:
    st.header("Pipeline status")
    if use_qdrant:
        try:
            q_count = collection_count(q_client, QDRANT_COLLECTION)
        except Exception:
            q_count = "?"
        st.write(f"- Qdrant vectors: **{q_count}**")
    else:
        st.write(f"- Index vectors: **{index.ntotal}**")
    st.write(f"- Chunks loaded: **{len(chunks)}**")
    st.write(f"- Default top-k: **{TOP_K}**")

st.subheader("Article preview")
if len(chunks) > 0:
    st.write(chunks[0]["text"])
    with st.expander("Show more chunk previews"):
        for chunk in chunks[:5]:
            st.markdown(f"**Chunk {chunk['id']}**")
            st.write(chunk["text"])
else:
    st.warning("No chunks are available in the pipeline.")

question = st.text_input("Ask a question", value="What year was AI founded as an academic discipline?")
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=TOP_K)

if st.button("Get answer"):
    if not question.strip():
        st.warning("Enter a question before searching.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            if use_qdrant:
                retrieved = qdrant_search(q_client, QDRANT_COLLECTION, question, embed_fn, chunks, top_k=top_k)
            else:
                retrieved = search(question, index, chunks, embed_fn, top_k=top_k)

            if not retrieved:
                st.warning("No relevant chunks were found for this question.")
            else:
                st.subheader("Retrieved context")
                for result in retrieved:
                    st.markdown(f"**Chunk {result['id']}** — score: `{result['score']:.4f}`")
                    st.write(result["text"])

                answer = ask_llm(question, retrieved)
                st.subheader("Bot answer")
                st.write(answer)

                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {"question": question, "answer": answer})

if st.session_state.get("history"):
    st.subheader("Question history")
    for item in st.session_state.history[:5]:
        st.markdown(f"**Q:** {item['question']}")
        st.write(f"**A:** {item['answer']}")
