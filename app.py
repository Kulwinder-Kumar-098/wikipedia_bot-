import sys
import os
import json
import time
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

# Ensure src/ is on the path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.wikipedia_chatbot import load_pipeline, TOP_K
from src.retreival import search, load_embedder
from src.llm_intergration import ask_llm
from src.qdrant_store import connect_client as qdrant_connect, search as qdrant_search, collection_count

# Page Configuration
st.set_page_config(
    page_title="Wiki RAG Dashboard",
    page_icon=":material/psychology:",
    layout="wide",
)

# Custom Style Rules for Sleek Dark Theme
st.markdown(
    """
    <style>
    /* Styling headers and custom badges */
    .title-gradient {
        background: linear-gradient(90deg, #A78BFA 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .badge-qdrant {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-faiss {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    /* Card borders and hover animations */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 15px;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #8B5CF6;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cache Resources
@st.cache_resource
def load_faiss_pipeline():
    try:
        return load_pipeline()
    except Exception:
        return None

@st.cache_resource
def load_qdrant_client(url, api_key):
    try:
        return qdrant_connect(url, api_key)
    except Exception:
        return None

@st.cache_data(ttl="10s")
def get_qdrant_count(_client, collection_name):
    if _client is None:
        return 0
    return collection_count(_client, collection_name)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latencies" not in st.session_state:
    st.session_state.latencies = [0.42, 0.38, 0.51, 0.45]  # baseline data

# Load Configurations
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "wiki")

# Load Pipelines
faiss_pipeline = load_faiss_pipeline()
has_faiss = faiss_pipeline is not None

q_client = None
has_qdrant = False
if QDRANT_URL:
    q_client = load_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    has_qdrant = q_client is not None

# Load Chunks
chunks = []
embed_fn = None
if has_faiss:
    _, chunks, embed_fn = faiss_pipeline
elif has_qdrant:
    try:
        chunks_path = Path("data/processed/chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embed_fn = load_embedder()
    except Exception:
        st.error("Failed to load local chunks for Qdrant mode.")

# Sidebar Configuration
with st.sidebar:
    st.markdown("### :material/settings: Configuration")
    
    # DB Engine Selector
    available_dbs = []
    if has_faiss:
        available_dbs.append("FAISS (Local)")
    if has_qdrant:
        available_dbs.append("Qdrant (Cloud)")
        
    if not available_dbs:
        st.error("No database engines are available. Please check FAISS index or Qdrant keys.")
        st.stop()
        
    # Set default choice
    default_db = "Qdrant (Cloud)" if has_qdrant else "FAISS (Local)"
    db_choice = st.segmented_control(
        "Vector Engine",
        options=available_dbs,
        default=default_db,
    )
    
    use_qdrant = db_choice == "Qdrant (Cloud)"
    
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    
    # RAG Parameters
    top_k = st.slider("Number of Chunks (K)", min_value=1, max_value=10, value=TOP_K)
    show_sources = st.toggle("Show Grounded Context", value=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    # API Status
    st.markdown("### :material/info: Environment Status")
    
    # Groq status
    if os.environ.get("GROQ_API_KEY"):
        st.badge("Groq LLM Connected", icon=":material/check_circle:", color="green")
    else:
        st.badge("Groq Key Missing", icon=":material/error:", color="red")
        
    # Vector store status
    if use_qdrant:
        if q_client:
            st.badge("Qdrant Cloud Connected", icon=":material/cloud_done:", color="green")
        else:
            st.badge("Qdrant Connection Failed", icon=":material/cloud_off:", color="red")
    else:
        if has_faiss:
            st.badge("FAISS Index Ready", icon=":material/folder_zip:", color="green")
        else:
            st.badge("FAISS Index Missing", icon=":material/error:", color="red")
            
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Clear history button
    if st.button("Clear Chat History", icon=":material/delete:", type="secondary", width="stretch"):
        st.session_state.messages = []
        st.session_state.latencies = [0.42, 0.38, 0.51, 0.45]
        st.toast("Chat history cleared!", icon=":material/delete_sweep:")
        st.rerun()

# Main Layout
# Header with gradient title
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 25px;">
        <span style="font-size: 38px;">🧠</span>
        <div>
            <h1 style="margin: 0; font-size: 32px;" class="title-gradient">
                Wikipedia RAG Chatbot Dashboard
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 14px;">
                Ask natural language questions grounded in your custom Wikipedia knowledge base.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI row
with st.container(horizontal=True):
    # Determine stats
    if use_qdrant and q_client:
        vectors_count = get_qdrant_count(q_client, QDRANT_COLLECTION)
        db_label = "Qdrant Cloud"
    else:
        vectors_count = faiss_pipeline[0].ntotal if has_faiss else 0
        db_label = "FAISS (Local)"
        
    st.metric(
        label="Knowledge Source",
        value="Artificial Intelligence",
        border=True,
    )
    st.metric(
        label="Active Vector DB",
        value=db_label,
        border=True,
    )
    st.metric(
        label="Indexed Chunks",
        value=f"{vectors_count} Chunks",
        border=True,
    )
    
    # Display the latency trends as a sparkline!
    latest_lat = st.session_state.latencies[-1]
    st.metric(
        label="Query Latency",
        value=f"{latest_lat:.2f}s",
        border=True,
        chart_data=st.session_state.latencies,
        chart_type="line",
    )

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Grid Layout: Left Column = Chat, Right Column = Insights & Preview
col_chat, col_insights = st.columns([2, 1], gap="medium")

# Left Column - Chat Interface
with col_chat:
    with st.container(border=True, height=580):
        st.markdown("##### :material/chat: Conversation")
        
        # Suggestion chips
        SUGGESTIONS = {
            ":material/help: When was AI founded?": "What year was AI founded as an academic discipline?",
            ":material/person: Father of AI?": "Who is regarded as the father of Artificial Intelligence?",
            ":material/explore: Main subfields?": "What are the main branches or subfields of AI?",
        }
        
        # We define a variable to hold prompt from user
        user_prompt = None
        
        if not st.session_state.messages:
            st.write("Welcome! Ask a question to get started, or select a quick prompt:")
            selected = st.pills("Try asking:", list(SUGGESTIONS.keys()), label_visibility="collapsed")
            if selected:
                user_prompt = SUGGESTIONS[selected]
        
        # Display existing messages
        for msg in st.session_state.messages:
            avatar = ":material/person:" if msg["role"] == "user" else ":material/robot:"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if show_sources and "sources" in msg and msg["sources"]:
                    with st.expander("🔍 View grounding sources", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f"<div style='margin-bottom: 8px;'>"
                                f"<span class='badge-qdrant' style='font-family: monospace;'>ID: {src['id']}</span> "
                                f"<span class='badge-faiss' style='font-family: monospace;'>Similarity Score: {src['score']:.4f}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            st.caption(src["text"])
                            
        # Bottom input bar
        chat_prompt = st.chat_input("Type your question about AI here...")
        if chat_prompt:
            user_prompt = chat_prompt
            
        # Process new query
        if user_prompt:
            # 1. Add user message to state
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            
            # Show typing effect or loading spinner
            with st.spinner("Analyzing context and generating answer..."):
                start_time = time.time()
                
                # 2. Retrieve relevant context
                if use_qdrant:
                    retrieved = qdrant_search(q_client, QDRANT_COLLECTION, user_prompt, embed_fn, chunks, top_k=top_k)
                else:
                    retrieved = search(user_prompt, faiss_pipeline[0], chunks, embed_fn, top_k=top_k)
                
                # 3. Generate answer grounded in context
                if not retrieved:
                    answer = "I'm sorry, I couldn't find any relevant context in the database to answer that question."
                else:
                    answer = ask_llm(user_prompt, retrieved)
                
                # 4. Measure and save query latency
                duration = time.time() - start_time
                st.session_state.latencies.append(duration)
                if len(st.session_state.latencies) > 10:
                    st.session_state.latencies.pop(0)
                    
            # 5. Append response and trigger refresh
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved
            })
            st.rerun()

# Right Column - Insights & Raw Document Exploration
with col_insights:
    # 1. Document Statistics
    with st.container(border=True):
        st.markdown("##### :material/bar_chart: Knowledge Base Stats")
        if chunks:
            lengths = [len(c["text"]) for c in chunks]
            avg_len = int(np.mean(lengths))
            max_len = max(lengths)
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.markdown(f"**Avg Chunk Size:**<br>`{avg_len} chars`", unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f"**Max Chunk Size:**<br>`{max_len} chars`", unsafe_allow_html=True)
                
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("**Character Length by Chunk ID:**")
            
            # Create a dataframe of chunk lengths for plotting
            df_lengths = pd.DataFrame({
                "Chunk ID": [c["id"] for c in chunks],
                "Length": lengths
            })
            st.bar_chart(df_lengths, x="Chunk ID", y="Length", height=150)
        else:
            st.warning("No statistical data available.")
            
    # 2. Raw Document Explorer
    with st.container(border=True):
        st.markdown("##### :material/description: Document Explorer")
        if chunks:
            selected_chunk_id = st.number_input(
                "Select Chunk ID to inspect",
                min_value=0,
                max_value=len(chunks)-1,
                value=0,
                step=1
            )
            st.markdown(f"**Chunk {selected_chunk_id} Content:**")
            st.info(chunks[selected_chunk_id]["text"])
        else:
            st.warning("No chunks to explore.")
