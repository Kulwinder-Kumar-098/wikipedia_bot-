# Wikipedia RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot that fetches a Wikipedia article, cleans and chunks the content, creates embeddings, stores them in a FAISS index, and answers user questions using retrieved context plus an LLM.

## Features
- Fetches Wikipedia content using the Wikipedia API
- Cleans and preprocesses raw article text
- Splits the article into meaningful chunks
- Generates embeddings with sentence-transformers
- Stores and searches vectors using FAISS
- Answers questions with retrieved context through Groq

## Project Structure
- data/raw/article_raw.txt - Raw fetched Wikipedia article
- data/processed/article_clean.txt - Cleaned article text
- data/processed/chunks.json - Chunked article data
- data/embeddings/ - Embedding outputs and metadata
- index/wiki_index.faiss - FAISS vector index
- src/ - Core pipeline modules
  - fetch.py - Download Wikipedia article
  - clean.py - Clean raw text
  - chunk.py - Split text into chunks
  - embeddings.py - Generate embeddings
  - faiss_store.py - Build FAISS index
  - retreival.py - Retrieve relevant chunks
  - llm_intergration.py - Generate answers with Groq
  - wikipedia_chatbot.py - Main chatbot UI

## Requirements
Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Setup
Create a .env file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## How to Run
1. Fetch the article:
   ```bash
   python src/fetch.py
   ```
2. Clean the article:
   ```bash
   python src/clean.py
   ```
3. Create chunks:
   ```bash
   python src/chunk.py
   ```
4. Generate embeddings:
   ```bash
   python src/embeddings.py
   ```
5. Build the FAISS index:
   ```bash
   python src/faiss_store.py
   ```
6. Start the console chatbot:
   ```bash
   python src/wikipedia_chatbot.py
   ```
7. Start the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Deployment
- Install dependencies: `pip install -r requirements.txt`
- Add your `GROQ_API_KEY` to `.env`
- Deploy on Streamlit Cloud by pointing to the repo root, or deploy on Heroku using the included `Procfile`

## Notes
- The chatbot expects the FAISS index and chunk file to exist before startup.
- If the embedding model or LLM is unavailable, the pipeline may fail until the required dependencies and API credentials are configured.
- The project is designed as a simple demonstration of a local RAG pipeline.
