# CogniRoute(Agentic RAG Pipeline)

> A production-grade Retrieval-Augmented Generation system with intelligent routing, multi-source retrieval, hallucination detection, and cited answers.

---

## What is this?

Most RAG systems blindly retrieve every time a user asks something. This one **thinks first**.

The Agentic RAG Pipeline uses an LLM-powered router to decide *whether* to retrieve, *where* to retrieve from (vector DB, web, or SQL), and *whether the result is good enough* before generating an answer. If confidence is low, it re-routes automatically instead of hallucinating.

```
User query → Preprocess → Agent router → Retrieval → Rerank → Quality check → Cited answer
                                ↑__________________________|  (re-route if low confidence)
```

---

## Features

- **Intelligent routing** — Agent decides the best tool per query: vector DB, web search, SQL, or direct answer
- **Multi-source retrieval** — Pulls from internal documents, live internet, and structured databases
- **Query rewriting** — Rewrites vague user questions into precise search queries before retrieval
- **Cross-encoder reranking** — Re-scores retrieved chunks with a more accurate cross-encoder model
- **Hallucination guard** — Scores context confidence before generation; re-routes on failure
- **Session memory** — Maintains conversation history for coherent multi-turn interactions
- **Cited answers** — Every factual claim is linked to its source document or URL
- **Streaming responses** — Answers stream token-by-token via Server-Sent Events

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini API (`gemini-2.5-flash` / `gemini-2.5-pro`) |
| Agent framework | LangChain / LangGraph |
| Vector database | Chroma (local) / Pinecone (cloud) |
| Embeddings | Google Gemini `gemini-embedding-001` (3072 dims) |
| Reranking | Cohere Rerank v3 |
| Web search | Tavily API |
| Backend | FastAPI + Python 3.11+ |
| Memory | Redis |
| Frontend | Streamlit |
| Evaluation | RAGAS |

---

## Project Structure

```
agentic-rag/
├── ingestion/
│   ├── loader.py            # Document loading (PDF, Markdown, web)
│   ├── chunker.py           # Text splitting with overlap
│   └── embedder.py          # Embedding + vector store upload
│
├── agent/
│   ├── preprocessor.py      # Query cleaning, history injection, rewriting
│   ├── router.py            # LLM-powered tool selection
│   └── memory.py            # Redis session memory buffer
│
├── retrieval/
│   ├── vector_retriever.py  # Chroma / Pinecone MMR search
│   ├── web_retriever.py     # Tavily live web search
│   └── sql_retriever.py     # LLM-generated SQL + execution
│
├── context/
│   ├── reranker.py          # Cohere cross-encoder reranking
│   └── assembler.py         # Token budgeting + context formatting
│
├── quality/
│   └── grader.py            # LLM confidence scorer + re-route logic
│
├── generation/
│   └── generator.py         # Final answer generation with citations
│
├── api/
│   └── main.py              # FastAPI server + SSE streaming endpoint
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── evaluation/
│   └── ragas_eval.py        # RAGAS pipeline evaluation
│
├── tests/
│   ├── test_router.py
│   ├── test_retrieval.py
│   └── test_quality.py
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
GEMINI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here        # optional, Chroma works locally
REDIS_URL=redis://localhost:6379
```

### 3. Ingest your documents

```bash
python -m ingestion.embedder --source ./docs --vectorstore chroma
```

### 4. Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Launch the frontend

```bash
streamlit run frontend/app.py
```

---

## How It Works

### Stage 1 — Document ingestion (offline)

Documents are loaded, cleaned, split into overlapping chunks (512 tokens, 64-token overlap), embedded using Google Gemini embeddings (`gemini-embedding-001`), and stored in a vector database.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " "]
)
```

### Stage 2 — Query preprocessing

The raw user message is normalized, merged with conversation history, and rewritten by an LLM into a precise search query.

```python
# "what about the pricing?" + history
# → "What are the enterprise pricing tiers for Product X?"
```

### Stage 3 — Agent routing

An LLM call reads the query and selects the best tool:

| Decision | When |
|---|---|
| `vector_db` | Question about internal docs, policies, knowledge base |
| `web_search` | Needs current or real-time information |
| `sql_query` | Structured data — orders, users, metrics |
| `direct` | General knowledge, no retrieval needed |

### Stage 4 — Multi-source retrieval

The selected tool fetches raw content. Vector search uses Maximal Marginal Relevance (MMR) to balance relevance with diversity across returned chunks.

### Stage 5 — Reranking & assembly

A Cohere cross-encoder rescores each chunk against the query (more accurate than cosine similarity). Chunks are trimmed to a token budget and formatted with source labels.

### Stage 6 — Quality guard

An LLM grader scores whether the context is sufficient to answer the query. If score < 0.7, the pipeline re-routes to a different tool rather than generating a low-confidence answer.

```python
# Score < 0.7 → re-route to web_search
# Score ≥ 0.7 → proceed to generation
```

### Stage 7 — Response generation

The Google Gemini API generates a cited answer, streaming tokens back to the frontend. Session memory is updated for the next turn.

---

## API Reference

### `POST /chat`

Send a message and receive a streamed response.

**Request**
```json
{
  "message": "What is the refund policy for enterprise plans?",
  "session_id": "user-123"
}
```

**Response** (Server-Sent Events)
```
data: {"token": "Enterprise"}
data: {"token": " plans"}
data: {"token": " include"}
...
data: {"done": true, "sources": ["policy.pdf", "pricing.md"]}
```

### `POST /ingest`

Upload and index a new document.

```json
{
  "file_path": "./docs/new_policy.pdf",
  "metadata": { "category": "legal" }
}
```

### `GET /health`

Returns pipeline status and vector store document count.

---

## Evaluation

Run RAGAS evaluation to measure pipeline quality:

```bash
python evaluation/ragas_eval.py --questions ./evaluation/test_questions.json
```

This measures four metrics:

| Metric | What it measures | Target |
|---|---|---|
| Faithfulness | Does the answer stay grounded in context? | > 0.85 |
| Answer relevancy | Is the answer relevant to the question? | > 0.80 |
| Context precision | Are the retrieved chunks actually useful? | > 0.75 |
| Context recall | Did retrieval find all relevant info? | > 0.70 |

---

## Configuration

Key settings in `.env`:

```env
# Retrieval
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=5
MMR_LAMBDA=0.7          # 0 = max diversity, 1 = max relevance

# Quality guard
CONFIDENCE_THRESHOLD=0.7
MAX_REREROUTES=2

# Memory
MEMORY_WINDOW=5         # number of past turns to inject

# Generation
MAX_TOKENS=1024
STREAMING=true
```

---

## Roadmap

- [ ] Multi-tenant support with per-user vector namespaces
- [ ] Multimodal retrieval (images + text in PDFs)
- [ ] LangSmith tracing for full observability
- [ ] Docker + docker-compose for one-command deploy
- [ ] Evaluation dashboard with RAGAS metrics over time
- [ ] Support for additional vector stores (Weaviate, pgvector)

---

## Requirements

```
python>=3.11
google-genai
langchain>=0.2.0
langchain-community>=0.2.0
cohere>=5.5.0
tavily-python>=0.3.0
chromadb>=0.5.0
pinecone-client>=3.0.0
fastapi>=0.111.0
uvicorn>=0.30.0
streamlit>=1.35.0
redis>=5.0.0
ragas>=0.1.9
tiktoken>=0.7.0
pypdf>=4.2.0
python-dotenv>=1.0.0
pydantic>=2.7.0
```

---

## License

MIT — free to use, modify, and deploy.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Make sure to update tests and run the RAGAS evaluation suite before submitting.

---

*Built with Google Gemini API, LangChain, and a strong belief that AI systems should know when they don't know.*
