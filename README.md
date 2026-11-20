# Enterprise HR & Policy Navigator with Evaluated Hybrid Search

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to answer complex employee HR and policy queries with granular citations (document name, page number, and policy version), equipped with strict refusal fallback (`"I cannot answer based on internal documentation."`) when the internal context is insufficient.

## Project Architecture

- **Backend**: FastAPI (Python 3.11 / typed, documented modules)
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: Persistent ChromaDB
- **Keyword Search**: BM25 (`rank_bm25`)
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Local LLM**: Ollama (`qwen2.5:7b-instruct`)
- **Frontend**: React + Vite + Tailwind CSS

---

## Directory Structure

```text
HR-Policy-Navigator-RAG/
├── backend/
│   ├── data/
│   │   ├── documents/       # Source policy PDFs
│   │   └── vectorstore/     # Persistent ChromaDB store
│   ├── embeddings/          # Dense vector representations
│   ├── evaluation/          # Retrieval & generation benchmark metrics
│   ├── generation/          # Context grounding, prompt templates, Ollama client
│   ├── ingestion/           # PDF parsing, metadata extraction, chunking
│   ├── mlops/               # Query logging, latency tracking, telemetry
│   ├── models/              # Pydantic schemas and internal models
│   ├── retrieval/           # BM25, ChromaDB retrieval, RRF, Cross-Encoder reranker
│   ├── config.py            # Centralized settings and environment variables
│   └── main.py              # FastAPI application server
├── docs/                    # Architecture notes and documentation
├── frontend/                # React + Vite + Tailwind frontend application
├── scripts/
│   └── make_sample_docs.py  # Fictional HR policy document generator
├── tests/
│   └── test_health.py       # API and system test suite
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt         # Pinned Python package dependencies
└── README.md
```

---

## Quickstart

### 1. Setup Virtual Environment & Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

### 3. Generate Sample Documents

Generate all 7 fictional HR policies (including multi-version leaves, remote overrides, messy layouts, and scanned bitmap copies):
```powershell
python -m scripts.make_sample_docs
```

### 4. Run Tests

```powershell
pytest
```

### 5. Launch FastAPI Backend Server

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
API documentation is accessible at `http://127.0.0.1:8000/docs`.

### Policy API

- `POST /documents/upload` — upload and index a `.pdf` or `.md` policy (maximum 20 MB).
- `GET /documents` — list stored policy files.
- `DELETE /documents/{id}` — remove a policy and its indexed chunks.
- `POST /ask` — retrieve policy passages and return a cited answer.
- `GET /health` — check API health.

Example question request:

```json
{
  "query": "How many paid annual leave days do employees get?",
  "mode": "hybrid_rerank"
}
```

The answer response includes `answer`, document/page/section `citations`, the retrieved
chunks and relevance scores, the selected mode, request latency, and a `refused` flag.
The API uses the configured local Ollama model and returns the fixed refusal when no
relevant reranked evidence is available.
