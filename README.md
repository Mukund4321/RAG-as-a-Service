# RAG-as-a-Service API

**A multi-tenant, production-grade Retrieval-Augmented Generation API with built-in evaluation and observability**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

---

## Overview

RAG-as-a-Service is a production-oriented evolution of a retrieval-augmented generation pipeline into a real, deployable, multi-tenant API — the kind of system an actual company could integrate against, not a notebook demo.

Any client can upload documents, get a scoped retrieval-backed chatbot, and query it through an authenticated REST API. Unlike most student RAG projects, this one is built with the engineering concerns that separate a prototype from a product: **authentication, per-tenant isolation, rate limiting, caching, retrieval quality evaluation, and observability.**

This project extends prior work (TNGOV — a LangChain + FAISS RAG pipeline over government documents) into a generalized, reusable service.

---

## Why This Project Exists

RAG is the most commonly demoed LLM pattern — and also the pattern most often shipped without rigor. Most RAG projects optimize for "does it answer the question" without ever measuring *how often* the retrieval step actually surfaces the right context, or what happens when 50 concurrent users hit the same API. This project treats RAG as a system to be engineered and evaluated, not a pipeline to be assembled once and demoed.

---

## Key Features

- **Multi-tenant architecture** — isolated document stores and chat scopes per API key/user
- **Document ingestion** — upload PDFs/text, automatic chunking + embedding
- **Scoped retrieval** — queries only search within a tenant's own document set
- **Authentication & rate limiting** — per-key auth, configurable request limits
- **Caching layer** — embedding cache + frequent-query cache to reduce redundant compute
- **Retrieval evaluation suite** — precision/recall/MRR measured against a labeled query set, not subjective "vibes"
- **Observability** — structured logging of latency, token cost, and retrieval quality per request
- **API documentation** — auto-generated via FastAPI/OpenAPI

---

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│   Client      │─────▶│   FastAPI Layer    │─────▶│   Auth & Rate         │
│  (API key)    │      │   (REST endpoints)  │      │   Limiter Middleware   │
└──────────────┘      └──────────────────┘      └──────────┬──────────┘
                                                              │
                       ┌──────────────────────────────────────┘
                       ▼
            ┌─────────────────────┐       ┌────────────────────┐
            │  Document Ingestion   │──────▶│   Vector Store       │
            │  (chunk + embed)       │      │   (per-tenant scoped, │
            └─────────────────────┘       │    FAISS-backed)       │
                                            └──────────┬──────────┘
                                                       │
                       ┌────────────────────────────────┘
                       ▼
            ┌─────────────────────┐       ┌────────────────────┐
            │  Retrieval + Cache    │──────▶│   LLM Generation     │
            │  Layer                  │      │   (with citations)    │
            └─────────────────────┘       └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌────────────────────┐
                                            │  Eval & Logging       │
                                            │  (precision/recall,   │
                                            │   latency, cost)       │
                                            └────────────────────┘
```

---

## Tech Stack

- **API framework**: FastAPI + Pydantic
- **Vector store**: FAISS (per-tenant scoped indices)
- **Embeddings**: sentence-transformers
- **LLM**: Anthropic / OpenAI API (configurable backend)
- **Caching**: Redis (or in-memory fallback for local dev)
- **Auth**: API key-based, JWT-ready
- **Database**: SQLite (dev) / PostgreSQL (production)
- **Deployment**: Docker, deployed on Render/Fly.io
- **Monitoring**: Structured JSON logging, optional Prometheus-compatible metrics endpoint

---

## Project Structure

```
rag-as-a-service/
├── app/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── auth/
│   │   ├── api_keys.py             # API key generation & validation
│   │   └── rate_limiter.py         # Request rate limiting middleware
│   ├── ingestion/
│   │   ├── document_loader.py      # PDF/text parsing
│   │   ├── chunker.py              # Chunking strategy
│   │   └── embedder.py             # Embedding generation + caching
│   ├── retrieval/
│   │   ├── vector_store.py         # Per-tenant FAISS index management
│   │   └── retriever.py            # Query → retrieved context
│   ├── generation/
│   │   └── generator.py            # LLM call + citation formatting
│   ├── eval/
│   │   ├── eval_set.py             # Labeled query/answer eval set
│   │   └── metrics.py              # Precision/recall/MRR calculations
│   └── observability/
│       └── logger.py               # Structured request logging
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_eval.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   └── api_reference.md            # Endpoint documentation
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/documents/upload` | Upload and ingest a document for a tenant |
| `GET` | `/v1/documents` | List ingested documents for the authenticated tenant |
| `DELETE` | `/v1/documents/{id}` | Remove a document and its embeddings |
| `POST` | `/v1/query` | Submit a query, receive a generated answer with citations |
| `GET` | `/v1/eval/run` | Trigger retrieval evaluation against the labeled eval set |
| `GET` | `/v1/health` | Service health check |

Full request/response schemas documented via auto-generated OpenAPI docs at `/docs` once deployed.

---

## Evaluation Methodology

Unlike most RAG demos, retrieval quality here is measured quantitatively:

1. A labeled evaluation set is built (queries paired with known-correct source chunks)
2. On each retrieval, the system checks whether the correct chunk appears in the top-k results
3. **Precision@k**, **Recall@k**, and **Mean Reciprocal Rank (MRR)** are computed and logged
4. Evaluation is re-run after any change to chunking strategy, embedding model, or retrieval parameters — so retrieval quality regressions are caught, not assumed away

---

## Results

> *To be populated once evaluation runs are complete.*

| Metric | Baseline (naive chunking) | Optimized (tuned chunking + caching) |
|---|---|---|
| Precision@5 | — | — |
| Recall@5 | — | — |
| MRR | — | — |
| Avg. query latency | — | — |
| Cache hit rate | — | — |

---

## Running Locally

```bash
# Clone and install
git clone https://github.com/Mukund4321/rag-as-a-service.git
cd rag-as-a-service
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add ANTHROPIC_API_KEY / OPENAI_API_KEY, etc.

# Run with Docker
docker-compose up --build

# Or run directly
uvicorn app.main:app --reload

# Run the evaluation suite
python -m app.eval.eval_set --run
```

API docs available at `http://localhost:8000/docs` once running.

---

## Deployment

Deployed via Docker on [Render / Fly.io — link to be added once live].

```bash
docker build -t rag-as-a-service .
docker run -p 8000:8000 --env-file .env rag-as-a-service
```

---

## Limitations

- FAISS indices are file-based; horizontal scaling would require migration to a managed vector DB (e.g., Pinecone, Weaviate) for multi-instance deployments
- Rate limiting is currently in-memory; production scaling would require a distributed limiter (Redis-backed)
- Evaluation set is manually curated and domain-specific; broader generalization claims would need a larger, more diverse eval set

---

## Roadmap

- [ ] Migrate to a managed vector DB for horizontal scaling
- [ ] Add streaming responses (SSE) for query endpoint
- [ ] Add usage-based billing simulation (tenant-level cost tracking)
- [ ] Expand eval set and add automated regression testing in CI
- [ ] Add webhook support for async document processing

---

## Author

**Mukund Saiteja**
Final-year CSE, SRM Institute of Science & Technology | ML Research Intern, IISc Bengaluru
[GitHub](https://github.com/Mukund4321) · ms9893@srmist.edu.in

---

## License

MIT License — see [LICENSE](LICENSE) for details.
