# RAG Pipeline — Week 2 Challenge 1

Production-ready RAG pipeline built on top of [Week 1 SLM Hosting](https://github.com/ntthai2/slm_hosting).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              RAG PIPELINE                            │
│                                                      │
│  OFFLINE:  PDFs → Docling → Chunks → Jina v3 →     │
│            Qdrant                                    │
│                                                      │
│  ONLINE:   Query → Embed → Qdrant Search →          │
│            Prompt → vLLM → Answer + Sources         │
│                                                      │
│  EVAL:     20 questions → RAGAS → Scores            │
└─────────────────────────────────────────────────────┘

External: vLLM (Week 1) ← RAG API calls via HTTP
Internal: Qdrant + Jina v3 in Docker Compose
```

---

## Tech Stack

| Component     | Choice                        | Why                                          |
|---------------|-------------------------------|----------------------------------------------|
| PDF Parsing   | Docling                       | Best for math formulas + images (97.9% table accuracy) |
| Chunking      | RecursiveCharacterTextSplitter| Fast, deterministic, 512 tokens + 50 overlap |
| Embedding     | Jina v3 (self-host, Infinity) | Best Vietnamese + multilingual, 1024-dim, free |
| Vector DB     | Qdrant v1.12                  | Rust-based, fast cosine search, easy Docker  |
| LLM           | Qwen2.5-1.5B-AWQ (Week 1)    | Reuse existing vLLM, saves VRAM on 8GB GPU   |
| Evaluation    | RAGAS                         | Industry standard, measures faithfulness + relevancy |

---

## Quick Start

### Prerequisites
- Start your Week 1 vLLM first:
  ```bash
  cd ../slm_hosting
  docker compose -f docker/docker-compose.yml up -d
  ```

### 1. Clone & configure
```bash
git clone <your-repo-url>
cd rag-project-weaction
cp .env.example .env
# Edit .env if needed (VLLM_URL, VLLM_API_KEY)
```

### 2. Add your PDFs
```bash
cp /path/to/your/*.pdf data/raw/
```

### 3. Start services
```bash
docker compose -f docker/docker-compose.yml up --build -d
# Wait ~2 min for Jina v3 to download and warm up
```

### 4. Ingest documents
```bash
curl -X POST http://localhost:8080/ingest
```

### 5. Query
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the definition of X?"}'
```

### 6. Health check
```bash
curl http://localhost:8080/health
```

### 7. Run evaluation
```bash
# First fill in eval/dataset.json with your 20 questions
pip install -r requirements.txt
python scripts/evaluate.py
```

---

## API

### `POST /query`
```json
// Request
{ "question": "What does the document say about X?" }

// Response
{
  "answer": "According to the document...",
  "sources": ["lecture1.pdf"],
  "context_used": 5
}
```

### `POST /ingest`
Ingests all PDFs from `data/raw/`.

### `GET /health`
```json
{
  "status": "ok",
  "services": {
    "qdrant": "ok",
    "embedding": "ok",
    "vllm": "ok"
  }
}
```

---

## RAGAS Evaluation Results

| Metric           | Score  | Target |
|------------------|--------|--------|
| Faithfulness     | _TBD_  | > 0.7  |
| Answer Relevancy | _TBD_  | > 0.7  |

_Run `python scripts/evaluate.py` and update this table._

---

## Project Structure

```
rag-project-weaction/
├── src/
│   ├── ingestion/       # Docling loader, chunker, embedder, Qdrant indexer
│   ├── retrieval/       # Vector search with similarity threshold
│   ├── generation/      # vLLM client + prompt builder
│   ├── evaluation/      # RAGAS runner
│   ├── api/             # FastAPI routes + schemas
│   └── core/            # Config, logging
├── config/
│   ├── prompts.yaml     # Prompt templates (not hardcoded)
│   └── settings.yaml    # Chunk size, top_k, thresholds
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── eval/
│   ├── dataset.json     # 20 eval questions
│   └── results/         # RAGAS output
├── data/raw/            # Put your PDFs here (gitignored)
└── scripts/
    ├── ingest.py
    └── evaluate.py
```
