# backend

- Absolute imports (`backend.*`) for robust module resolution
- Logger writes to `<repo>/logs` (rotating)
- Pipelines share paths from `backend/config/config.py`
- Incremental embeddings → FAISS (atomic writes, retry)
- Metadata stores both absolute `path` and repo-relative `rel_path`
- No data/vectorstore included

## Install
```bash
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY
```

## Build vector store
Put files under `data/raw/<category>/...` then:
```bash
./pipelines/ingest_local.sh
# Windows:
# powershell -ExecutionPolicy Bypass -File pipelines/ingest_local.ps1
```

Artifacts:
```
vectorstore/dev/current/faiss_index.idx
vectorstore/dev/current/metadatas.json
```

## Run backend
```bash
python -m backend.app
# or: gunicorn backend.app:app -b 0.0.0.0:5000
```

Health: `GET /health`
RAG: `POST /api/ask-rag` with `{ "question": "...", "top_k": 5 }`

## Notes (FAISS on Windows)
- pip wheels for `faiss-cpu` are limited on native Windows.
- Prefer WSL2/Ubuntu or conda (`conda install -c pytorch faiss-cpu`), or switch to Chroma for dev.
