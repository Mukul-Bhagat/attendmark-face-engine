# AttendMark Face Engine Service

Internal Python service for face embeddings and class-scoped FAISS matching.

## Run locally

```bash
cd face_service
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8088
```

Render and Docker deploys are `PORT`-aware via:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8088}
```

## Required headers

If `FACE_ENGINE_API_KEY` (or `INTERNAL_API_KEY`) is set, every `/v1/*` request must include:

`x-internal-api-key: <key>`

## Endpoints

- `GET /healthz`
- `POST /v1/enroll`
- `POST /v1/match/class`
- `POST /v1/index/upsert`
- `POST /v1/index/remove`
- `POST /v1/quality-check`
- `POST /v1/index/warmup`
