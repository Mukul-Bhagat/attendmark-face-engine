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
uvicorn app.main:app --host ${FACE_ENGINE_LISTEN_HOST:-0.0.0.0} --port ${PORT:-8088}
```

## Startup behavior

The service now starts the HTTP server immediately and initializes InsightFace in the background (default mode).
During model initialization, `GET /healthz` returns `503` with model state details until the engine is ready.
If constructor arguments are invalid (for example unsupported `allowed_modules`), startup now fails fast with an explicit model error instead of silently dropping options.

`GET /livez` always returns `200` when the process is up. Use this endpoint for platform liveness checks (for example Railway deployment healthchecks), while keeping `/healthz` as the strict readiness check.

## Low-memory tuning (Render-friendly defaults)

These env vars are optional:

- `FACE_MODEL_NAME` (default: `buffalo_s`)
- `FACE_MODEL_DET_SIZE` (default: `320`)
- `FACE_MODEL_ALLOWED_MODULES` (default: `detection,recognition`)
- `FACE_MODEL_INIT_MODE` (default: `background`, set `lazy` for on-demand init)
- `OMP_NUM_THREADS` (recommended: `1`)
- `OPENBLAS_NUM_THREADS` (recommended: `1`)
- `MKL_NUM_THREADS` (recommended: `1`)
- `NUMEXPR_NUM_THREADS` (recommended: `1`)

## Required headers

If `FACE_ENGINE_API_KEY` (or `INTERNAL_API_KEY`) is set, every `/v1/*` request must include:

`x-internal-api-key: <key>`

## Endpoints

- `GET /`
- `GET /livez`
- `GET /healthz`
- `POST /v1/enroll`
- `POST /v1/match/class`
- `POST /v1/index/upsert`
- `POST /v1/index/remove`
- `POST /v1/quality-check`
- `POST /v1/index/warmup`

## Railway quick setup

1. Generate a public domain for the service in Railway:
   - Service Settings -> Networking -> Public Networking -> `Generate Domain`
2. Set Railway healthcheck path to `/livez`.
3. Keep backend pointed to this URL:
   - `FACE_ENGINE_BASE_URL=https://attendmark-face-engine-production.up.railway.app`
4. Keep auth keys identical in backend and face engine:
   - backend: `INTERNAL_API_KEY` and/or `FACE_ENGINE_API_KEY`
   - face engine: `INTERNAL_API_KEY` and/or `FACE_ENGINE_API_KEY`
5. If private-network connectivity in older Railway environments is flaky, set:
   - `FACE_ENGINE_LISTEN_HOST=::`
