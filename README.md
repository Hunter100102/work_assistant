# RAG Starter (Render-ready)

This is a super-light Retrieval-Augmented Generation starter for Render.

## Endpoints
- `GET /` — simple UI to upload and chat
- `POST /ingest` — upload `.txt/.md/.pdf` or a `.zip` containing them
- `POST /chat` — ask a question; answers grounded in your data
- `GET /stats` — check number of chunks and storage path
- `GET /healthz` — health check
- `GET /favicon.ico` — returns 204 (no icon)

## Deploy (Render)
1. Create a new **Web Service** from this repo.
2. Set env vars:
   - `OPENAI_API_KEY` (required)
   - Optional: `OPENAI_CHAT_MODEL=gpt-4o-mini`, `OPENAI_EMBED_MODEL=text-embedding-3-small`
3. Add a **Persistent Disk** mounted at `/opt/render/project/src/storage` and set `STORAGE_DIR` to same path.
4. Start command (already in Procfile): `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Local dev
```
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your key
uvicorn main:app --reload
```

## Notes
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims) — inexpensive and good enough.
- Chat: `gpt-4o-mini` by default; change via env var.
- PDFs that are scans may contain no extractable text; try OCR'd PDFs or plain `.txt/.md` first.
