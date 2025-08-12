# Minimal RAG Starter for Render (FastAPI)

This is a lightweight Retrieval-Augmented Generation (RAG) service designed to run on Render's free Web Service (512 MiB RAM). It lets you upload documents, builds embeddings using OpenAI, and answers questions grounded in your data.

## Features
- **/ingest**: Upload `.txt`, `.md`, or `.pdf` files. They are chunked and embedded.
- **/chat**: Ask questions; the service retrieves top chunks and sends them to the model as context.
- **/ (UI)**: Simple web UI for uploading files and chatting.
- **Low memory**: No large local models. Uses OpenAI APIs and a tiny NumPy-based store on disk.

## Quick Start (Local)
1. `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
4. `uvicorn main:app --reload`
5. Open http://127.0.0.1:8000/

## Deploy to Render
1. Create a **Web Service** from this repo.
2. In **Environment**:
   - Add `OPENAI_API_KEY` (from OpenAI)
   - Optionally `OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL`
3. Optional: Add a **Persistent Disk** (e.g., 1GB) mounted at `/opt/render/project/src/storage` so your index survives deploys.
4. Deploy. Render will run: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

## Endpoints
- `GET /healthz` → `{"ok": true}`
- `POST /ingest` (multipart form): field `files` accepts multiple files. Allowed: .txt, .md, .pdf
- `POST /chat` (JSON): `{ "message": "Your question" }`

## Notes
- By default, data & index are stored in `./storage`. On Render, that resolves inside your project. For persistence across deploys, add a disk and set env var `STORAGE_DIR=/opt/render/project/src/storage`.
- Keep your uploads small to avoid hitting memory limits. This design streams and chunks, but very large PDFs can still be heavy.
- If you need absolutely free inference/embeddings, swap the OpenAI calls for a hosted free endpoint you control; otherwise OpenAI will incur small usage cost.
