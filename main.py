import os, io, json, math
from typing import List
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
import numpy as np
from PyPDF2 import PdfReader

from vector_store import TinyVectorStore

# --- Config from env ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# Optional: delay import until runtime to keep startup light
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()
store = TinyVectorStore(STORAGE_DIR)

# --- Utilities ---

def read_text_from_file(name: str, data: bytes) -> str:
    lower = name.lower()
    if lower.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    elif lower.endswith(".pdf"):
        # Light PDF text extraction
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
        return "\n".join(parts)
    else:
        raise ValueError("Unsupported file type. Please upload .txt, .md, or .pdf")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(texts: List[str]) -> np.ndarray:
    if not client:
        raise RuntimeError("OPENAI_API_KEY not configured")
    # OpenAI returns 1536-dim for text-embedding-3-small
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs)

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]

def build_context(q: str, hits: List[dict]) -> str:
    ctx = []
    total = 0
    for h in hits:
        t = h["text"]
        if total + len(t) > MAX_CONTEXT_CHARS:
            t = t[:max(0, MAX_CONTEXT_CHARS - total)]
        ctx.append(f"[Source: {h['source']} • chunk {h['chunk']}]\n{t}")
        total += len(t)
        if total >= MAX_CONTEXT_CHARS:
            break
    joined = "\n\n".join(ctx)
    return f"You are a helpful assistant. Use ONLY the provided sources. If the answer isn't in the sources, say you don't know.\n\nQUESTION: {q}\n\nSOURCES:\n{joined}"

def chat_with_context(prompt: str) -> str:
    if not client:
        raise RuntimeError("OPENAI_API_KEY not configured")
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You answer using only the given context. If unsure, say you don't know."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# --- Schemas ---
class ChatIn(BaseModel):
    message: str

# --- Routes ---

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    # Serve the simple UI
    return FileResponse("static/index.html")

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    added = 0
    file_names = []
    for f in files:
        file_names.append(f.filename)
        data = await f.read()
        try:
            text = read_text_from_file(f.filename, data)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        # embed in batches to avoid large payloads
        batch = 32
        metas = []
        all_vecs = []
        for i in range(0, len(chunks), batch):
            batch_texts = chunks[i:i+batch]
            vecs = embed_texts(batch_texts)
            all_vecs.append(vecs)
            for j, t in enumerate(batch_texts):
                metas.append({
                    "source": f.filename,
                    "chunk": i + j,
                    "text": t
                })
        if all_vecs:
            emb = np.vstack(all_vecs)
            store.add(emb, metas)
            added += len(metas)
    return {"ok": True, "added": added, "files": file_names}

@app.post("/chat")
async def chat(payload: ChatIn):
    q = payload.message.strip()
    if not q:
        return {"ok": False, "error": "Empty message"}
    if client is None:
        return {"ok": False, "error": "OPENAI_API_KEY not set"}
    q_emb = embed_query(q)
    results = store.search(q_emb, top_k=TOP_K)
    metadatas = [m for (score, m) in results]
    prompt = build_context(q, metadatas)
    answer = chat_with_context(prompt)
    sources = []
    for m in metadatas:
        sources.append(f"{m['source']}#{m['chunk']}")
    return {"ok": True, "reply": answer, "sources": sources}
