import os, io, zipfile
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel
import numpy as np
from PyPDF2 import PdfReader

from vector_store import TinyVectorStore

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()
store = TinyVectorStore(STORAGE_DIR)

# ---- Helpers ----
def is_supported(name: str) -> bool:
    low = name.lower()
    return low.endswith((".txt", ".md", ".pdf"))

def read_text_from_file(name: str, data: bytes) -> str:
    low = name.lower()
    if low.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    elif low.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for p in reader.pages:
            parts.append(p.extract_text() or "")
        return "\n".join(parts)
    else:
        raise ValueError("Unsupported file type")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= L:
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    if not client:
        raise RuntimeError("OPENAI_API_KEY not configured")
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs)

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]

def build_context(q: str, hits: List[dict]) -> str:
    blocks = []
    used = 0
    for h in hits:
        t = h["text"]
        if used + len(t) > MAX_CONTEXT_CHARS:
            t = t[:max(0, MAX_CONTEXT_CHARS - used)]
        blocks.append(f"[Source: {h['source']} • chunk {h['chunk']}]\n{t}")
        used += len(t)
        if used >= MAX_CONTEXT_CHARS:
            break
    joined = "\n\n".join(blocks)
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

# ---- Schemas ----
class ChatIn(BaseModel):
    message: str

# ---- Routes ----
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/stats")
def stats():
    try:
        return {"ok": True, "chunks": len(store.meta), "storage_dir": STORAGE_DIR}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    try:
        added = 0
        seen_names = []

        for f in files:
            filename = f.filename or "upload"
            seen_names.append(filename)
            data = await f.read()

            # If ZIP, iterate members
            if filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.namelist() if is_supported(m)]
                    for m in members:
                        with zf.open(m) as zf_file:
                            blob = zf_file.read()
                        text = read_text_from_file(m, blob)
                        chunks = chunk_text(text)
                        if not chunks:
                            continue
                        metas, all_vecs = [], []
                        for i in range(0, len(chunks), 32):
                            batch = chunks[i:i+32]
                            vecs = embed_texts(batch)
                            all_vecs.append(vecs)
                            for j, t in enumerate(batch):
                                metas.append({"source": m, "chunk": i + j, "text": t})
                        if all_vecs:
                            emb = np.vstack(all_vecs)
                            store.add(emb, metas)
                            added += len(metas)
                continue

            # Regular file
            if not is_supported(filename):
                return JSONResponse({"ok": False, "error": "Unsupported file type. Upload .txt, .md, .pdf, or a .zip containing them."}, status_code=415)

            text = read_text_from_file(filename, data)
            chunks = chunk_text(text)
            if not chunks:
                continue
            metas, all_vecs = [], []
            for i in range(0, len(chunks), 32):
                batch = chunks[i:i+32]
                vecs = embed_texts(batch)
                all_vecs.append(vecs)
                for j, t in enumerate(batch):
                    metas.append({"source": filename, "chunk": i + j, "text": t})
            if all_vecs:
                emb = np.vstack(all_vecs)
                store.add(emb, metas)
                added += len(metas)

        return {"ok": True, "added": added, "files": seen_names}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/chat")
async def chat(payload: ChatIn):
    try:
        q = payload.message.strip()
        if not q:
            return {"ok": False, "error": "Empty message"}
        if client is None:
            return {"ok": False, "error": "OPENAI_API_KEY not set"}
        q_emb = embed_query(q)
        results = store.search(q_emb, top_k=TOP_K)
        metadatas = [m for (score, m) in results]
        if not metadatas:
            return {"ok": True, "reply": "I don't know.", "sources": []}
        prompt = build_context(q, metadatas)
        answer = chat_with_context(prompt)
        sources = [f"{m['source']}#{m['chunk']}" for m in metadatas]
        return {"ok": True, "reply": answer, "sources": sources}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
