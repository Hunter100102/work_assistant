import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

app = FastAPI()

# Keep this permissive, or restrict to your site(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://automatingsolutions.com", "https://hunter100102.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512

@app.get("/healthz")
async def healthz():
    return {"ok": True}

_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def startup() -> None:
    global _client
    # Tight connection pool keeps memory small and responses snappy
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, limits=limits)

@app.on_event("shutdown")
async def shutdown() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None

@app.post("/chat")
async def chat(req: ChatRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    payload = {
        "model": req.model or OPENAI_MODEL,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        r = await _client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return {"reply": content}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))
