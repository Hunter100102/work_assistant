import os
from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT: don't import heavy libs at module import time
# (do it lazily inside request handlers or on_startup)
app = FastAPI()

class ChatIn(BaseModel):
    message: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

# Lazy singletons
_llm_pipeline = {"ready": False, "obj": None, "err": None}

@app.on_event("startup")
def init_lazy():
    # Keep startup light to avoid OOM. Just verify we can import later.
    # Do not download NLTK or build indexes here.
    pass

def get_llm():
    """Lazy-load heavy stuff on first request."""
    if not _llm_pipeline["ready"] and _llm_pipeline["err"] is None:
        try:
            # Import here (lazy)
            # from llama_index.core import ...
            # Minimal example — replace with your real loader but keep memory small
            _llm_pipeline["obj"] = object()
            _llm_pipeline["ready"] = True
        except Exception as e:
            _llm_pipeline["err"] = str(e)
    return _llm_pipeline

@app.post("/chat")
def chat(payload: ChatIn):
    pipe = get_llm()
    if pipe["err"]:
        return {"ok": False, "error": pipe["err"]}
    # TODO: run your LlamaIndex / model inference here, but keep it lean
    reply = f"You said: {payload.message}"
    return {"ok": True, "reply": reply}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    # Bind to 0.0.0.0 so Render can reach it
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
