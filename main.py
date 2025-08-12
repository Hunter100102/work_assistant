
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn

# Load .env locally (Render uses its Environment tab)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown

app = FastAPI(lifespan=lifespan)

# You can remove CORS entirely when UI and API share the same origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

STATIC_DIR = "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Avoid a 404 for browsers that auto-request /favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    # 204 = No Content; avoids error noise in console
    return Response(status_code=204)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
@app.post("/chat/", response_model=ChatResponse)  # tolerate trailing slash
async def chat(req: ChatRequest):
    # TODO: call your real agent here (OpenAI/Ollama/etc.) using secrets from env
    # For now we just echo:
    return {"reply": f"echo: {req.message}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
