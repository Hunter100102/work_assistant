import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

app = Flask(__name__)
CORS(app)

# --- Configuration ---
PORT = int(os.environ.get("PORT", 10000))  # Render sets PORT at runtime
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "phi3:mini")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "index_storage")
DATA_DIR = os.environ.get("DATA_DIR", "data")

# NOTE: On Render, localhost:11434 will NOT work unless you're also running Ollama
# in the same container/process (not typical) or as another service with a reachable URL.
llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    request_timeout=500,
    context_window=2048,
    additional_kwargs={"num_ctx": 2048},
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build/load index (consider moving to lazy init if this is slow)
if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

query_engine = index.as_query_engine()
conversation_history = []

@app.get("/")
def health():
    return "OK", 200

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "")

    context = "\n".join([f"You: {m['user']}\nBot: {m['bot']}" for m in conversation_history[-5:]])
    full_prompt = f"{context}\nYou: {user_input}\nBot:"
    response = query_engine.query(full_prompt)

    conversation_history.append({"user": user_input, "bot": str(response)})
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=PORT)
