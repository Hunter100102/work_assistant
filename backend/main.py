# main.py
from flask import Flask, request, jsonify
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

llm = Ollama(
    model="phi3:mini",
    base_url="http://localhost:11434",
    request_timeout=500,
    context_window=2048,
    additional_kwargs={"num_ctx": 2048},
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

PERSIST_DIR = "index_storage"
if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

query_engine = index.as_query_engine()

# 🧠 Memory store for basic conversation history
conversation_history = []

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    global conversation_history

    # Basic memory simulation: prepend history to prompt
    context = "\n".join([f"You: {msg['user']}\nBot: {msg['bot']}" for msg in conversation_history[-5:]])
    full_prompt = f"{context}\nYou: {user_input}\nBot:"

    response = query_engine.query(full_prompt)
    conversation_history.append({"user": user_input, "bot": str(response)})
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    app.run(port=5000)
