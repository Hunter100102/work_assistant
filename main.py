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

# SET UP YOUR LOCAL LLM
llm = Ollama(
    model="phi3:mini",
    base_url="http://localhost:11434",
    request_timeout=500,
    context_window=2048,
    additional_kwargs={"num_ctx": 2048},
)

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Print a quick test to confirm the LLM is reachable
print("LLM test: ", llm.complete("What is the company name?"))

# Define where you want to store the index
PERSIST_DIR = "index_storage"

if os.path.exists(PERSIST_DIR):
    # Load the index from disk (cache)
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ Loaded index from cache.")
else:
    # First-time build
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Built and saved index to cache.")

# Create a query engine from the index
query_engine = index.as_query_engine()

# Start interactive chatbot loop
print("🤖 Mercedes Work Assistant Ready! Type a question or 'exit'")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = query_engine.query(user_input)
    print("Bot:", response)
