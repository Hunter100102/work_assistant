from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set the LLM to use the correct local model
llm = Ollama(
    model="phi3:mini",
    base_url="http://localhost:11434",
    request_timeout=500,
    context_window=2048,  # Cap the token window
    additional_kwargs={"num_ctx": 2048}
)
print(llm.complete("What is the company name?"))


# Use lightweight HuggingFace embedding model
Settings.llm = llm  # Ensure Settings uses it globally
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load work documents from the local folder
documents = SimpleDirectoryReader("data").load_data()

# Build a vector index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine from the index
query_engine = index.as_query_engine()

# Start interactive chatbot
print("🤖 Mercedes Work Assistant Ready! Type a question or 'exit'")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    response = query_engine.query(user_input)
    print("Bot:", response)
