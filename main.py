from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import OllamaLLM

# Load documents from the data folder
documents = SimpleDirectoryReader("data").load_data()

# Use local Ollama model
llm = OllamaLLM(model="mistral")

# Index documents
index = VectorStoreIndex.from_documents(documents)

# Query engine with local model
query_engine = index.as_query_engine(llm=llm)

# Chat loop
print("📎 Work Chatbot ready! Ask your question or type 'exit'")
while True:
    query = input("🔍 You: ")
    if query.lower() in ['exit', 'quit']:
        break
    response = query_engine.query(query)
    print("🤖 Bot:", response)
