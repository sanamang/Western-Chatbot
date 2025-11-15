import os
from dotenv import load_dotenv

import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


DATA_DIR = "data"

# --- Load API key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")
os.environ["OPENAI_API_KEY"] = api_key

# --- Configure global models ---
Settings.embed_model = OpenAIEmbedding()
Settings.llm = OpenAI(model="gpt-4o-mini")  # or "gpt-4o"


def build_index():
    """
    Build a FAISS-backed index in memory from all files in ./data.
    This avoids loading from disk and the UnicodeDecodeError.
    """
    print("📄 Loading documents from", DATA_DIR)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # FAISS dimension for OpenAI embeddings (text-embedding-3-small)
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)

    from llama_index.vector_stores.faiss import FaissVectorStore

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("💾 Building in-memory FAISS index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    print("✅ Index ready in memory.")
    return index


def chat_loop():
    print("\n🤖 Western University AI Chatbot")
    print("Type 'exit' to quit.\n")

    index = build_index()
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        response = query_engine.query(user_input)
        print("\nAI:", str(response), "\n")


if __name__ == "__main__":
    chat_loop()
