import os
import faiss
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")
os.environ["OPENAI_API_KEY"] = api_key

DATA_DIR = "data"
INDEX_DIR = "index_storage"


def build_index():
    embed_model = OpenAIEmbedding()  # text-embedding-3-small by default
    Settings.embed_model = embed_model

    print("📄 Loading documents from", DATA_DIR)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # FAISS dimension = 1536 for OpenAI small embedding
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("💾 Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("📦 Persisting index to", INDEX_DIR)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("✅ Done! Index built successfully.")


if __name__ == "__main__":
    build_index()
