import os

from dotenv import load_dotenv
import streamlit as st
import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore


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


@st.cache_resource(show_spinner=True)
def build_index():
    """
    Build a FAISS-backed index in memory from all files in ./data.
    Cached so it's only built once per server session.
    """
    st.write("📄 Loading documents from", DATA_DIR)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # FAISS dimension for OpenAI embeddings (text-embedding-3-small)
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    st.write("💾 Building in-memory FAISS index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    st.write("✅ Index ready.")
    return index


def main():
    st.set_page_config(
        page_title="Western University AI Chatbot",
        page_icon="🎓",
        layout="centered",
    )

    # Simple dark-ish background feel
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top, #111827 0, #020617 55%, #000 100%);
            color: #e5e7eb;
        }
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎓 Western University AI Chatbot")
    st.write(
        "Ask anything about Western University admissions, programs, residence, or student services. "
        "Answers come from scraped Western pages using a FAISS + RAG backend."
    )

    # Build / load index (cached)
    index = build_index()
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I’m the Western University AI assistant. "
                    "Ask me about admission requirements, programs, residence, tuition, "
                    "or any student services."
                ),
            }
        ]

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about Western...")
    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(user_input)
                answer = str(response)
                st.markdown(answer)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
