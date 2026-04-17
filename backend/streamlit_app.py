import os
import time

from dotenv import load_dotenv
import streamlit as st
import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore


DATA_DIR = "data"
INDEX_DIR = "index_storage"
FAISS_PATH = os.path.join(INDEX_DIR, "default__vector_store.json")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")
os.environ["OPENAI_API_KEY"] = api_key

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

SYSTEM_PROMPT = PromptTemplate(
    "You are the Western University AI Assistant — a knowledgeable, professional, "
    "and friendly guide for students and prospective students at Western University "
    "(London, Ontario, Canada). You assist with admissions, programs, tuition, "
    "residence, scholarships, campus life, and student services.\n\n"
    "Use only the context below from the Western University website to answer. "
    "If you cannot find a direct answer, say so and suggest visiting uwo.ca or "
    "contacting the Western admissions office.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer:"
)

HELP_TOPICS = [
    {
        "icon": "📚",
        "title": "Admissions Info",
        "desc": "Requirements, deadlines & how to apply.",
    },
    {
        "icon": "🎓",
        "title": "Programs & Faculties",
        "desc": "Explore undergraduate & graduate programs.",
    },
    {
        "icon": "🏠",
        "title": "Residence & Housing",
        "desc": "On-campus living options and costs.",
    },
    {
        "icon": "💰",
        "title": "Tuition & Scholarships",
        "desc": "Fees, financial aid, and bursaries.",
    },
]

SUGGESTED = [
    "What are the admission requirements for engineering?",
    "How much is first-year tuition?",
    "What residence options are available?",
    "What scholarships can I apply for?",
    "How do I apply to Western University?",
    "What programs does the Ivey Business School offer?",
]

CAMPUS_IMG = "https://www.uwo.ca/brand/assets/img/uw_campus.jpg"


@st.cache_resource(show_spinner=False)
def load_index():
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=INDEX_DIR,
        )
        return load_index_from_storage(storage_context)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=Settings.embed_model
    )
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Greetings! I am the Western University AI. How may I assist your "
                    "academic journey today? Whether you need help with admissions, "
                    "programs, residence, or understanding life at Western, I am here "
                    "to guide you."
                ),
                "sources": [],
            }
        ]
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = ["Current Session"]
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * { box-sizing: border-box; }

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif !important;
            background: #f5f4f8 !important;
        }

        /* ── Hide default streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ── Full-width layout ── */
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e8e4f0;
            width: 240px !important;
            min-width: 240px !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 0 !important;
        }
        .sidebar-logo {
            background: linear-gradient(135deg, #4F2683 0%, #6b35a8 100%);
            padding: 1.4rem 1.2rem 1.2rem;
            color: white;
        }
        .sidebar-logo .brand {
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -0.2px;
            line-height: 1.2;
        }
        .sidebar-logo .sub {
            font-size: 0.7rem;
            opacity: 0.75;
            margin-top: 2px;
            font-weight: 400;
        }
        .sidebar-logo .avatar {
            width: 36px; height: 36px;
            background: rgba(255,255,255,0.25);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem;
            margin-bottom: 0.7rem;
        }
        .sidebar-section {
            padding: 0.8rem 1rem;
        }
        .sidebar-section-label {
            font-size: 0.65rem;
            font-weight: 600;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .sidebar-item {
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.5rem 0.6rem;
            border-radius: 8px;
            font-size: 0.83rem;
            color: #374151;
            cursor: pointer;
            transition: background 0.15s;
            font-weight: 500;
        }
        .sidebar-item:hover { background: #f3f0f9; color: #4F2683; }
        .sidebar-item.active {
            background: #f3f0f9;
            color: #4F2683;
            font-weight: 600;
        }
        .sidebar-item .icon { font-size: 0.9rem; width: 18px; text-align: center; }
        .upgrade-btn {
            margin: 0.5rem 1rem 1rem;
            background: linear-gradient(135deg, #4F2683 0%, #6b35a8 100%);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1rem;
            font-size: 0.83rem;
            font-weight: 600;
            width: calc(100% - 2rem);
            cursor: pointer;
            text-align: center;
        }
        .sidebar-divider { border-top: 1px solid #f0edf7; margin: 0.3rem 1rem; }

        /* ── Right panel ── */
        .right-panel {
            background: #ffffff;
            border-left: 1px solid #e8e4f0;
            padding: 1.2rem;
            height: 100vh;
            overflow-y: auto;
        }
        .right-panel-title {
            font-size: 0.65rem;
            font-weight: 700;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.8rem;
        }
        .help-card {
            background: #f9f7fd;
            border: 1px solid #e8e4f0;
            border-radius: 10px;
            padding: 0.75rem;
            margin-bottom: 0.6rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        .help-card:hover {
            background: #f0ebfa;
            border-color: #c4a8e8;
        }
        .help-card-title {
            font-size: 0.82rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.2rem;
        }
        .help-card-desc {
            font-size: 0.73rem;
            color: #6b7280;
        }
        .campus-img {
            width: 100%;
            border-radius: 10px;
            margin-top: 0.8rem;
            object-fit: cover;
            height: 110px;
        }
        .campus-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #4F2683;
            text-align: center;
            margin-top: 0.4rem;
        }

        /* ── Chat area ── */
        .chat-header {
            padding: 1rem 1.5rem 0.8rem;
            border-bottom: 1px solid #ede9f5;
            background: white;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .chat-header-avatar {
            width: 38px; height: 38px;
            background: linear-gradient(135deg, #4F2683, #6b35a8);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }
        .chat-header-name {
            font-size: 0.95rem;
            font-weight: 700;
            color: #1f2937;
        }
        .chat-header-status {
            font-size: 0.72rem;
            color: #10b981;
            display: flex; align-items: center; gap: 0.3rem;
        }
        .status-dot {
            width: 6px; height: 6px;
            background: #10b981;
            border-radius: 50%;
            display: inline-block;
        }

        /* ── Messages ── */
        .msg-row {
            display: flex;
            padding: 0.6rem 1.5rem;
            gap: 0.7rem;
        }
        .msg-row.user {
            flex-direction: row-reverse;
            justify-content: flex-start;
        }
        .msg-avatar {
            width: 32px; height: 32px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.85rem;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .msg-avatar.ai {
            background: linear-gradient(135deg, #4F2683, #6b35a8);
            color: white;
        }
        .msg-avatar.user {
            background: #e0d5f5;
            color: #4F2683;
        }
        .msg-bubble {
            max-width: 72%;
            border-radius: 16px;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            line-height: 1.55;
        }
        .msg-bubble.ai {
            background: #f4f2f9;
            color: #1f2937;
            border-radius: 4px 16px 16px 16px;
            border: 1px solid #e8e4f0;
        }
        .msg-bubble.user {
            background: linear-gradient(135deg, #4F2683, #6b35a8);
            color: white;
            border-radius: 16px 16px 4px 16px;
        }
        .msg-time {
            font-size: 0.65rem;
            color: #9ca3af;
            margin-top: 4px;
            text-align: right;
        }
        .msg-time.ai { text-align: left; }

        /* ── Input area ── */
        .input-bar {
            border-top: 1px solid #ede9f5;
            background: white;
            padding: 0.9rem 1.5rem;
        }
        .disclaimer {
            text-align: center;
            font-size: 0.68rem;
            color: #9ca3af;
            padding: 0.4rem 1.5rem 0.6rem;
            background: white;
            letter-spacing: 0.02em;
        }
        .disclaimer strong { color: #6b7280; }

        /* ── Streamlit chat input override ── */
        [data-testid="stChatInput"] {
            border: 1.5px solid #e8e4f0 !important;
            border-radius: 12px !important;
            background: #faf9fd !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #4F2683 !important;
            box-shadow: 0 0 0 3px rgba(79,38,131,0.1) !important;
        }
        [data-testid="stChatInputSubmitButton"] > svg {
            fill: #4F2683 !important;
        }
        [data-testid="stChatInputSubmitButton"] {
            background: linear-gradient(135deg, #4F2683, #6b35a8) !important;
            border-radius: 8px !important;
        }
        [data-testid="stChatInputSubmitButton"] > svg {
            fill: white !important;
        }

        /* ── Spinner ── */
        .stSpinner > div { border-top-color: #4F2683 !important; }

        /* ── Suggest buttons ── */
        .suggest-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            padding: 0.5rem 1.5rem 0.8rem;
        }
        .suggest-chip {
            background: #f4f2f9;
            border: 1px solid #ddd6f3;
            border-radius: 20px;
            padding: 0.4rem 0.8rem;
            font-size: 0.75rem;
            color: #4F2683;
            cursor: pointer;
            text-align: center;
            font-weight: 500;
            transition: all 0.15s;
        }
        .suggest-chip:hover { background: #ede6f9; }

        /* ── Stbutton resets ── */
        .stButton > button {
            background: #f4f2f9 !important;
            border: 1px solid #ddd6f3 !important;
            border-radius: 20px !important;
            color: #4F2683 !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            padding: 0.35rem 0.85rem !important;
            height: auto !important;
            line-height: 1.4 !important;
            transition: all 0.15s !important;
        }
        .stButton > button:hover {
            background: #ede6f9 !important;
            border-color: #b89ee0 !important;
        }
        .clear-btn > button {
            background: white !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
            color: #6b7280 !important;
            font-size: 0.75rem !important;
        }
        .clear-btn > button:hover {
            background: #fef2f2 !important;
            border-color: #fca5a5 !important;
            color: #ef4444 !important;
        }

        /* scrollbar */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #d4c9ed; border-radius: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_html():
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
            <div class="avatar">🎓</div>
            <div class="brand">Western University AI</div>
            <div class="sub">AI Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("✏️", "New Chat", True),
        ("🕐", "Recent Conversations", False),
        ("🔖", "Saved Snippets", False),
        ("📁", "Archive", False),
    ]
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    for icon, label, active in nav_items:
        cls = "sidebar-item active" if active else "sidebar-item"
        if label == "New Chat":
            if st.sidebar.button(f"{icon}  {label}", key="new_chat_btn", use_container_width=True):
                st.session_state.messages = [st.session_state.messages[0]]
                st.rerun()
        else:
            st.sidebar.markdown(
                f'<div class="{cls}"><span class="icon">{icon}</span>{label}</div>',
                unsafe_allow_html=True,
            )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="upgrade-btn">⚡ Upgrade to Pro</div>', unsafe_allow_html=True
    )
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            <div class="sidebar-item"><span class="icon">⚙️</span>Settings</div>
            <div class="sidebar-item"><span class="icon">🚪</span>Log Out</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_time():
    return time.strftime("Today, %I:%M %p")


def main():
    st.set_page_config(
        page_title="Western University AI",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_css()
    init_session()
    sidebar_html()

    with st.spinner("Loading knowledge base..."):
        try:
            index = load_index()
        except Exception as e:
            st.error(f"Failed to load knowledge base: {e}")
            st.stop()

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        text_qa_template=SYSTEM_PROMPT,
    )

    # ── 2-column layout: chat | right panel ──
    chat_col, right_col = st.columns([3, 1], gap="small")

    # ────────────────────────── RIGHT PANEL ──────────────────────────
    with right_col:
        st.markdown(
            """
            <div class="right-panel">
                <div class="right-panel-title">Help Topics</div>
            """,
            unsafe_allow_html=True,
        )
        for topic in HELP_TOPICS:
            st.markdown(
                f"""
                <div class="help-card">
                    <div class="help-card-title">{topic["icon"]} {topic["title"]}</div>
                    <div class="help-card-desc">{topic["desc"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
                <div class="right-panel-title" style="margin-top:1.2rem">Institutional Resources</div>
                <img class="campus-img"
                     src="https://communications.uwo.ca/img/western_uwo_social_sharing.jpg"
                     onerror="this.src='https://www.uwo.ca/img/maps/campus.jpg'" />
                <div class="campus-label">🏛 Explore Campus Services</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ────────────────────────── CHAT PANEL ──────────────────────────
    with chat_col:
        # Header
        st.markdown(
            f"""
            <div class="chat-header">
                <div class="chat-header-avatar">🎓</div>
                <div>
                    <div class="chat-header-name">Atelier AI</div>
                    <div class="chat-header-status">
                        <span class="status-dot"></span> Online — Western University Assistant
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested questions (only on fresh chat)
        if len(st.session_state.messages) == 1:
            st.markdown('<div style="padding: 0.8rem 1.5rem 0.3rem;">'
                        '<span style="font-size:0.72rem;color:#9ca3af;font-weight:600;'
                        'text-transform:uppercase;letter-spacing:0.06em;">Quick questions</span>'
                        '</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, q in enumerate(SUGGESTED):
                if cols[i % 2].button(q, key=f"sq_{i}"):
                    st.session_state.pending_question = q
                    st.rerun()

        # Message history
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                st.markdown(
                    f"""
                    <div class="msg-row">
                        <div class="msg-avatar ai">🎓</div>
                        <div>
                            <div class="msg-bubble ai">{content}</div>
                            <div class="msg-time ai">{format_time()}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    with st.expander("📎 Sources", expanded=False):
                        for s in msg["sources"]:
                            st.markdown(f"- `{s}`")
            else:
                st.markdown(
                    f"""
                    <div class="msg-row user">
                        <div class="msg-avatar user">👤</div>
                        <div>
                            <div class="msg-bubble user">{content}</div>
                            <div class="msg-time">{format_time()}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Clear button (top-right of chat)
        cc1, cc2 = st.columns([5, 1])
        with cc2:
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            if st.button("🗑 Clear", key="clear_chat"):
                st.session_state.messages = [st.session_state.messages[0]]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Chat input
        pending = st.session_state.pop("pending_question", None)
        user_input = st.chat_input("Ask the Western University AI...") or pending

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})

            with st.spinner("Searching Western's knowledge base..."):
                try:
                    response = query_engine.query(user_input)
                    answer = str(response)
                    sources = []
                    if hasattr(response, "source_nodes"):
                        seen = set()
                        for node in response.source_nodes:
                            fname = node.metadata.get("file_name", "")
                            if fname and fname not in seen:
                                seen.add(fname)
                                sources.append(fname)
                except Exception:
                    answer = (
                        "I'm sorry, I encountered an error. Please try again or "
                        "visit [uwo.ca](https://www.uwo.ca) for official information."
                    )
                    sources = []

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.rerun()

        # Disclaimer
        st.markdown(
            """
            <div class="disclaimer">
                <strong>ACADEMIC INTEGRITY IS OUR STANDARD.</strong>
                AI may generate incorrect information — always verify with official Western sources.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
