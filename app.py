import streamlit as st

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, OLLAMA_MODEL, TOP_K_RESULTS
from ui.sidebar import render_sidebar
from ui.chat import render_chat_area

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "qa_chain"     not in st.session_state: st.session_state.qa_chain     = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "db_dir"       not in st.session_state: st.session_state.db_dir       = None
if "vectordb"     not in st.session_state: st.session_state.vectordb     = None

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("📄 Document Q&A Assistant")
st.caption(
    f"LLM: **{OLLAMA_MODEL}** via Ollama  ·  "
    f"Embeddings: **all-MiniLM-L6-v2** (HuggingFace)  ·  "
    f"Vector DB: **ChromaDB**  ·  UI: **Streamlit**"
)

render_sidebar()
render_chat_area()
