import os
import tempfile

import streamlit as st

from config import CHUNK_OVERLAP, CHUNK_SIZE, OLLAMA_MODEL, TOP_K_RESULTS
from qa_chain import build_qa_chain


def render_sidebar() -> None:
    """Render the full sidebar: config info, PDF uploader, and clear-chat button."""
    with st.sidebar:
        _render_config_info()
        st.divider()
        _render_uploader()
        st.divider()
        _render_clear_button()


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_config_info() -> None:
    st.header("⚙️ Configuration")
    st.markdown(f"**LLM:** `{OLLAMA_MODEL}`")
    st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown(f"**Chunk size:** {CHUNK_SIZE} · **Overlap:** {CHUNK_OVERLAP}")
    st.markdown(f"**Top-K:** {TOP_K_RESULTS}")


def _render_uploader() -> None:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if not uploaded_file:
        return

    if not st.button("🔄 Process Document", use_container_width=True):
        return

    with st.spinner("Ingesting & embedding… this may take a moment."):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            qa_chain, n_chunks = build_qa_chain(tmp_path)
            st.session_state.qa_chain     = qa_chain
            st.session_state.chat_history = []
            st.success(f"✅ Ready! Split into **{n_chunks}** chunks.")

        except Exception as e:
            st.error(f"Error processing document: {e}")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


def _render_clear_button() -> None:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
