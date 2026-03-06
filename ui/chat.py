import streamlit as st


def render_chat_area() -> None:
    """Render chat history and the input box. Handles Q&A when a question is submitted."""
    if st.session_state.qa_chain is None:
        _render_welcome()
        return

    st.success("✅ Document loaded. Ask anything about it below.")
    _render_history()
    _handle_input()


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_welcome() -> None:
    st.info("👈 Upload a PDF in the sidebar and click **Process Document** to get started.")

    with st.expander("🚀 Setup Instructions"):
        st.markdown("""
**Step 1 — Install Ollama:** https://ollama.com/download

**Step 2 — Pull the LLM:**
```bash
ollama pull gemma3:1b
```

**Step 3 — Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Run:**
```bash
streamlit run app.py
```

> `all-MiniLM-L6-v2` downloads automatically from HuggingFace on first run (~90 MB).
        """)


def _render_history() -> None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])


def _handle_input() -> None:
    question = st.chat_input("Ask a question about the document…")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result  = st.session_state.qa_chain({"query": question})
                answer  = result["result"].strip()
                sources = result.get("source_documents", [])
            except Exception as e:
                answer  = f"⚠️ Error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            _render_sources(sources)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


def _render_sources(sources: list) -> None:
    with st.expander("📚 Source chunks used"):
        for i, src in enumerate(sources, 1):
            page = src.metadata.get("page", "?")
            st.markdown(f"**Chunk {i} — Page {page}:**")
            st.caption(
                src.page_content[:400] +
                ("…" if len(src.page_content) > 400 else "")
            )
