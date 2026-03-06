import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import tempfile
import shutil
import os
import gc
import time
import tempfile
import shutil

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from typing import List

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_MODEL      = "gemma3:1b"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 100
TOP_K_RESULTS     = 4
CHROMA_COLLECTION = "rag_docs"

# ── Session state defaults ────────────────────────────────────────────────────
if "qa_chain"     not in st.session_state: st.session_state.qa_chain     = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "db_dir"       not in st.session_state: st.session_state.db_dir       = None
if "vectordb"     not in st.session_state: st.session_state.vectordb     = None

# ── Strict RAG prompt ─────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful and friendly document assistant. Your job is to answer the user's question using the provided context.

Follow these guidelines:
- If the answer is clearly present in the context, answer it directly and concisely.
- If the answer is partially present, use what is available and say "Based on the document..." to indicate you are inferring.
- If the context has related or relevant information that can help, use it to give the most helpful response possible.
- Try to understand the intent behind the question and answer in a natural, conversational tone.
- Only if the context has absolutely NO relevant information at all, respond with: "I couldn't find relevant information about this in the document. Could you try rephrasing your question?"
- Never make up facts, statistics, or information not grounded in the context.

Context:
{context}

Question: {question}

Answer:""",
)

# ── Custom HuggingFace Embeddings class (LangChain compatible) ────────────────
class MiniLMEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using sentence-transformers/all-MiniLM-L6-v2
    loaded directly via HuggingFace transformers (no sentence-transformers lib needed).
    Uses mean pooling + L2 normalisation — same as the official SentenceTransformer.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state          # (B, T, H)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
               torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pooling(output, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Process in small batches to avoid OOM on large PDFs
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._encode(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


# ── Cache the embedding model (load once, reuse across sessions) ──────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings() -> MiniLMEmbeddings:
    return MiniLMEmbeddings()


# ── Helper: safely delete old ChromaDB (Windows-safe) ────────────────────────
def cleanup_old_db():
    if st.session_state.vectordb is not None:
        try:
            del st.session_state.vectordb
            st.session_state.vectordb = None
            gc.collect()
            time.sleep(0.5)
        except Exception:
            pass

    if st.session_state.db_dir and os.path.exists(st.session_state.db_dir):
        for _ in range(3):
            try:
                shutil.rmtree(st.session_state.db_dir)
                st.session_state.db_dir = None
                break
            except Exception:
                time.sleep(0.5)


# ── Helper: build QA chain ────────────────────────────────────────────────────
def build_qa_chain(pdf_path: str):
    """Ingest PDF → chunk → embed → store in ChromaDB → return RetrievalQA chain."""

    # 1. Load PDF
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    if not documents:
        raise ValueError("Could not extract any text from the PDF.")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # 3. Clean up old DB (Windows file-lock safe)
    cleanup_old_db()

    # 4. Embed & store in ChromaDB
    db_dir = tempfile.mkdtemp(prefix="chroma_rag_")
    st.session_state.db_dir = db_dir

    embeddings = load_embeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=CHROMA_COLLECTION,
    )
    vectordb.persist()
    st.session_state.vectordb = vectordb

    # 5. Retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # 6. LLM (Ollama — only used for generation)
    llm = Ollama(model=OLLAMA_MODEL, temperature=2, num_predict=1024)

    # 7. RetrievalQA with strict prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return qa_chain, len(chunks)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📄 Document Q&A Assistant")
st.caption(
    f"LLM: **{OLLAMA_MODEL}** via Ollama  ·  "
    f"Embeddings: **all-MiniLM-L6-v2** (HuggingFace)  ·  "
    f"Vector DB: **ChromaDB**  ·  UI: **Streamlit**"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"**LLM:** `{OLLAMA_MODEL}`")
    st.markdown(f"**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown(f"**Chunk size:** {CHUNK_SIZE} · **Overlap:** {CHUNK_OVERLAP}")
    st.markdown(f"**Top-K:** {TOP_K_RESULTS}")
    st.divider()

    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("🔄 Process Document", use_container_width=True):
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

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────
if st.session_state.qa_chain is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process Document** to get started.")

    with st.expander("🚀 Setup Instructions"):
        st.markdown("""
**Step 1 — Install Ollama:** https://ollama.com/download

**Step 2 — Pull the LLM (only Ollama model needed now):**
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

else:
    st.success("✅ Document loaded. Ask anything about it below.")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Source chunks used"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} — Page {page}:**")
                        st.caption(
                            src.page_content[:400] +
                            ("…" if len(src.page_content) > 400 else "")
                        )

    # Chat input
    question = st.chat_input("Ask a question about the document…")
    if question:
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
                with st.expander("📚 Source chunks used"):
                    for i, src in enumerate(sources, 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} — Page {page}:**")
                        st.caption(
                            src.page_content[:400] +
                            ("…" if len(src.page_content) > 400 else "")
                        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })