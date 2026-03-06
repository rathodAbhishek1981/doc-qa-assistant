# 📄 Document Q&A Assistant

A local RAG (Retrieval-Augmented Generation) pipeline that lets you upload a PDF and ask questions about it — powered entirely by local models via Ollama. No paid APIs, no data leaves your machine.

---

## 🗂️ Project Structure

```
doc-qa-assistant/
├── main.py                    # App entrypoint
├── requirements.txt
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py            # All tunable config (models, chunk sizes, paths)
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py        # Orchestrator: ties ingestion → retrieval → generation
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_loader.py      # PDF loading + chunking
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py    # ChromaDB build, load, search
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py             # Ollama LLM + strict RAG prompt
│   └── ui/
│       ├── __init__.py
│       └── app.py             # Streamlit frontend
├── data/                      # (Optional) Store PDFs locally
└── vectorstore/               # ChromaDB persisted data (auto-created)
    └── chroma_db/
```

---

## ✅ Prerequisites

### 1. Install Ollama

Download from [https://ollama.com](https://ollama.com) and install for your OS.

### 2. Pull Required Models

```bash
# LLM for answer generation
ollama pull gemma3:1b

# Embedding model for vector search
ollama pull nomic-embed-text
```

### 3. Python 3.9+

Verify your Python version:
```bash
python --version
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd doc-qa-assistant

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Make sure Ollama is running in the background:
```bash
ollama serve
```

Then launch the app:
```bash
python main.py
```

Or directly via Streamlit:
```bash
streamlit run src/ui/app.py
```

The app will open at **http://localhost:8501**

### How to use:
1. Upload a PDF using the sidebar file uploader
2. Click **Process Document** and wait for embedding to complete
3. Type your question in the chat box
4. The assistant answers strictly from the document content

---

## 🧠 Architecture & Design Decisions

### Chunking Strategy
- **Chunk size: 500 characters** — Large enough to preserve meaningful context for typical PDF paragraphs, small enough to keep retrieval precise.
- **Overlap: 100 characters** — Prevents losing important context that spans chunk boundaries (e.g., a sentence split between two chunks).
- **Splitter: `RecursiveCharacterTextSplitter`** — Respects natural text boundaries (paragraphs → sentences → words) before hard-cutting.

### Embedding Model
`nomic-embed-text` via Ollama — lightweight, high-quality open-source embeddings that run fully locally with no API calls.

### RAG Flow
```
PDF Upload
    ↓
PyPDFLoader (load pages)
    ↓
RecursiveCharacterTextSplitter (chunk)
    ↓
OllamaEmbeddings / nomic-embed-text (embed)
    ↓
ChromaDB (persist vectors locally)
    ↓
User Question → Similarity Search (top 4 chunks)
    ↓
Strict RAG Prompt → gemma3:1b via Ollama
    ↓
Answer (document-only, no hallucination)
```

### Strict No-Hallucination Policy
The LLM prompt explicitly instructs the model:
- Answer ONLY from provided context
- If answer not found → return: *"I cannot find the answer to that question in the provided document."*
- Temperature set to `0.1` for deterministic, factual responses

---

## 🛠️ Configuration

All settings are in `config/settings.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemma3:1b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| LLM Inference | Ollama (gemma3:1b) |
| Embeddings | Ollama (nomic-embed-text) |
| Orchestration | LangChain |
| Vector Database | ChromaDB (local) |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |
