# 📄 Document Q&A Assistant

A local Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask questions about it — all running privately on your machine using Ollama. No paid APIs, no data leaves your computer.

---

## 🧠 Architecture Overview

```
PDF Upload
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter
    │
    ▼
MiniLM Embeddings (all-MiniLM-L6-v2, HuggingFace)
    │
    ▼
ChromaDB (local vector store)
    │
    ▼
Similarity Search (Top-K retrieval)
    │
    ▼
Ollama LLM (gemma3:1b) + RAG Prompt
    │
    ▼
Streamlit Chat UI
```

**Chunking Strategy:** A chunk size of `500` tokens with an overlap of `100` was chosen to balance context completeness and retrieval precision. 500 tokens is large enough to capture a full paragraph or logical thought without splitting mid-sentence, while the 100-token overlap ensures that answers spanning chunk boundaries are not missed. `RecursiveCharacterTextSplitter` with `["\n\n", "\n", ".", " "]` separators preserves natural paragraph and sentence boundaries, which produces cleaner embeddings and more coherent retrieved context.

---

## ✅ Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download), then pull the required model:

```bash
ollama pull gemma3:1b
```

> The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically from HuggingFace on first run.

### 2. Python 3.9+

Ensure you have Python 3.9 or higher installed:

```bash
python --version
```

---

## ⚙️ Installation

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/document-qa-assistant.git
cd document-qa-assistant
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Make sure Ollama is running in the background, then start the app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**Steps:**
1. Upload a PDF using the sidebar.
2. Click **Process Document** and wait for the embedding to complete.
3. Type your question in the chat input.
4. The assistant will answer based strictly on the document content.

---

## 📁 Project Structure

```
rag_app/
├── app.py              # Entry point — Streamlit page config & layout
├── config.py           # All constants (model names, chunk sizes) & RAG prompt
├── embeddings.py       # MiniLMEmbeddings class + cached HuggingFace loader
├── vector_store.py     # PDF loading, chunking, ChromaDB lifecycle & retriever
├── qa_chain.py         # Wires retriever + LLM into a RetrievalQA chain
├── ui/
│   ├── sidebar.py      # Sidebar: config info, file uploader, clear-chat button
│   └── chat.py         # Chat history rendering, input handling, source expander
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| LLM Inference | [Ollama](https://ollama.com) (`gemma3:1b`) |
| Orchestration | LangChain |
| Vector Database | ChromaDB (local) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| UI | Streamlit |

---

## 💬 Bot Behaviour

- **Answer found in document** → Responds directly and concisely based on the context.
- **Answer not in document** → Responds with: *"I couldn't find relevant information about this in the document. Could you try rephrasing your question?"*
- The LLM is strictly prompted to never hallucinate or use outside knowledge.

---

## 📦 Dependencies

See [`requirements.txt`](./requirements.txt) for the full list. Key packages:

```
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-core
chromadb
pypdf
transformers
torch
ollama
```

---

## 🔒 Privacy

All processing happens locally on your machine. No data is sent to any external API or cloud service.
