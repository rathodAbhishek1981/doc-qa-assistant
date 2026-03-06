import gc
import os
import shutil
import tempfile
import time

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHROMA_COLLECTION, TOP_K_RESULTS
from embeddings import load_embeddings


def cleanup_old_db() -> None:
    """Safely delete the previous ChromaDB directory (Windows file-lock safe)."""
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


def load_and_split(pdf_path: str):
    """Load a PDF and split it into chunks. Returns (chunks, page_count)."""
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    if not documents:
        raise ValueError("Could not extract any text from the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks) -> Chroma:
    """Embed chunks and persist them in a fresh ChromaDB directory."""
    cleanup_old_db()

    db_dir = tempfile.mkdtemp(prefix="chroma_rag_")
    st.session_state.db_dir = db_dir

    embeddings = load_embeddings()
    vectordb   = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=CHROMA_COLLECTION,
    )
    vectordb.persist()
    st.session_state.vectordb = vectordb
    return vectordb


def get_retriever(vectordb: Chroma):
    """Return a similarity retriever from an existing vector store."""
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )
