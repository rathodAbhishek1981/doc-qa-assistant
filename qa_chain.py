from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

from config import OLLAMA_MODEL, RAG_PROMPT
from vector_store import build_vector_store, get_retriever, load_and_split


def build_qa_chain(pdf_path: str):
    """
    Full pipeline: ingest PDF → chunk → embed → store → return (chain, chunk_count).
    """
    chunks   = load_and_split(pdf_path)
    vectordb = build_vector_store(chunks)
    retriever = get_retriever(vectordb)

    llm = Ollama(model=OLLAMA_MODEL, temperature=2, num_predict=1024)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return qa_chain, len(chunks)
