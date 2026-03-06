from typing import List

import streamlit as st
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from config import EMBEDDING_MODEL


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
        token_embeddings     = model_output.last_hidden_state  # (B, T, H)
        input_mask_expanded  = (
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
        """Process in small batches to avoid OOM on large PDFs."""
        batch_size     = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._encode(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings() -> MiniLMEmbeddings:
    """Load the embedding model once and cache it across Streamlit sessions."""
    return MiniLMEmbeddings()
