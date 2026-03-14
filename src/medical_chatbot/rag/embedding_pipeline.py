"""
embedding_pipeline.py
────────────────────────────────────────────────
Generates dense vector embeddings using SentenceTransformers.
Default model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE      = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Description: Our Vector Generator! It loads up SentenceTransformers to mathematically convert Arabic text into dense AI-readable arrays.
class DocumentEmbedder:
    """
    Wraps SentenceTransformers for producing sentence embeddings.

    Usage:
        embedder = DocumentEmbedder()
        vecs = embedder.embed(["مرحبا", "كيف حالك"])  # np.ndarray
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        self.device = device
        logger.info(f"Loading Embedding Model from '{model_name}' ...")
        
        # sentence-transformers automatically handles device placement if device is None
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
            
        self.model = SentenceTransformer(model_name, **kwargs)
        logger.info(f"Embedding Model ({model_name}) loaded successfully.")

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # Description: Takes a massive list of strings and efficiently processes them through the transformer model in optimized batches.
    def embed(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Embed a list of texts.

        Returns: numpy array of shape (len(texts), embedding_dim), L2-normalized.
        """
        logger.info(f"Embedding {len(texts):,} texts in batches of {batch_size}...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True, # L2 normalizes the embeddings for cosine similarity via FAISS IP
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    # Description: Used at runtime. Takes the user's single question and mathematicaly embeds it exactly the same way we embedded our dataset.
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed([query], batch_size=1)


# Singleton instance (lazy-loaded at first use)
_embedder:DocumentEmbedder | None = None

# Description: A singleton wrapper that ensures we only load the massive embedding model into RAM one single time.
def get_embedder() -> DocumentEmbedder:
    """Return the global embedder (loaded once)."""
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedder()
    return _embedder

