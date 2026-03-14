"""
vector_store.py
────────────────────────────────────────────────
FAISS-based vector store for dense retrieval.

Uses IndexFlatIP (inner product) with L2-normalized vectors,
which is equivalent to cosine similarity.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "indexes/faiss_index.bin")
METADATA_PATH    = os.getenv("METADATA_PATH",    "indexes/metadata.pkl")
TOP_K            = int(os.getenv("TOP_K", "5"))


# Description: Our wrapper class around Meta's blazing-fast FAISS system, which lets us query millions of vectors instantly.
class FAISSVectorStore:
    """
    Manages a FAISS flat inner-product index.

    Attributes:
        index    : The FAISS index object.
        metadata : List of dicts storing per-vector metadata.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: List[dict] = []

    # ── Build ──────────────────────────────────────────

    # Description: Ingests all of our mathematically generated vectors into the FAISS structure.
    def build(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        """
        Build the FAISS index from pre-computed embeddings.

        Args:vectors  : (N, D) float32 L2-normalized embeddings.
            metadata : List of N dicts (one per vector).
        """
        assert vectors.shape[1] == self.embedding_dim, (
            f"Dimension mismatch: expected {self.embedding_dim}, got {vectors.shape[1]}"
        )
        self.index    = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(vectors)
        self.metadata = metadata
        logger.info(f"Built FAISS index with {self.index.ntotal:,} vectors (dim={self.embedding_dim}).")

    # ── Persist ────────────────────────────────────────

    # Description: Persists the binary FAISS indexes safely to the filesystem alongside their metadata.
    def save(self, index_path: str = FAISS_INDEX_PATH, meta_path: str = METADATA_PATH) -> None:
        """Save the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:pickle.dump(self.metadata, f)
        logger.info(f"Saved FAISS index → {index_path}")
        logger.info(f"Saved metadata    → {meta_path}")

    # Description: Restores the binary FAISS files. If they don't exist, we appropriately crash and warn the developer to run `build_index` first.
    @classmethod
    def load(
        cls,
        index_path: str = FAISS_INDEX_PATH,
        meta_path:  str = METADATA_PATH,
    ) -> "FAISSVectorStore":
        """Load a pre-built FAISS index from disk."""
        if not os.path.exists(index_path):raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run: python src/medical_chatbot/rag/build_index.py"
            )
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        store = cls(embedding_dim=index.d)
        store.index    = index
        store.metadata = metadata
        logger.info(f"Loaded FAISS index ({index.ntotal:,} vectors, dim={index.d}).")
        return store

    # ── Search ─────────────────────────────────────────

    # Description: The dense retrieval entrypoint! Hands FAISS our embedded query, and magically gets back the top mathematical neighbors.
    def search(
        self,
        query_vector: np.ndarray,
        k: int = TOP_K,
    ) -> List[Tuple[dict, float]]:
        """
        Search the index for the k nearest neighbours.

        Args:query_vector : (1, D) or (D,) float32 L2-normalized vector.
            k            : Number of results to return.

        Returns:
            List of (metadata_dict, similarity_score) tuples, sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index is not loaded. Call build() or load() first.")

        query_vector = np.atleast_2d(query_vector).astype(np.float32)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS returns -1 for missing results
                continue
            results.append((self.metadata[idx], float(score)))

        return results   # already sorted descending by FAISS


# ── Singleton loader ───────────────────────────────────

_vector_store: FAISSVectorStore | None = None


# Description: Singleton loader that ensures the heavy FAISS index stays permanently cached in memory.
def get_vector_store() -> FAISSVectorStore:
    """Return the global FAISS vector store (loaded once from disk)."""
    global _vector_store
    if _vector_store is None:_vector_store = FAISSVectorStore.load()
    return _vector_store
