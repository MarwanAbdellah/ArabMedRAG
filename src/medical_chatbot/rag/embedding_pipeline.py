"""
embedding_pipeline.py
────────────────────────────────────────────────
Generates dense vector embeddings using AraBERT (aubmindlab/bert-base-arabertv2).

AraBERT is an Arabic-native BERT model trained on 77 GB of Arabic text,
providing superior Arabic medical term understanding compared to
multilingual models.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "aubmindlab/bert-base-arabertv2")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))


class DocumentEmbedder:
    """
    Wraps AraBERT for producing Arabic-native sentence embeddings.

    Uses mean-pooling over token embeddings with L2 normalization
    for compatibility with FAISS IndexFlatIP (cosine similarity).

    Usage:
        embedder = DocumentEmbedder()
        vecs = embedder.embed(["مرحبا", "ما هي اعراض السكري"])  # np.ndarray
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading AraBERT model '{model_name}' on {self.device}...")

        # Optional: AraBERT preprocessor for text normalization
        self.preprocessor = None
        try:
            from arabert.preprocess import ArabertPreprocessor
            self.preprocessor = ArabertPreprocessor(model_name=model_name)
            logger.info("ArabertPreprocessor loaded for text normalization.")
        except ImportError:
            logger.warning(
                "arabert package not installed — skipping ArabertPreprocessor. "
                "Install with: pip install arabert"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"AraBERT model loaded successfully. dim={self.embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size  # 768 for AraBERT

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings weighted by attention mask."""
        token_embeddings = model_output.last_hidden_state
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(
            input_mask.sum(1), min=1e-9
        )

    def _preprocess(self, texts: List[str]) -> List[str]:
        """Apply AraBERT preprocessing if available."""
        if self.preprocessor is None:
            return texts
        return [self.preprocessor.preprocess(t) for t in texts]

    def embed(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Embed a list of texts.

        Returns: numpy array of shape (len(texts), embedding_dim), L2-normalized.
        """
        logger.info(f"Embedding {len(texts):,} texts with AraBERT (batch_size={batch_size})...")
        texts = self._preprocess(texts)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            embeddings = self._mean_pooling(output, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

            if (i // batch_size) % 50 == 0 and i > 0:
                logger.info(f"  ... embedded {i + len(batch):,} / {len(texts):,} texts")

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed([query], batch_size=1)


# Singleton instance (lazy-loaded at first use)
_embedder: DocumentEmbedder | None = None


def get_embedder() -> DocumentEmbedder:
    """Return the global embedder (loaded once)."""
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedder()
    return _embedder
