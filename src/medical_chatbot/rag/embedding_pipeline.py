"""
embedding_pipeline.py
────────────────────────────────────────────────
Generates dense vector embeddings for Arabic medical retrieval.

Uses intfloat/multilingual-e5-base (DEFAULT):
  Fine-tuned via contrastive learning on 1B+ multilingual pairs.
  Uses asymmetric query: / passage: prefixes for retrieval.
  Typically +25-40% NDCG over raw masked-language-model baselines.
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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
BATCH_SIZE      = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_SEQ_LENGTH  = int(os.getenv("MAX_SEQ_LENGTH", "512"))

# Models that use asymmetric query:/passage: prefixes (E5 family)
_E5_PREFIXES = ("intfloat/multilingual-e5", "intfloat/e5-")


class DocumentEmbedder:
    """
    Produces L2-normalized dense embeddings for Arabic medical retrieval.

    For E5 models: automatically prepends "query: " or "passage: " to
    enable asymmetric retrieval (dramatically improves relevance).
    For other models: no prefix is added (legacy behaviour).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._use_e5_prefix = any(model_name.startswith(p) for p in _E5_PREFIXES)

        logger.info(f"Loading embedding model '{model_name}' on {self.device} "
                    f"(E5-prefix={self._use_e5_prefix})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"Embedding model loaded. dim={self.embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size  # 768 for E5-base

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """Attention-mask-weighted mean pool over token embeddings."""
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Tokenize, forward-pass, mean-pool, L2-normalize a batch."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded)
        pooled = self._mean_pooling(output, encoded["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Embed a list of document/passage texts (for index building).

        E5 models: prepends "passage: " to each text.
        """
        prefix = "passage: " if self._use_e5_prefix else ""
        prefixed = [prefix + t for t in texts]

        logger.info(f"Embedding {len(texts):,} documents (prefix={prefix!r}, "
                    f"batch_size={batch_size})...")
        batches = []
        for i in range(0, len(prefixed), batch_size):
            batches.append(self._encode_batch(prefixed[i: i + batch_size]))
            if (i // batch_size) % 50 == 0 and i > 0:
                logger.info(f"  ... {i + batch_size:,} / {len(prefixed):,}")
        return np.vstack(batches).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Returns shape (1, dim).

        E5 models: prepends "query: " to the text.
        """
        prefix = "query: " if self._use_e5_prefix else ""
        return self._encode_batch([prefix + query]).astype(np.float32)

    # Kept for backward compatibility with any code calling embed() directly
    def embed(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Alias for embed_documents (no prefix distinction — legacy use)."""
        return self.embed_documents(texts, batch_size=batch_size)


# ── Singleton ─────────────────────────────────────────────────────────────────

_embedder: DocumentEmbedder | None = None


def get_embedder() -> DocumentEmbedder:
    """Return the global embedder (loaded once per process)."""
    global _embedder
    if _embedder is None:
        _embedder = DocumentEmbedder()
    return _embedder
