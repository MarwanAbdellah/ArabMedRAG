"""
keyword_index.py
────────────────────────────────────────────────
BM25 keyword index for sparse retrieval fallback.

Uses `rank_bm25` (Okapi BM25) on Arabic tokenized text.
The index is saved as a pickle file for fast re-loading.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "indexes/bm25_index.pkl")
TOP_K           = int(os.getenv("TOP_K", "5"))


# Arabic stopwords to remove from BM25 tokenization so matching focuses on
# disease names and medical terms, not question words like "ما" or "هي"
ARABIC_STOPWORDS = {
    # Question words
    "ما", "هي", "هو", "هل", "كيف", "لماذا", "متى", "أين", "من",
    # Prepositions & conjunctions
    "في", "عن", "على", "إلى", "مع", "أو", "و", "ب", "ل", "ك",
    "بها", "لها", "فيها", "منها", "عنها",
    # Demonstratives & relatives
    "هذا", "هذه", "ذلك", "تلك", "الذي", "التي",
    # Verbs & auxiliaries
    "يكون", "كان", "ليس", "يمكن", "قد", "لا", "نعم",
    "فقط", "أيضا", "جدا", "بعض", "كل", "أي", "يتم",
    # Single chars
    "أ", "ا", "ي",
}


# Description: A custom text cleanser! It rips out diacritics and punctuation so that the keyword search doesn't fail over simple grammar differences.
def _tokenize_arabic(text: str) -> List[str]:
    """
    Arabic tokenizer for BM25 matching.
    Removes diacritics, punctuation, and common stopwords
    so BM25 focuses on medical/disease terms.
    """
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)
    # Remove punctuation and numbers
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in ARABIC_STOPWORDS]


# Description: This class wraps our sparse keyword retrieval system, perfect as a fallback when the sophisticated vector search gets confused.
class BM25KeywordIndex:
    """
    BM25 keyword index backed by rank_bm25.

    Usage:index = BM25KeywordIndex.build(chunks)
        results = index.search("ألم في الصدر", k=5)
    """

    def __init__(self, bm25: BM25Okapi, metadata: List[dict]):
        self.bm25     = bm25
        self.metadata = metadata

    # ── Build ──────────────────────────────────────────

    # Description: Ingests all of our parsed text chunks and calculates the crucial TF-IDF scoring arrays for keyword matching.
    @classmethod
    def build(cls, metadata: List[dict]) -> "BM25KeywordIndex":
        """
        Build BM25 index from a list of metadata dicts.

        Each dict must contain at least a "text" key.
        """
        logger.info("Building BM25 index ...")
        corpus = [_tokenize_arabic(m["text"]) for m in metadata]
        bm25   = BM25Okapi(corpus)
        logger.info(f"BM25 index built over {len(corpus):,} documents.")
        return cls(bm25=bm25, metadata=metadata)

    # ── Persist ────────────────────────────────────────

    # Description: Dumps the processed BM25 indices straight to disk so we don't have to recalculate them on every boot.
    def save(self, path: str = BM25_INDEX_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metadata": self.metadata}, f)
        logger.info(f"Saved BM25 index → {path}")

    # Description: Restores the BM25 system from the saved pickle files dynamically at runtime.
    @classmethod
    def load(cls, path: str = BM25_INDEX_PATH) -> "BM25KeywordIndex":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 index not found at '{path}'. "
                "Run: python src/medical_chatbot/rag/build_index.py"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded BM25 index ({len(data['metadata']):,} documents).")
        return cls(bm25=data["bm25"], metadata=data["metadata"])

    # ── Search ─────────────────────────────────────────

    # Description: Our sparse retrieval entrypoint! Supply a query, and it returns the top chunks that contain the exact requested words.
    def search(self, query: str, k: int = TOP_K) -> List[Tuple[dict, float]]:
        """
        Search the BM25 index.

        Returns:List of (metadata_dict, bm25_score) tuples, sorted descending.
        """
        query_tokens = _tokenize_arabic(query)
        if not query_tokens:
            return []

        scores  = self.bm25.get_scores(query_tokens)
        top_idx = scores.argsort()[::-1][:k]

        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append((self.metadata[idx], float(scores[idx])))

        return results


# ── Singleton loader ───────────────────────────────────

_bm25_index: BM25KeywordIndex | None = None


# Description: Another singleton! Returns the active global BM25 Index, loading it fresh purely on the first call.
def get_bm25_index() -> BM25KeywordIndex:
    """Return the global BM25 index (loaded once from disk)."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25KeywordIndex.load()
    return _bm25_index
