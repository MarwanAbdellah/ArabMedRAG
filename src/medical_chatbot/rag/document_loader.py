"""
document_loader.py
────────────────────────────────────────────────
Loads the Arabic medical Q&A CSV dataset.

Dataset columns:
    q_body       – Arabic question text
    a_body       – Arabic answer text
    category     – Arabic category label
    category_en  – English category label
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────

# Description: A structured represention of a single question-and-answer pair from our dataset, complete with category metadata.
@dataclass
class MedicalDocument:
    """A single medical Q&A document."""
    doc_id: int
    text: str          # Combined question + answer
    question: str
    answer: str
    category: str      # Arabic category
    category_en: str   # English category


# ─────────────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────────────

# Description: Reads our massive CSV dataset file and transforms each row into usable `MedicalDocument` objects for downstream processing.
def load_documents(
    csv_path: str,
    sample: Optional[int] = None,
    min_answer_len: int = 20,
) -> List[MedicalDocument]:
    """
    Load medical Q&A documents from CSV.

    Args:csv_path:       Path to concatenated_df.csv
        sample:         If set, randomly sample this many rows (for testing).
        min_answer_len: Drop rows where the answer is too short.

    Returns:
        List of MedicalDocument objects.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    logger.info(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)

    # Drop rows with missing critical fields
    df = df.dropna(subset=["q_body", "a_body"])

    # Fill optional fields
    df["category"]    = df["category"].fillna("عام")
    df["category_en"] = df["category_en"].fillna("General")

    # Filter very short answers (noise)
    df = df[df["a_body"].str.len() >= min_answer_len]

    # Optional sampling (useful for index building tests)
    if sample and sample > 0:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        logger.info(f"Sampled {len(df)} rows.")
    else:
        logger.info(f"Loaded {len(df):,} rows.")

    documents: List[MedicalDocument] = []
    for idx, row in enumerate(df.itertuples(index=False)):
        q = str(row.q_body).strip()
        a = str(row.a_body).strip()
        combined = f"السؤال: {q}\nالجواب: {a}"
        documents.append(
            MedicalDocument(
                doc_id=idx,
                text=combined,
                question=q,
                answer=a,
                category=str(row.category).strip(),
                category_en=str(row.category_en).strip(),
            )
        )

    return documents
