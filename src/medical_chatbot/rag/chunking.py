"""
chunking.py
────────────────────────────────────────────────
Splits MedicalDocuments into fixed-size token chunks
using tiktoken (cl100k_base tokenizer).

Chunk size : 500 tokens
Overlap    : 100 tokens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import tiktoken

from .document_loader import MedicalDocument

logger = logging.getLogger(__name__)

CHUNK_SIZE   = 500   # tokens
CHUNK_OVERLAP = 100  # tokens


# ─────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────

# Description: A simple container that holds a specific slice of a medical document.
@dataclass
class TextChunk:
    """A text chunk derived from a MedicalDocument."""
    chunk_id:    int
    doc_id:      int
    text:        str
    question:    str       # original question (for citation display)
    category_en: str
    category:    str


# ─────────────────────────────────────────────────────
#  Chunker
# ─────────────────────────────────────────────────────

# Description: This class handles the heavy lifting of taking a massive document and cleanly cutting it up into searchable, overlapping text blocks.
class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap    = overlap
        # cl100k_base works for Arabic text (byte-pair encoding)
        self.enc = tiktoken.get_encoding("cl100k_base")

    # Description: A low-level method that slides a token window over a long array to ensure we don't lose context between chunks.
    def _chunk_tokens(self, tokens: List[int]) -> List[List[int]]:
        """Split a token list into overlapping windows."""
        chunks = []
        start  = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            if end == len(tokens):
                break
            start += self.chunk_size - self.overlap
        return chunks

    # Description: The public dispatcher! Give it a list of raw documents, and it will return a massive list of properly-sized text chunks.
    def chunk_documents(self, documents: List[MedicalDocument]) -> List[TextChunk]:
        """
        Chunk all documents into overlapping token windows.

        For short Q&A pairs that fit within chunk_size, a single chunk
        is produced (no splitting needed).
        """
        all_chunks:List[TextChunk] = []
        chunk_id = 0

        for doc in documents:
            tokens = self.enc.encode(doc.text)

            if len(tokens) <= self.chunk_size:
                # Short document → single chunk (most Q&A pairs)
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=doc.text,
                        question=doc.question,
                        category_en=doc.category_en,
                        category=doc.category,
                    )
                )
                chunk_id += 1
            else:
                # Long document → windowed chunks
                windows = self._chunk_tokens(tokens)
                for window in windows:
                    chunk_text = self.enc.decode(window)
                    all_chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            text=chunk_text,
                            question=doc.question,
                            category_en=doc.category_en,
                            category=doc.category,
                        )
                    )
                    chunk_id += 1

        logger.info(f"Produced {len(all_chunks):,} chunks from {len(documents):,} documents.")
        return all_chunks


# Description: The public dispatcher! Give it a list of raw documents, and it will return a massive list of properly-sized text chunks.
def chunk_documents(
    documents: List[MedicalDocument],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Convenience function wrapping the Chunker class."""
    return Chunker(chunk_size=chunk_size, overlap=overlap).chunk_documents(documents)
