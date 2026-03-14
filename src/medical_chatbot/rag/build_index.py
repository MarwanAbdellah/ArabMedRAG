"""
build_index.py
────────────────────────────────────────────────
One-shot CLI script to build and save both indexes:
  1. FAISS vector index (dense retrieval)
  2. BM25 keyword index (sparse retrieval fallback)

Usage:
    # Full dataset (may take 15-30 min on CPU)
    python src/medical_chatbot/rag/build_index.py

    # Quick test with 1000 rows
    python src/medical_chatbot/rag/build_index.py --sample 1000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from src.medical_chatbot.rag.document_loader   import load_documents
from src.medical_chatbot.rag.chunking           import chunk_documents
from src.medical_chatbot.rag.embedding_pipeline import get_embedder
from src.medical_chatbot.rag.vector_store       import FAISSVectorStore
from src.medical_chatbot.rag.keyword_index      import BM25KeywordIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


# Description: This function grabs whatever flags you passed in from your terminal (like --sample 1000) so we can adjust the script's behavior.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 medical indexes.")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Number of rows to sample (0 = full dataset)"
    )
    parser.add_argument(
        "--data",   type=str, default=os.getenv("DATA_PATH", "data/concatenated_df.csv"),
        help="Path to the CSV dataset"
    )
    parser.add_argument(
        "--faiss",  type=str, default=os.getenv("FAISS_INDEX_PATH", "indexes/faiss_index.bin"),
        help="Output path for FAISS index"
    )
    parser.add_argument(
        "--bm25",   type=str, default=os.getenv("BM25_INDEX_PATH", "indexes/bm25_index.pkl"),
        help="Output path for BM25 index"
    )
    parser.add_argument(
        "--meta",   type=str, default=os.getenv("METADATA_PATH",   "indexes/metadata.pkl"),
        help="Output path for FAISS metadata"
    )
    return parser.parse_args()


# Description: Our indexing pipeline starts here! It reads the raw data, creates vector embeddings, and builds a BM25 keyword index so our chatbot can search lightning fast later.
def main() -> None:
    args   = parse_args()
    t0     = time.time()

    # ── Step 1: Load documents ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 / 5  –  Loading documents")
    logger.info("=" * 60)
    documents = load_documents(args.data, sample=args.sample)
    logger.info(f"Loaded {len(documents):,} documents.")

    # ── Step 2: Chunk documents ────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 / 5  –  Chunking documents")
    logger.info("=" * 60)
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks):,} chunks.")

    # ── Step 3: Build metadata list ────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 / 5  –  Building metadata list")
    logger.info("=" * 60)
    metadata = [
        {
            "chunk_id":    c.chunk_id,
            "doc_id":      c.doc_id,
            "text":        c.text,
            "question":    c.question,
            "category":    c.category,
            "category_en": c.category_en,
        }
        for c in chunks
    ]
    logger.info(f"Prepared metadata for {len(metadata):,} chunks.")

    # ── Step 4: Build & save FAISS index ──────────────
    logger.info("=" * 60)
    logger.info("STEP 4 / 5  –  Embedding (SentenceTransformers) + FAISS index build")
    logger.info("=" * 60)
    embedder = get_embedder()
    texts    = [m["text"] for m in metadata]
    vectors  = embedder.embed(texts)

    faiss_store = FAISSVectorStore(embedding_dim=embedder.embedding_dim)
    faiss_store.build(vectors, metadata)
    faiss_store.save(index_path=args.faiss, meta_path=args.meta)

    # ── Step 5: Build & save BM25 index ───────────────
    logger.info("=" * 60)
    logger.info("STEP 5 / 5  –  BM25 keyword index build")
    logger.info("=" * 60)
    bm25_index = BM25KeywordIndex.build(metadata)
    bm25_index.save(path=args.bm25)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"✅  Index build complete in {elapsed / 60:.1f} min.")
    logger.info(f"   FAISS  → {args.faiss}")
    logger.info(f"   BM25   → {args.bm25}")
    logger.info(f"   Meta   → {args.meta}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
