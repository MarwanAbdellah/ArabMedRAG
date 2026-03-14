"""
hybrid_search_tool.py
────────────────────────────────────────────────
Hybrid retrieval tool: Vector (FAISS) + Keyword (BM25)

Optimized fast-inference flow:
  1. Run FAISS vector search
  2. If top similarity < VECTOR_THRESHOLD → also run BM25
  3. Merge + deduplicate results
  4. Return top-K chunks as JSON
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TOP_K             = int(os.getenv("TOP_K", "5"))
VECTOR_THRESHOLD  = float(os.getenv("VECTOR_THRESHOLD", "0.50"))


# Description: Pydantic format for the hybrid search query.
class HybridSearchInput(BaseModel):
    query: str = Field(..., description="The Arabic medical query to retrieve documents for.")


# Description: The crown jewel of retrieval! Combines semantic FAISS and keyword BM25 into an intelligent, threshold-managed lookup mechanism.
class HybridSearchTool(BaseTool):
    name: str        = "hybrid_search_tool"
    description: str = (
        "Retrieve the most relevant Arabic medical knowledge for a query. "
        "Uses FAISS vector search first; if confidence is low, also runs "
        "BM25 keyword search. Returns top-5 document chunks as JSON."
    )
    args_schema: type[BaseModel] = HybridSearchInput

    # Description: Executes semantic lookup first! If confidence is below `RELEVANCE_THRESHOLD`, we seamlessly launch BM25 and organically merge the results.
    def _run(self, query: str) -> str:
        from src.medical_chatbot.rag.embedding_pipeline import get_embedder
        from src.medical_chatbot.rag.vector_store import get_vector_store
        from src.medical_chatbot.rag.keyword_index import get_bm25_index

        relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.80"))

        # ── Step 1: Vector search ──────────────────────
        embedder     = get_embedder()
        vector_store = get_vector_store()
        query_vec    = embedder.embed_query(query)
        
        # Retrieve more candidates to account for filtering
        search_k     = max(TOP_K * 3, 10)
        vector_hits  = vector_store.search(query_vec, k=search_k)

        top_score = vector_hits[0][1] if vector_hits else 0.0
        logger.info(f"Vector search top score: {top_score:.3f} (cut-off: {relevance_threshold})")

        results: dict[int, tuple[dict, float]] = {}
        for meta, score in vector_hits:
            if score >= relevance_threshold:
                results[meta["chunk_id"]] = (meta, score)

        # ── Step 2: BM25 fallback (conditional) ────────
        used_bm25 = False
        if top_score < VECTOR_THRESHOLD or len(results) < TOP_K:
            logger.info("Insufficient highly relevant results → running BM25 fallback ...")
            bm25_index  = get_bm25_index()
            bm25_hits   = bm25_index.search(query, k=search_k)
            used_bm25   = True

            for meta, score in bm25_hits:
                cid = meta["chunk_id"]
                normalized_score = min(score / 20.0, 1.0)
                if cid not in results and normalized_score >= relevance_threshold:
                    results[cid] = (meta, normalized_score)

        # ── Step 3: Rank + return top K ─────────────────
        ranked = sorted(results.values(), key=lambda x: x[1], reverse=True)[:TOP_K]

        chunks = []
        for rank, (meta, score) in enumerate(ranked, start=1):
            chunks.append({
                "rank":        rank,
                "score":       round(score, 4),
                "category_en": meta.get("category_en", ""),
                "category":    meta.get("category",    ""),
                "question":    meta.get("question",    ""),
                "text":        meta.get("text", "")[:int(os.getenv("MAX_CHUNK_CHARS", "600"))],
            })

        output = {
            "query":            query,
            "total_results":    len(chunks),
            "used_bm25_fallback": used_bm25,
            "top_similarity":   round(top_score, 4),
            "chunks":           chunks,
        }

        logger.info(f"Hybrid search returned {len(chunks)} chunks (BM25 used: {used_bm25}).")
        return json.dumps(output, ensure_ascii=False, indent=2)
