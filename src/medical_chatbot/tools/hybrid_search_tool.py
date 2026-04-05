"""
hybrid_search_tool.py
────────────────────────────────────────────────
Hybrid retrieval tool: Vector (FAISS) + Keyword (BM25)
with entity-boosted search for Arabic disease names.

Optimized flow:
  1. Extract disease entity from query
  2. Run FAISS vector search (full query + entity-focused)
  3. If top similarity < VECTOR_THRESHOLD → also run BM25
  4. Merge + deduplicate results (entity-match chunks get 1.3x boost)
  5. Return top-K chunks as JSON
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


def _entity_in_result(disease_entity: str, meta: dict) -> bool:
    """Return True if key words from disease_entity appear in the result question or text.

    Uses 5-char threshold to exclude short Arabic grammar words (e.g. مرض=4 chars)
    while matching meaningful disease terms (e.g. السكر=6, السرطان=7).
    """
    content_words: list[str] = []
    for word in disease_entity.split():
        if len(word) >= 5:
            content_words.append(word)
            # Also try root without definite article ال
            if word.startswith("ال") and len(word) > 4:
                root = word[2:]
                if len(root) >= 3:
                    content_words.append(root)
    if not content_words:
        return True  # entity too short to be discriminative — allow all
    combined = (meta.get("question", "") + " " + meta.get("text", "")).lower()
    return any(w.lower() in combined for w in content_words)


VECTOR_THRESHOLD  = float(os.getenv("VECTOR_THRESHOLD", "0.50"))
ENTITY_BOOST      = float(os.getenv("ENTITY_BOOST", "1.3"))


class HybridSearchInput(BaseModel):
    query: str = Field(..., description="The Arabic medical query to retrieve documents for.")


class HybridSearchTool(BaseTool):
    name: str        = "hybrid_search_tool"
    description: str = (
        "Retrieve the most relevant Arabic medical knowledge for a query. "
        "Uses FAISS vector search first; if confidence is low, also runs "
        "BM25 keyword search. Includes entity-boosted search for disease names. "
        "Returns top-5 document chunks as JSON."
    )
    args_schema: type[BaseModel] = HybridSearchInput

    def _run(self, query: str) -> str:
        from src.medical_chatbot.rag.embedding_pipeline import get_embedder
        from src.medical_chatbot.rag.vector_store import get_vector_store
        from src.medical_chatbot.rag.keyword_index import get_bm25_index
        from src.medical_chatbot.tools.disease_entity_extractor import extract_disease_entity

        relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.80"))

        # ── Step 0: Extract disease entity ─────────────
        entity_info = extract_disease_entity(query)
        disease_entity = entity_info.get("disease_entity", "")
        has_entity = (
            disease_entity
            and disease_entity != query
            and entity_info.get("extraction_method") == "pattern"
        )
        logger.info(f"Disease entity: '{disease_entity}' (has_entity={has_entity})")

        # ── Step 1: Vector search (full query) ─────────
        embedder     = get_embedder()
        vector_store = get_vector_store()
        query_vec    = embedder.embed_query(query)
        
        search_k     = max(TOP_K * 3, 10)
        vector_hits  = vector_store.search(query_vec, k=search_k)

        top_score = vector_hits[0][1] if vector_hits else 0.0
        logger.info(f"Vector search top score: {top_score:.3f} (cut-off: {relevance_threshold})")

        results: dict[int, tuple[dict, float]] = {}
        for meta, score in vector_hits:
            if score >= relevance_threshold:
                results[meta["chunk_id"]] = (meta, score)

        # ── Step 1b: Entity-focused vector search ──────
        entity_chunk_ids: set[int] = set()
        if has_entity:
            entity_vec = embedder.embed_query(disease_entity)
            entity_hits = vector_store.search(entity_vec, k=search_k)
            logger.info(f"Entity search for '{disease_entity}': {len(entity_hits)} hits")

            for meta, score in entity_hits:
                cid = meta["chunk_id"]
                entity_chunk_ids.add(cid)
                # Apply entity boost
                boosted_score = score * ENTITY_BOOST
                if cid in results:
                    # Keep the higher score between full-query and entity-boosted
                    existing_score = results[cid][1]
                    if boosted_score > existing_score:
                        results[cid] = (meta, boosted_score)
                elif score >= relevance_threshold * 0.7:
                    # Lower threshold for entity-matched chunks
                    results[cid] = (meta, boosted_score)

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
                # Boost BM25 results that match the entity
                if has_entity and cid in entity_chunk_ids:
                    normalized_score *= ENTITY_BOOST
                # When entity is known, reject BM25 results that don't mention it
                # (prevents e.g. "مرض السل" from appearing in a "السكر" query)
                if has_entity and cid not in entity_chunk_ids:
                    if not _entity_in_result(disease_entity, meta):
                        continue
                if cid not in results and normalized_score >= relevance_threshold:
                    results[cid] = (meta, normalized_score)

            # Also run BM25 with just the entity if we have one
            if has_entity:
                entity_bm25_hits = bm25_index.search(disease_entity, k=search_k)
                for meta, score in entity_bm25_hits:
                    cid = meta["chunk_id"]
                    normalized_score = min(score / 20.0, 1.0) * ENTITY_BOOST
                    if cid not in results and normalized_score >= relevance_threshold * 0.7:
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
                "entity_match": meta["chunk_id"] in entity_chunk_ids,
            })

        output = {
            "query":              query,
            "disease_entity":     disease_entity if has_entity else None,
            "total_results":      len(chunks),
            "used_bm25_fallback": used_bm25,
            "top_similarity":     round(top_score, 4),
            "chunks":             chunks,
        }

        logger.info(f"Hybrid search returned {len(chunks)} chunks (BM25 used: {used_bm25}, entity: {disease_entity}).")
        return json.dumps(output, ensure_ascii=False, indent=2)
