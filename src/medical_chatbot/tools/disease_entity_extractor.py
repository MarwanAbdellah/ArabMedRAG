"""
disease_entity_extractor.py
────────────────────────────────────────────────
Extract Arabic disease entities from medical queries.

Solves the core problem: when a user asks "ما هي اعراض مرض السكري",
the system should focus on "مرض السكري" (the disease) rather than
"ما هي اعراض" (the question pattern).

Uses regex-based pattern matching with Arabic medical question templates.
"""

from __future__ import annotations

import json
import logging
import re

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
#  Arabic question patterns → intent mapping
# ─────────────────────────────────────────────────────

INTENT_PATTERNS = [
    # "ما هي أعراض مرض X" / "ما هي اعراض X"
    (r"ما\s+(?:هي|هو|هن)\s+(?:أعراض|اعراض)\s+(.+?)(?:\?|؟|$)", "symptoms"),
    # "ما أعراض X" (short form)
    (r"ما\s+(?:أعراض|اعراض)\s+(.+?)(?:\?|؟|$)", "symptoms"),
    # "ما هي أنواع/انواع مرض X"
    (r"ما\s+(?:هي|هو)\s+(?:أنواع|انواع)\s+(?:مرض\s+)?(.+?)(?:\?|؟|$)", "types"),
    # "ما أنواع X"
    (r"ما\s+(?:أنواع|انواع)\s+(?:مرض\s+)?(.+?)(?:\?|؟|$)", "types"),
    # "ما هي أسباب X"
    (r"ما\s+(?:هي|هو)\s+أسباب\s+(.+?)(?:\?|؟|$)", "causes"),
    # "ما أسباب X"
    (r"ما\s+أسباب\s+(.+?)(?:\?|؟|$)", "causes"),
    # "ما هو علاج X"
    (r"ما\s+(?:هو|هي)\s+علاج\s+(.+?)(?:\?|؟|$)", "treatment"),
    # "ما علاج X"
    (r"ما\s+علاج\s+(.+?)(?:\?|؟|$)", "treatment"),
    # "كيف يتم تشخيص/علاج X"
    (r"كيف\s+(?:يتم\s+)?(?:تشخيص|علاج|الوقاية\s+من)\s+(.+?)(?:\?|؟|$)", "diagnosis"),
    # "ما هو مرض X" / "ما هي X"
    (r"ما\s+(?:هو|هي)\s+(?:مرض\s+)?(.+?)(?:\?|؟|$)", "definition"),
    # "هل X خطير/معدي/وراثي"
    (r"هل\s+(.+?)\s+(?:خطير|معدي|وراثي|مزمن)", "severity"),
    # "أعاني من X"
    (r"(?:أعاني|اعاني|عندي)\s+(?:من\s+)?(.+?)(?:\?|؟|$)", "symptoms"),
    # "ما مضاعفات X"
    (r"ما\s+(?:هي\s+)?مضاعفات\s+(.+?)(?:\?|؟|$)", "complications"),
    # "ما الوقاية من X"
    (r"ما\s+(?:هي\s+)?(?:طرق\s+)?الوقاية\s+(?:من\s+)?(.+?)(?:\?|؟|$)", "prevention"),
]


# ─────────────────────────────────────────────────────
#  Arabic stopwords to strip from extracted entities
# ─────────────────────────────────────────────────────

ARABIC_STOPWORDS = {
    # Question words
    "ما", "هي", "هو", "هل", "كيف", "لماذا", "متى", "أين", "من",
    # Prepositions & conjunctions
    "في", "عن", "على", "إلى", "مع", "أو", "و", "ب", "ل", "ك",
    # Demonstratives & relatives
    "هذا", "هذه", "ذلك", "تلك", "الذي", "التي",
    # Medical question framework words (NOT disease names)
    "أعراض", "اعراض", "علاج", "أسباب", "تشخيص", "الوقاية", "مضاعفات",
    "أنواع", "انواع", "نوع", "أنواع",
    "طرق", "يتم", "يمكن", "كان", "ليس", "قد", "لا",
    # Single-char noise
    "أ", "ا", "ي",
}


def extract_disease_entity(query: str) -> dict:
    """
    Extract disease entity and query intent from an Arabic medical query.

    Returns:
        dict with keys: disease_entity, query_intent, full_query, extraction_method
    """
    # Remove diacritics for matching
    normalized = re.sub(r"[\u064B-\u065F]", "", query.strip())

    for pattern, intent in INTENT_PATTERNS:
        match = re.search(pattern, normalized, re.UNICODE)
        if match:
            entity = match.group(1).strip()
            # Remove trailing punctuation
            entity = re.sub(r"[؟\?\.!]$", "", entity).strip()
            # Clean stopwords from extracted entity
            tokens = entity.split()
            cleaned = [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
            if cleaned:
                return {
                    "disease_entity": " ".join(cleaned),
                    "query_intent": intent,
                    "full_query": query,
                    "extraction_method": "pattern",
                }

    # Fallback: remove common question words, return remaining as entity
    tokens = normalized.split()
    entity_tokens = [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
    return {
        "disease_entity": " ".join(entity_tokens) if entity_tokens else query,
        "query_intent": "general",
        "full_query": query,
        "extraction_method": "fallback",
    }


# ─────────────────────────────────────────────────────
#  CrewAI Tool wrapper
# ─────────────────────────────────────────────────────

class DiseaseEntityInput(BaseModel):
    query: str = Field(..., description="Arabic medical query to extract disease entity from.")


class DiseaseEntityExtractorTool(BaseTool):
    name: str = "disease_entity_extractor"
    description: str = (
        "Extract the disease name entity from an Arabic medical query. "
        "Returns JSON with 'disease_entity' and 'query_intent' fields."
    )
    args_schema: type[BaseModel] = DiseaseEntityInput

    def _run(self, query: str) -> str:
        result = extract_disease_entity(query)
        logger.info(
            f"Extracted entity: '{result['disease_entity']}' "
            f"(intent: {result['query_intent']}, method: {result['extraction_method']})"
        )
        return json.dumps(result, ensure_ascii=False)
