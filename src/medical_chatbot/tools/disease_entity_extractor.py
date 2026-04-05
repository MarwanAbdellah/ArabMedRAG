"""
disease_entity_extractor.py
────────────────────────────────────────────────
Understands *what the user wants* from an Arabic medical query and
extracts the most useful search key for the retrieval pipeline.

Three query archetypes are handled:

1. Disease-focused question
   "ما هي أعراض مرض السكري؟"
   → entity = "السكري"  |  intent = "symptoms"

2. Patient self-describing symptoms
   "أعاني من ألم في الصدر وضيق في التنفس منذ يومين"
   → entity = "ألم في الصدر وضيق في التنفس"  |  intent = "symptom_description"

3. Doctor / clinical history
   "مريض عمره ٥٠ سنة يشكو من ارتفاع في الضغط والدوخة"
   → entity = "ارتفاع في الضغط والدوخة"  |  intent = "clinical_history"

For archetypes 2 & 3 the full symptom / complaint phrase is kept intact
so the retrieval layer can match the complete clinical picture.
"""

from __future__ import annotations

import json
import logging
import re

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
#  Intent patterns  (order matters — most specific first)
# ─────────────────────────────────────────────────────

# Each entry: (regex, intent, keep_full_phrase)
#   keep_full_phrase=True  → do NOT strip stopwords; return the whole matched group
#   keep_full_phrase=False → strip Arabic stopwords (disease-name queries)

INTENT_PATTERNS: list[tuple[str, str, bool]] = [

    # ── Clinical / doctor-style patient history ───────────────────────────────
    # "مريض/مريضة عمره/عمرها X يشكو/تشكو من Y"
    (r"مريض(?:ة)?\s+(?:\S+\s+){0,6}(?:يشكو|تشكو|يعاني|تعاني)\s+من\s+(.+?)(?:\?|؟|$)", "clinical_history", True),
    # "سيدة / رجل يشكو من Y"
    (r"(?:سيدة|رجل|طفل|طفلة|مريض|مريضة)\s+(?:.{0,40})\s+(?:يشكو|تشكو|يعاني|تعاني)\s+من\s+(.+?)(?:\?|؟|$)", "clinical_history", True),
    # "مريض يعاني من X" (shorter form)
    (r"(?:مريض|مريضة)\s+(?:يعاني|تعاني)\s+من\s+(.+?)(?:\?|؟|$)", "clinical_history", True),

    # ── Patient self-description of symptoms ─────────────────────────────────
    # "أعاني / اعاني من X"
    (r"(?:أعاني|اعاني)\s+من\s+(.+?)(?:\?|؟|$)", "symptom_description", True),
    # "عندي / عندي X"
    (r"(?:عندي|عندي)\s+(.+?)(?:\?|؟|$)", "symptom_description", True),
    # "لدي X"
    (r"لدي\s+(.+?)(?:\?|؟|$)", "symptom_description", True),
    # "أشعر بـ / أحس بـ X"
    (r"(?:أشعر|اشعر|أحس|احس)\s+(?:بـ?|ب)\s*(.+?)(?:\?|؟|$)", "symptom_description", True),
    # "يؤلمني X / يؤلم X"
    (r"يؤلم(?:ني|ك|ه|ها)?\s+(.+?)(?:\?|؟|$)", "symptom_description", True),
    # "ظهر لي / بدأت أشعر بـ"
    (r"(?:ظهر(?:ت)?\s+لي|بدأت\s+أشعر\s+بـ?)\s+(.+?)(?:\?|؟|$)", "symptom_description", True),
    # "أجد صعوبة في X"
    (r"أجد\s+صعوبة\s+في\s+(.+?)(?:\?|؟|$)", "symptom_description", True),

    # ── Disease-focused questions ─────────────────────────────────────────────
    # "ما هي / ما هو أعراض X"
    (r"ما\s+(?:هي|هو|هن)\s+(?:أعراض|اعراض)\s+(.+?)(?:\?|؟|$)", "symptoms", False),
    # "ما أعراض X" (short)
    (r"ما\s+(?:أعراض|اعراض)\s+(.+?)(?:\?|؟|$)", "symptoms", False),
    # "ما هي أنواع / انواع X"
    (r"ما\s+(?:هي|هو)\s+(?:أنواع|انواع)\s+(?:مرض\s+)?(.+?)(?:\?|؟|$)", "types", False),
    (r"ما\s+(?:أنواع|انواع)\s+(?:مرض\s+)?(.+?)(?:\?|؟|$)", "types", False),
    # "ما هي أسباب X" / "ما أسباب X"
    (r"ما\s+(?:هي|هو)\s+أسباب\s+(.+?)(?:\?|؟|$)", "causes", False),
    (r"ما\s+أسباب\s+(.+?)(?:\?|؟|$)", "causes", False),
    # "ما هو علاج X" / "ما علاج X"
    (r"ما\s+(?:هو|هي)\s+علاج\s+(.+?)(?:\?|؟|$)", "treatment", False),
    (r"ما\s+علاج\s+(.+?)(?:\?|؟|$)", "treatment", False),
    # "كيف يتم تشخيص / علاج X"
    (r"كيف\s+(?:يتم\s+)?(?:تشخيص|علاج|الوقاية\s+من)\s+(.+?)(?:\?|؟|$)", "diagnosis", False),
    # "ما هو مرض X" / "ما هي X"
    (r"ما\s+(?:هو|هي)\s+(?:مرض\s+)?(.+?)(?:\?|؟|$)", "definition", False),
    # "هل X خطير / معدي / وراثي"
    (r"هل\s+(.+?)\s+(?:خطير|معدي|وراثي|مزمن|قاتل)", "severity", False),
    # "ما مضاعفات X"
    (r"ما\s+(?:هي\s+)?مضاعفات\s+(.+?)(?:\?|؟|$)", "complications", False),
    # "ما الوقاية من X"
    (r"ما\s+(?:هي\s+)?(?:طرق\s+)?الوقاية\s+(?:من\s+)?(.+?)(?:\?|؟|$)", "prevention", False),
]


# ─────────────────────────────────────────────────────
#  Arabic stopwords — only stripped for disease-name intents
# ─────────────────────────────────────────────────────

ARABIC_STOPWORDS = {
    # Question words
    "ما", "هي", "هو", "هل", "كيف", "لماذا", "متى", "أين", "من",
    # Prepositions & conjunctions
    "في", "عن", "على", "إلى", "مع", "أو", "و", "ب", "ل", "ك",
    # Demonstratives & relatives
    "هذا", "هذه", "ذلك", "تلك", "الذي", "التي",
    # Medical-question framework words (NOT disease names)
    "أعراض", "اعراض", "علاج", "أسباب", "تشخيص", "الوقاية", "مضاعفات",
    "أنواع", "انواع", "نوع",
    "طرق", "يتم", "يمكن", "كان", "ليس", "قد", "لا",
    # Single-char noise
    "أ", "ا", "ي",
}


def _strip_trailing(text: str) -> str:
    """Remove trailing Arabic/Latin punctuation."""
    return re.sub(r"[؟\?\.!،,]+$", "", text).strip()


def extract_disease_entity(query: str) -> dict:
    """
    Extract the medical entity and query intent from an Arabic query.

    Returns a dict with:
        disease_entity   — the extracted search term / phrase
        query_intent     — one of: symptoms | symptom_description |
                           clinical_history | causes | treatment |
                           diagnosis | types | definition | severity |
                           complications | prevention | general
        full_query       — original query unchanged
        extraction_method — "pattern" | "fallback"
    """
    # Strip diacritics for robust matching
    normalized = re.sub(r"[\u064B-\u065F]", "", query.strip())

    for pattern, intent, keep_full in INTENT_PATTERNS:
        match = re.search(pattern, normalized, re.UNICODE)
        if match:
            entity = _strip_trailing(match.group(1).strip())

            if keep_full:
                # Symptom descriptions / clinical histories:
                # keep the whole phrase — it IS the medical concept
                if entity:
                    return {
                        "disease_entity": entity,
                        "query_intent": intent,
                        "full_query": query,
                        "extraction_method": "pattern",
                    }
            else:
                # Disease-name intents: strip framework stopwords but preserve
                # multi-word compound names (e.g. "ارتفاع ضغط الدم")
                tokens = entity.split()
                cleaned = [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
                if cleaned:
                    return {
                        "disease_entity": " ".join(cleaned),
                        "query_intent": intent,
                        "full_query": query,
                        "extraction_method": "pattern",
                    }

    # ── Fallback: remove question words, return what's left ──────────────────
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
    query: str = Field(..., description="Arabic medical query to extract entity from.")


class DiseaseEntityExtractorTool(BaseTool):
    name: str = "disease_entity_extractor"
    description: str = (
        "Extract the medical entity and intent from an Arabic query. "
        "Handles disease-name questions, patient symptom descriptions, "
        "and doctor-style clinical histories. "
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
