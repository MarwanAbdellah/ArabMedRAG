"""
language_detection_tool.py
────────────────────────────────────────────────
Detects the language of the user query using langdetect.
Always signals that the response must be in Arabic.
"""

from __future__ import annotations

import json
import logging

from crewai.tools import BaseTool
from langdetect import LangDetectException, detect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LANG_MAP = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh-cn": "Chinese",
    "tr": "Turkish",
}


# Description: Enforces payload scheme for the detecting logic.
class LanguageDetectionInput(BaseModel):
    query: str = Field(..., description="The user's medical query text.")


# Description: A hyper-simple wrapper bridging standard Python `langdetect` logic into our complex CrewAI framework.
class LanguageDetectionTool(BaseTool):
    name: str        = "language_detection_tool"
    description: str = (
        "Detect the language of a user query. "
        "Returns a JSON object with 'detected_language' and 'response_language'. "
        "The response language is always Arabic regardless of input language."
    )
    args_schema: type[BaseModel] = LanguageDetectionInput

    # Description: Forces all answers to inherently be handled in Arabic regardless of original input.
    def _run(self, query: str) -> str:
        try:
            lang_code = detect(query)
            detected  = LANG_MAP.get(lang_code, lang_code.upper())
        except LangDetectException:
            detected  = "Unknown"

        result = {
            "query":              query,
            "detected_language":  detected,
            "response_language":  "Arabic",
            "note":               "The system always responds in Arabic regardless of input language.",
        }
        logger.info(f"Language detected: {detected}")
        return json.dumps(result, ensure_ascii=False)
