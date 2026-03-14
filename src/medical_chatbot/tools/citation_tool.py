"""
citation_tool.py
────────────────────────────────────────────────
Formats retrieved chunks into numbered citations
that the Arabic Medical Response Agent can embed
directly into its answer.
"""

from __future__ import annotations

import json
import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Description: Pydantic schema enforcing exactly what arguments the LLM gives this tool.
class CitationInput(BaseModel):
    retrieved_json: str = Field(
        ...,
        description=(
            "JSON string output from hybrid_search_tool, "
            "containing a 'chunks' list."
        ),
    )


# Description: A CrewAI tool bridging the gap. It nicely formats our retrieved search results so the Arabic Answer Agent can easily inject numerical citations.
class CitationGroundingTool(BaseTool):
    name: str        = "citation_grounding_tool"
    description: str = (
        "Accept retrieved medical document chunks and format them "
        "as numbered Arabic citations. Returns citation text and a "
        "context block for the response agent."
    )
    args_schema: type[BaseModel] = CitationInput

    # Description: Triggered implicitly by the overarching CrewAI backend when this tool is selected by an agent.
    def _run(self, retrieved_json: str) -> str:
        try:
            data = json.loads(retrieved_json)
        except json.JSONDecodeError:
            return json.dumps(
                {"error": "Invalid JSON input to citation_grounding_tool"},
                ensure_ascii=False,
            )

        chunks = data.get("chunks", [])
        if not chunks:
            return json.dumps(
                {"citations": [], "context": "لا توجد مصادر مسترجعة."},
                ensure_ascii=False,
            )

        citations    = []
        context_parts = []

        for chunk in chunks:
            rank    = chunk.get("rank", "?")
            cat_en  = chunk.get("category_en", "General Medicine")
            text    = chunk.get("text", "")[:400]   # snippet for display

            citation_label = f"[{rank}] {cat_en}"
            citations.append({
                "number":   rank,
                "source":   cat_en,
                "snippet":  text,
                "label":    citation_label,
            })
            context_parts.append(f"[{rank}] ({cat_en})\n{chunk.get('text', '')}")

        context_block = "\n\n".join(context_parts)

        # Build Arabic citation list for embedding in response
        arabic_citation_list = "\n".join(
            f"{c['number']}. {c['source']}" for c in citations
        )

        output = {
            "citations":            citations,
            "arabic_citation_list": arabic_citation_list,
            "context":              context_block,
        }
        logger.info(f"Formatted {len(citations)} citations.")
        return json.dumps(output, ensure_ascii=False, indent=2)
