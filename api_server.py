"""
api_server.py
────────────────────────────────────────────────
FastAPI REST API for the Arabic Medical Chatbot.

Provides API key-authenticated endpoints for querying the medical
chatbot, checking health, and listing medical categories.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_server")


# ─────────────────────────────────────────────────────
#  API Key Authentication
# ─────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_keys() -> set[str]:
    """Load valid API keys from env or api_keys.json file."""
    keys: set[str] = set()

    # From environment variable (comma-separated)
    keys_str = os.getenv("API_KEYS", "")
    keys.update(k.strip() for k in keys_str.split(",") if k.strip())

    # From api_keys.json file
    key_file = os.path.join(os.path.dirname(__file__), "api_keys.json")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                keys.update(file_data.get("keys", []))
        except Exception as e:
            logger.warning(f"Failed to load api_keys.json: {e}")

    if not keys:
        logger.warning(
            "No API keys configured! Set API_KEYS env var or create api_keys.json. "
            "The API will reject all authenticated requests."
        )

    return keys


VALID_KEYS = _load_api_keys()


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Validate the X-API-Key header."""
    if not api_key or api_key not in VALID_KEYS:
        raise HTTPException(
            status_code=401,
            detail="مفتاح API غير صالح أو مفقود. تأكد من تضمين X-API-Key في الرأس.",
        )
    return api_key


# ─────────────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Medical query request body."""
    query: str = Field(
        ..., min_length=2, max_length=1000,
        description="سؤال طبي بالعربية",
        examples=["ما هي أعراض مرض السكري؟"],
    )
    mode: str = Field(
        default="hybrid",
        description="وضع الاسترجاع: rag | bm25 | internet | hybrid | all",
    )
    history: list[dict] = Field(
        default=[],
        description="سياق المحادثة السابقة (اختياري)",
    )


class QueryResponse(BaseModel):
    """Medical query response."""
    answer: str
    meta: dict = {}
    disclaimer: str = "⚕️ هذه المعلومات لأغراض تعليمية فقط ولا تُغني عن استشارة الطبيب المختص."


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    embedding_model: str
    index_loaded: bool
    version: str = "1.0.0"


# ─────────────────────────────────────────────────────
#  Application Lifespan
# ─────────────────────────────────────────────────────

_crew = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the chatbot pipeline on startup."""
    global _crew
    logger.info("Loading Arabic Medical Chatbot pipeline...")
    try:
        from src.medical_chatbot.crew import arabic_chatbot
        _crew = arabic_chatbot()
        logger.info("Pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        _crew = None
    yield


# ─────────────────────────────────────────────────────
#  FastAPI Application
# ─────────────────────────────────────────────────────

app = FastAPI(
    title="Arabic Medical AI API",
    description="واجهة برمجة تطبيقات المساعد الطبي العربي — Arabic Medical Chatbot API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Public health check endpoint (no API key required)."""
    return HealthResponse(
        status="healthy" if _crew is not None else "degraded",
        model=os.getenv("OPENROUTER_MODEL", os.getenv("OLLAMA_MODEL", "unknown")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "aubmindlab/bert-base-arabertv2"),
        index_loaded=_crew is not None,
        version="1.0.3",
    )


@app.get("/api/v1/categories", dependencies=[Depends(verify_api_key)])
async def list_categories():
    """List available medical categories."""
    from src.medical_chatbot.tools.classifier_tool import CATEGORY_ARABIC_LABELS
    return {
        "categories": [
            {"key": k, "label_ar": v}
            for k, v in CATEGORY_ARABIC_LABELS.items()
        ]
    }


@app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_medical(req: QueryRequest):
    """
    Query the Arabic Medical Chatbot.

    Send a medical question in Arabic and receive an AI-generated response
    grounded in retrieved medical knowledge.
    """
    if _crew is None:
        raise HTTPException(
            status_code=503,
            detail="النظام قيد التحميل. يرجى المحاولة مرة أخرى بعد قليل.",
        )

    # Validate mode
    valid_modes = {"rag", "bm25", "internet", "hybrid", "all"}
    if req.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"وضع الاسترجاع غير صالح: {req.mode}. الأوضاع المتاحة: {', '.join(valid_modes)}",
        )

    try:
        t0 = time.time()
        logger.info(f"Processing query: '{req.query[:50]}...' (mode={req.mode})")

        raw_result = _crew.run(
            req.query,
            history=req.history,
            mode=req.mode,
        )
        elapsed = round(time.time() - t0, 1)

        # Parse the crew response
        try:
            data = json.loads(raw_result)
            answer = data.get("final_answer", raw_result)
            meta = data.get("meta", {})
        except (json.JSONDecodeError, TypeError):
            answer = raw_result
            meta = {}

        meta["elapsed"] = elapsed
        logger.info(f"Query completed in {elapsed}s (answer_len={len(answer)})")

        return QueryResponse(answer=answer, meta=meta)

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ أثناء معالجة السؤال: {str(e)}",
        )


# ─────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
