"""
cache.py
────────────────────────────────────────────────
Thread-safe in-memory LRU cache for Arabic medical query results.

Cache key = "{mode}:{query_intent}:{normalized_query}"
  • mode         — retrieval mode (rag / bm25 / hybrid / all / internet)
  • query_intent — extracted from disease_entity_extractor (symptoms /
                   symptom_description / clinical_history / causes /
                   treatment / … / general)
  • normalized_query — diacritics stripped, whitespace collapsed, lowercased

Using intent in the key means:
  "ما هي أعراض السكري"  (symptoms)  ≠  "ما علاج السكري"  (treatment)
so users asking the same disease but different aspects get the right answer.

Configuration via env vars:
  CACHE_ENABLED   "true" / "false"  (default: "true")
  CACHE_MAX_SIZE  integer           (default: 1000)
  CACHE_TTL_SECS  integer seconds   (default: 7200 = 2 h)
"""

from __future__ import annotations

import os
import re
import time
import threading
from collections import OrderedDict
from typing import Optional

CACHE_ENABLED  = os.getenv("CACHE_ENABLED",  "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL_SECS = int(os.getenv("CACHE_TTL_SECS", "7200"))


# ─────────────────────────────────────────────────────
#  Normalizer
# ─────────────────────────────────────────────────────

def _normalize(query: str) -> str:
    """Canonical form: strip tashkeel, collapse whitespace, lowercase."""
    text = re.sub(r"[\u064B-\u065F]", "", query.strip())
    text = re.sub(r"\s+", " ", text).lower()
    return text


# ─────────────────────────────────────────────────────
#  Cache class
# ─────────────────────────────────────────────────────

class QueryCache:
    """
    Thread-safe LRU cache with per-entry TTL.

    Entries that exceed CACHE_TTL_SECS are treated as expired on the
    next read; a background sweep is not needed given typical traffic.
    """

    def __init__(
        self,
        max_size: int = CACHE_MAX_SIZE,
        ttl: int = CACHE_TTL_SECS,
    ) -> None:
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size
        self.ttl = ttl
        self._hits   = 0
        self._misses = 0

    # ── Key construction ──────────────────────────────────────────────────────

    def _key(self, query: str, mode: str, intent: str) -> str:
        return f"{mode}:{intent}:{_normalize(query)}"

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self, query: str, mode: str = "hybrid", intent: str = "general"
    ) -> Optional[str]:
        """Return cached result string or None if absent / expired."""
        if not CACHE_ENABLED:
            return None
        key = self._key(query, mode, intent)
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, ts = self._store[key]
            if time.time() - ts > self.ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)   # refresh LRU position
            self._hits += 1
            return value

    def put(
        self, query: str, mode: str, intent: str, result: str
    ) -> None:
        """Store a result; evicts the oldest entry when at capacity."""
        if not CACHE_ENABLED:
            return
        key = self._key(query, mode, intent)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (result, time.time())
            if len(self._store) > self.max_size:
                self._store.popitem(last=False)  # evict LRU

    def invalidate(self, query: str, mode: str, intent: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        key = self._key(query, mode, intent)
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Flush the entire cache and reset counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        with self._lock:
            return {
                "enabled":     CACHE_ENABLED,
                "size":        len(self._store),
                "max_size":    self.max_size,
                "hits":        self._hits,
                "misses":      self._misses,
                "hit_rate":    round(self._hits / total, 3) if total else 0.0,
                "ttl_seconds": self.ttl,
            }


# ── Singleton ──────────────────────────────────────────────────────────────────

_cache: QueryCache | None = None


def get_cache() -> QueryCache:
    """Return the process-wide cache instance (created once)."""
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache
