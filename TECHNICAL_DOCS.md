# ArabMedRAG — Technical Documentation

**System:** Arabic Medical Question-Answering powered by Agentic RAG  
**Stack:** CrewAI · FAISS · BM25 · E5 base · Any LLM · FastAPI · Streamlit · Telegram  
**Dataset:** 341,476 Arabic medical Q&A pairs

---

## Table of Contents

1. [Overview](#1-overview)
2. [AI Lingo Glossary](#2-ai-lingo-glossary)
3. [Repository Layout](#3-repository-layout)
4. [RAG Layer](#4-rag-layer)
5. [Tools Layer](#5-tools-layer)
6. [Orchestration](#6-orchestration)
7. [Interfaces](#7-interfaces)
8. [Evaluation & Infrastructure](#8-evaluation--infrastructure)
9. [End-to-End Request Trace](#9-end-to-end-request-trace)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Verification](#11-verification)

---

## 1. Overview

ArabMedRAG is a production-grade Arabic medical chatbot that answers health questions by retrieving real Q&A pairs from a large corpus — it never fabricates medical facts from model memory alone. When a user submits a query, six specialized AI agents run in sequence: one detects language, one classifies the query (and instantly exits with an alert if an emergency is detected), one retrieves the most relevant documents using both dense-vector and keyword search, one formats citations, one generates the final Arabic answer via an LLM, and a last one checks the answer for unsupported claims. Three deployment surfaces exist: a Streamlit web UI, a FastAPI REST API, and a Telegram bot.

```
User Query (Arabic / any language)
      │
      ▼
[1] Language Detection Agent     — langdetect library → forces Arabic response
      │
      ▼
[2] Medical Classification Agent — keyword rules → 21 specialties
      │
      ├── EMERGENCY? ──► Immediate Arabic alert (bypasses LLM entirely)
      │
      ▼
[3] Disease Entity Extractor     — regex patterns → entity + intent label
      │
      ├── Cache hit? ──► Return cached answer
      │
      ▼
[4] Hybrid Retrieval Agent       — FAISS (dense) + BM25 (sparse) → top-5 chunks
      │
      ▼
[5] Citation Grounding Agent     — numbered source list + context block
      │
      ├── Context too weak? ──► Optional Serper internet search
      │
      ▼
[6] LLM Generation               — temperature=0.3, max_tokens=1200
      │
      ▼
[7] Hallucination Detection Agent — lexical coverage check
      │
      ▼
Final Arabic Medical Answer (citations + medical disclaimer)
```

---

## 2. AI Lingo Glossary

| Term | Plain-English Definition |
|------|--------------------------|
| **RAG** (Retrieval-Augmented Generation) | Instead of letting the LLM rely on memorized training data, we first *retrieve* relevant text chunks from a database, then feed those chunks into the LLM prompt so it answers based on real documents. |
| **Agentic RAG** | RAG extended with multiple autonomous *agents*, each responsible for a sub-task (retrieval, citation, hallucination check). Agents hand off results to each other in a pipeline orchestrated by CrewAI. |
| **Embedding** | A numeric vector (list of floats) that encodes the *semantic meaning* of text. Similar sentences have similar vectors regardless of exact wording. Here: 768-dimensional vectors from the E5 model. |
| **Cosine Similarity** | A distance metric between two vectors — measures the angle between them. Score of 1.0 = identical meaning, 0.0 = unrelated. Used to find the most semantically similar document chunks to a query. |
| **FAISS `IndexFlatIP`** | Facebook AI Similarity Search (FAISS) — a C++ library for fast nearest-neighbour search over millions of vectors. `IndexFlatIP` computes inner products (equivalent to cosine similarity when vectors are L2-normalized). |
| **BM25** (Best Match 25) | A classic keyword-ranking algorithm from information retrieval. Scores documents by term frequency and inverse document frequency — no neural network needed. Used as a sparse-retrieval fallback. |
| **Hybrid Retrieval** | Combining both dense (embedding/FAISS) and sparse (BM25) search. Dense finds semantically similar results; sparse finds exact keyword matches. Together they cover more cases. |
| **Chunking** | Splitting long documents into smaller overlapping windows (500 tokens, 100-token overlap) so the retrieval model doesn't have to process huge texts at once. |
| **Token** | The atomic unit a language model reads — roughly a word or word-piece. `cl100k_base` is the tokenizer used by GPT/OpenAI models; it handles Arabic adequately with BPE encoding. |
| **Mean-Pooling** | Averaging all token embeddings in a sequence into one fixed-size vector representing the whole sentence. Used in `DocumentEmbedder` to turn variable-length text into a 768-d vector. |
| **L2 Normalization** | Scaling a vector so its length (L2 norm) = 1.0. After this, inner product equals cosine similarity — a mathematical equivalence that lets FAISS's `IndexFlatIP` work as a cosine-similarity index. |
| **LLM** (Large Language Model) | A neural network trained on vast text corpora that can generate fluent text. Here: any API-based or local LLM configured via `LLM_MODEL` and `LLM_API_KEY` environment variables. |
| **Temperature** | A sampling parameter (0.3 here) that controls LLM randomness. Lower = more deterministic and factual; higher = more creative but potentially less accurate. |
| **max_tokens** | The upper limit on how many tokens the LLM may generate in one response (1200 here). Prevents runaway outputs and controls cost. |
| **Hallucination** | When an LLM generates plausible-sounding but factually unsupported text. In a medical context this is dangerous — hence the hallucination-checker agent. |
| **Grounding** | Anchoring the LLM's answer to retrieved source documents so every claim can be traced back to a real Q&A pair. The citation agent performs grounding. |
| **Citation** | A numbered reference (`[1]`, `[2]`, …) pointing back to the source chunk used in the answer, formatted with category and snippet. |
| **MLflow Trace / Span** | MLflow is an ML experiment-tracking platform. A *trace* records one end-to-end pipeline run; a *span* is a named sub-step within it (here: `"arabic_medical_pipeline"` with `span_type="CHAIN"`). |
| **CrewAI Agent** | An autonomous unit with a *role*, *goal*, and *backstory* that directs how a CrewAI-managed LLM behaves during its assigned task. Agents are declaratively defined in `agents.yaml`. |
| **CrewAI Task** | A unit of work assigned to an agent — includes a description, expected output format, and optional `context` (output of prior tasks it can read). Defined in `tasks.yaml`. |
| **CrewAI Tool** | A Python class (subclassing `BaseTool`) that an agent can invoke to perform an action — e.g., search a vector store, call the classifier. Tools have a Pydantic-validated input schema. |
| **Pydantic Schema** | A Python data-validation library. Tool `args_schema` classes enforce that agents pass correctly typed arguments — if not, the call fails with a clear error rather than silently corrupting data. |
| **langdetect** | A Python port of Google's language-detection library. Identifies the language of text in ~50 languages from statistical character-frequency models. |
| **Singleton** | A design pattern where a class has only one instance shared across the whole process. Used for the embedder, FAISS store, BM25 index, and cache — expensive to load, so loaded once and reused. |
| **LRU Cache / TTL** | *Least Recently Used* — evicts oldest entries when the cache is full. *Time-to-Live* — evicts entries after N seconds regardless of use. Both applied to `QueryCache` (1000 entries, 2-hour TTL). |
| **Emergency Triage** | Rule-based detection of life-threatening symptom keywords (chest pain, stroke, etc.) that bypasses the full pipeline and immediately returns a "call emergency services" message in Arabic. |
| **Entity Extraction** | Identifying the medical subject of a query (e.g., "السكري" = diabetes) using regex patterns rather than a full NER model — fast and deterministic. |
| **Intent Classification** | Labelling the *purpose* of a query: is the user asking for `symptoms`, `treatment`, `causes`, `prevention`, `definition`? Drives retrieval strategy and BM25 entity-presence filtering. |
| **E5 Prefix** | The `intfloat/multilingual-e5` model was trained with `"passage: "` prepended to documents and `"query: "` prepended to queries. Omitting these prefixes degrades retrieval quality. |

---

## 3. Repository Layout

```
graduation_final/
│
├── app.py                      # Streamlit web UI (RTL Arabic chat)
├── api_server.py               # FastAPI REST API (port 8000)
├── api_keys.json               # API key store (git-ignored in prod)
├── requirements.txt            # Pip dependencies (installed into venv on EC2)
├── README.md                   # User-facing quickstart
├── TECHNICAL_DOCS.md           # This file
├── mlflow.db                   # Local MLflow SQLite tracking store
├── .env / .env.example         # Runtime secrets and config
├── .gitignore
│
├── data/
│   └── concatenated_df.csv    # 341,476 Arabic medical Q&A pairs
│
├── indexes/                   # Auto-generated by build_index.py
│   ├── faiss_index.bin        # Serialized FAISS IndexFlatIP
│   ├── bm25_index.pkl         # Serialized BM25Okapi object
│   └── metadata.pkl           # List of chunk dicts (text, category, …)
│
├── src/medical_chatbot/
│   ├── __init__.py
│   ├── crew.py                # Core pipeline + CrewAI Crew definition
│   ├── cache.py               # LRU + TTL query cache
│   ├── main.py                # CLI entrypoint
│   │
│   ├── config/
│   │   ├── agents.yaml        # Declarative agent definitions (6 agents)
│   │   └── tasks.yaml         # Declarative task definitions (6 tasks)
│   │
│   ├── rag/
│   │   ├── document_loader.py     # CSV → MedicalDocument objects
│   │   ├── chunking.py            # 500-token chunks, 100-token overlap
│   │   ├── embedding_pipeline.py  # E5 embedder (768-d, CUDA-aware)
│   │   ├── vector_store.py        # FAISS index wrapper
│   │   ├── keyword_index.py       # BM25 index + Arabic tokenizer
│   │   └── build_index.py         # One-shot CLI: ingest → embed → save
│   │
│   └── tools/
│       ├── language_detection_tool.py
│       ├── classifier_tool.py
│       ├── disease_entity_extractor.py
│       ├── hybrid_search_tool.py
│       ├── citation_tool.py
│       └── hallucination_checker_tool.py
│
├── telegram_bot/
│   ├── bot.py                 # Telegram bot (python-telegram-bot)
│   └── requirements.txt
│
├── thesis_book/               # Full 200+ page LaTeX thesis
│   └── book.tex               # compile: pdflatex book.tex (×2)
│
└── .github/workflows/
    └── deploy.yml             # CI: SSH into EC2, run ~/deploy.sh (git pull + restart screen sessions)
```

---

## 4. RAG Layer

The RAG layer converts the raw CSV into searchable indexes. It runs **once** (via `build_index.py`) before the chatbot starts; thereafter indexes are loaded from disk.

### `document_loader.py`

```python
@dataclass
class MedicalDocument:
    doc_id: int
    text: str        # "السؤال: {q}\nالجواب: {a}" — combined for embedding
    question: str    # raw q_body
    answer: str      # raw a_body
    category: str    # Arabic specialty label (default: "عام")
    category_en: str # English specialty label (default: "General")
```

- `load_documents(csv_path, sample=None, min_answer_len=20)` — reads with pandas, drops rows missing `q_body`/`a_body`, fills missing category columns, **filters answers shorter than 20 characters** (removes stub entries), optionally random-samples with `random_state=42` for reproducibility.
- The combined `text` field (`"السؤال: …\nالجواب: …"`) gives the embedding model both question and answer context, improving retrieval for paraphrase queries.

### `chunking.py`

```python
CHUNK_SIZE = 500    # max tokens per chunk
CHUNK_OVERLAP = 100 # tokens shared between adjacent chunks
```

- Uses **tiktoken `cl100k_base`** (OpenAI's BPE tokenizer) — handles Arabic via byte-level fallback.
- `Chunker._chunk_tokens(tokens)` — sliding window: yields `tokens[i : i+CHUNK_SIZE]` stepping by `CHUNK_SIZE - CHUNK_OVERLAP`.
- Documents shorter than `CHUNK_SIZE` are kept as a single chunk (no splitting needed).
- Each `TextChunk` dataclass carries: `chunk_id`, `doc_id`, `text`, `question`, `category_en`, `category` — metadata needed by downstream retrieval and citation tools.
- The 100-token overlap ensures that a query spanning a chunk boundary can still find the relevant context.

### `embedding_pipeline.py`

```python
DEFAULT_MODEL = "intfloat/multilingual-e5-base"  # env: EMBEDDING_MODEL
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
```

- `DocumentEmbedder` loads the model via HuggingFace `AutoTokenizer` + `AutoModel` (raw `transformers`, not `sentence-transformers`).
- **Mean-pooling** over the last hidden state, masked by the attention mask (pads ignored), then **L2-normalized** — making inner product equal cosine similarity.
- **E5 prefix logic**: auto-detects the `intfloat/` model family and prepends `"passage: "` to corpus documents, `"query: "` to user queries. This is required by the E5 training protocol; skipping it causes measurable quality degradation.
- CUDA auto-detected via `torch.cuda.is_available()` — falls back to CPU silently.
- `get_embedder()` — module-level singleton; the 438 MB model is loaded once and reused.
- Public API: `embed_documents(texts) → np.ndarray`, `embed_query(text) → np.ndarray (shape 1×768)`.

### `vector_store.py`

```python
FAISS_INDEX_PATH = "indexes/faiss_index.bin"
METADATA_PATH    = "indexes/metadata.pkl"
TOP_K = 5
```

- `FAISSVectorStore(embedding_dim=768)` wraps `faiss.IndexFlatIP` (exact inner-product search — no approximation, correct for datasets under ~1 M vectors).
- `build(vectors, metadata)` — adds the full matrix at once; `metadata` is a list of dicts (one per chunk), stored separately in a pickle file since FAISS only stores float vectors.
- `search(query_vector, k)` — calls `index.search()`, returns `(metadata_list, scores)` sorted descending. Scores are cosine similarities in [−1, 1].
- `get_vector_store()` — singleton; FAISS index stays in RAM between queries.

### `keyword_index.py`

```python
BM25_INDEX_PATH = "indexes/bm25_index.pkl"
TOP_K = 5
```

- `_tokenize_arabic(text)` — Arabic-specific pipeline:
  1. Strip Unicode diacritics (tashkeel, harakat) — ranges `U+0610-061A`, `U+064B-065F`.
  2. Remove non-Arabic characters (Latin, numbers, punctuation).
  3. Drop tokens from a hardcoded `ARABIC_STOPWORDS` set (question words: هل, ما, كيف; prepositions: في, من, على; demonstratives: هذا, تلك; etc.).
  4. Drop 1-character tokens (usually noise).
- `BM25KeywordIndex.build(metadata)` — tokenizes all chunk texts, constructs `BM25Okapi` object.
- `search(query, k)` — tokenizes query, calls `bm25.get_scores()`, filters scores ≤ 0 (no match), returns top-k sorted descending.
- `get_bm25_index()` — singleton.

### `build_index.py`

CLI: `python src/medical_chatbot/rag/build_index.py [--sample N]`

Five logged pipeline steps:
1. **Load** — `load_documents()` from `data/concatenated_df.csv`.
2. **Chunk** — `chunk_documents()` → list of `TextChunk`.
3. **Metadata** — convert chunks to plain dicts (JSON-serialisable).
4. **Embed** — `embed_documents()` with `passage:` prefix → `float32` matrix.
5. **Index** — build + save FAISS and BM25 from the same metadata.

Environment variables control paths (`DATA_PATH`, `FAISS_INDEX_PATH`, `BM25_INDEX_PATH`, `METADATA_PATH`). `sys.path` manipulation ensures the `src` package is importable without a formal install.

---

## 5. Tools Layer

All tools inherit from `crewai.tools.BaseTool`. Each defines an `args_schema` (Pydantic `BaseModel`) that validates agent inputs. All `_run()` methods return **JSON strings** (`json.dumps(..., ensure_ascii=False)`) so the downstream CrewAI agent receives structured data it can parse.

### `language_detection_tool.py`

```python
class LangDetectionTool(BaseTool):
    def _run(self, query: str) -> str
```

- Calls `langdetect.detect(query)`, maps ISO 639-1 codes via `LANG_MAP` (ar, en, fr, de, es, zh-cn, tr).
- Catches `LangDetectException` (too short / ambiguous) → `"Unknown"`.
- **Always returns `response_language: "Arabic"`** regardless of input language — the system enforces Arabic-only responses.

### `classifier_tool.py`

```python
EMERGENCY_KEYWORDS = ["ألم حاد في الصدر", "chest pain", "سكتة دماغية", "stroke", ...]  # bilingual
CATEGORY_KEYWORDS  = {specialty: [ar_kw, en_kw, ...], ...}  # 21 specialties
```

- `_normalize(text)` — strips diacritics + lowercases for case-insensitive matching.
- `_run(query)` — emergency check runs first (short-circuit): if any emergency keyword found in normalized query → returns `EMERGENCY_RESPONSE` (pre-written Arabic alert), sets `is_emergency=True`.
- Otherwise: scores each of 21 specialties by counting substring matches; category with highest score wins. Tie-breaks to `general_medicine`.
- No ML model — pure substring matching. Fast and deterministic; avoids LLM overhead for a classification step that needs 100% reliability on safety-critical emergency routing.

### `disease_entity_extractor.py`

```python
INTENT_PATTERNS = [
    (r"تاريخ مرضي.*?عن\s+(.+)", "clinical_history", True),
    (r"أعاني من\s+(.+)",          "symptom_description", True),
    (r"(?:أعراض|symptoms)\s+(.+)", "symptoms", False),
    (r"علاج\s+(.+)",               "treatment", False),
    ...  # 11 intent patterns total
]
```

- Three query archetypes the patterns handle:
  1. **Disease-focused**: "ما هي أعراض السكري؟" → entity = "السكري", intent = "symptoms"
  2. **Patient self-description**: "أعاني من ألم في الصدر" → entity = "ألم في الصدر", intent = "symptom_description"
  3. **Clinical history**: "تاريخ مرضي يعاني من ضغط الدم" → full phrase preserved, intent = "clinical_history"
- `keep_full_phrase=True` for symptom/clinical intents — preserves multi-word complaints as the entity.
- `keep_full_phrase=False` — strips Arabic stopwords from the extracted group, leaving the core medical term.
- Fallback (no pattern matches): entire query becomes the entity, intent = `"general"`.
- `extract_disease_entity(query) → dict` with keys: `disease_entity`, `query_intent`, `full_query`, `extraction_method` (`"pattern"` or `"fallback"`).

### `hybrid_search_tool.py`

```python
TOP_K              = 5     # final chunks returned
VECTOR_THRESHOLD   = 0.50  # minimum cosine score to trust vector results
ENTITY_BOOST       = 1.3   # score multiplier for entity-matched chunks
RELEVANCE_THRESHOLD= 0.80  # high-confidence threshold
MAX_CHUNK_CHARS    = 600   # snippet truncation
```

Retrieval flow in `_run(query)`:

1. **Entity extraction** — calls `extract_disease_entity(query)`.
2. **Primary vector search** — embed query with `query:` prefix, search FAISS (`k = max(15, 10)`).
3. **Entity-boosted search** (if entity found) — embed the entity string separately, search again with a lowered threshold (`0.7 × VECTOR_THRESHOLD`), multiply matching scores by `ENTITY_BOOST`.
4. **BM25 fallback** (if `top_score < VECTOR_THRESHOLD` OR `len(results) < TOP_K`) — tokenize query, run BM25, normalize scores as `min(score / 20, 1.0)`.
5. **BM25 entity filter** — for non-symptom intents, require the entity text to appear in the chunk (checks content words ≥5 chars, also tries stripping Arabic definite article `"ال"` prefix). This avoids pulling in tangentially related documents.
6. Merge, deduplicate by chunk ID, sort by score, truncate to `TOP_K`.
7. Return JSON with: `disease_entity`, `query_intent`, `used_bm25_fallback`, `top_similarity`, `chunks[]`.

RAG singleton components loaded lazily inside `_run` — avoids importing them at module load time (which would fail if indexes don't exist yet).

### `citation_tool.py`

```python
class CitationGroundingTool(BaseTool):
    def _run(self, retrieved_json: str) -> str
```

- Parses the `chunks` array from `hybrid_search_tool`'s JSON output.
- Produces per-chunk: `{number, source (category_en), snippet (≤400 chars), label (category Arabic)}`.
- Builds a `context_block` — the full text of all chunks joined with `[rank] (category_en)` headers. This is what gets injected into the LLM prompt.
- Builds `arabic_citation_list` — e.g., `"1. Cardiology\n2. Dermatology"`.
- Edge cases: empty `chunks` → `"لا توجد مصادر مسترجعة."` (no sources found); JSON parse error → error payload.

### `hallucination_checker_tool.py`

```python
MIN_COVERAGE_THRESHOLD = 0.15  # 15% of content words must appear in context
MIN_SENTENCE_LENGTH    = 5     # words — shorter sentences skipped
```

- `_clean(text)` — strips Arabic diacritics (incl. `U+0670` superscript alif) and non-Arabic punctuation, lowercases.
- `_split_sentences(text)` — regex split on `[.!?؟،\n]`, keeps only sentences > 10 characters.
- For each sentence in the generated answer:
  1. Skip **disclaimer sentences** containing `"تعليمية"`, `"استشارة"`, or `"طبيب"` — these are template text, not factual claims.
  2. Compute content-word set from the retrieved context (words > 3 chars).
  3. Compute *coverage* = fraction of answer sentence's content words found in the context word set.
  4. If `coverage < 0.15` and sentence has ≥ 5 content words → flag as unsupported.
- Returns: `hallucination_detected` (bool), `flagged_count`, up to 3 `unsupported_claims`, Arabic recommendation string.
- Pure stdlib — no external model. Fast, language-agnostic (works on Arabic because it ignores short function words).

---

## 6. Orchestration

### `crew.py`

The central file — both the CrewAI crew definition and the procedural pipeline runner.

#### CrewAI Crew (declarative, `@Crew` + `@agent`/`@task` decorators)

```python
@CrewBase
class arabic_chatbot:
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent def language_detection_agent(self) -> Agent: ...
    @agent def medical_classification_agent(self) -> Agent: ...
    @agent def hybrid_retrieval_agent(self) -> Agent: ...
    @agent def citation_grounding_agent(self) -> Agent: ...
    @agent def arabic_medical_response_agent(self) -> Agent: ...
    @agent def hallucination_detection_agent(self) -> Agent: ...

    @crew def crew(self) -> Crew:
        return Crew(..., process=Process.sequential, verbose=True, tracing=True)
```

- **6 agents** loaded from `agents.yaml` — each has a `role`, `goal`, and `backstory` that guides the LLM's persona for that step.
- `Process.sequential` — agents run one after another; each task can reference earlier task outputs via `context:`.
- `tracing=True` — CrewAI emits trace events compatible with LangSmith/MLflow.

#### Procedural Pipeline (`run()` method)

The `run()` method bypasses sequential Crew execution for speed and fine-grained control:

```
intent_classifier → language_detect → classifier (emergency?) →
entity_extractor → cache_lookup → retrieval → citation_grounding →
context_evaluator (LLM judge: relevance score 1-10) →
[if score < 6: Serper internet search] →
LLM generate (Qwen2.5, temperature=0.3, max_tokens=1200) →
hallucination_check → cache_store
```

Key decisions encoded in `run()`:
- **Emergency bypass**: `is_emergency` flag set by classifier → returns `emergency_response` string immediately, no LLM call.
- **Greeting / non-medical bypass**: `intent in {"تحية", "غير طبي"}` → canned response strings without retrieval.
- **Context truncation**: `context_block` limited to 2500 chars to stay within LLM context window.
- **Mode switching**: `mode` param (`rag|bm25|hybrid|all|internet`) sets `VECTOR_THRESHOLD` env var before calling the hybrid tool — `1.0` forces BM25 only, `0.0` forces vector only.
- **`on_progress` callback**: called at each pipeline step with a status string, enabling live UI updates (Streamlit `st.status`, Telegram message edits).

#### MLflow Tracing

```python
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)       # SQLite: mlflow.db
mlflow.set_experiment("arabic_medical_chatbot")
mlflow.litellm.autolog(log_traces=True)             # auto-captures LLM calls
with mlflow.start_span("arabic_medical_pipeline", span_type="CHAIN"):
    mlflow.log_params({"emergency": ..., "intent": ..., "cache_hit": ...})
    mlflow.log_metrics({"elapsed_time": ..., "results_count": ..., ...})
```

Enabled only when `MLFLOW_ENABLED=true`. Every pipeline run logs params (query metadata), metrics (timing, result counts), and a hierarchical trace of sub-spans.

#### LLM Configuration (`_build_llm()`)

```python
# Generic LLM configuration via environment variables
model = LLM_MODEL      # e.g., "groq/llama-3.3-70b-versatile", "openai/gpt-4", etc.
api_key = LLM_API_KEY  # API key for the chosen provider
```

The LLM is accessed through LiteLLM's unified interface — `crew.py` supports any provider that LiteLLM supports (OpenAI, Groq, Anthropic, Ollama, OpenRouter, etc.).

### `cache.py`

```python
class QueryCache:
    def __init__(self, max_size=1000, ttl=7200): ...
    def get(self, mode, intent, query) -> Optional[str]: ...
    def put(self, mode, intent, query, answer): ...
```

- Cache key: `f"{mode}:{intent}:{_normalize(query)}"` — `_normalize` strips Arabic diacritics so "السُكَّر" and "السكر" hit the same cache entry.
- LRU eviction: when `len(cache) >= max_size`, the oldest-inserted key is deleted.
- TTL eviction: each entry stored with `time.time()` timestamp; `get()` checks age and returns `None` if stale.
- Thread-safe via `threading.Lock` — safe for concurrent requests in the FastAPI async context.
- Env vars: `CACHE_ENABLED` (default `true`), `CACHE_MAX_SIZE` (1000), `CACHE_TTL_SECS` (7200).
- `get_cache()` — module-level singleton.

### `main.py`

```python
def run_query(query: str) -> str:
    bot = arabic_chatbot()
    return bot.run(query)

def cli():
    # --query / -q for single-shot
    # else: REPL loop with Arabic prompt
```

Minimal CLI wrapper — primarily for local testing and batch evaluation.

### `config/agents.yaml` and `config/tasks.yaml`

Declarative YAML consumed by the `@CrewBase` decorator machinery.

**agents.yaml** defines for each of the 6 agents:
- `role` — short title (e.g., `"Arabic Medical Response Specialist"`)
- `goal` — what the agent optimizes for
- `backstory` — persona/context paragraph that shapes LLM behavior during the task
- `llm` — injected at runtime from `_build_llm()`
- `tools` — list of `BaseTool` instances passed to each agent

**tasks.yaml** defines for each of the 6 tasks:
- `description` — detailed prompt including expected JSON output schema
- `expected_output` — target output format description
- `context` — list of task names whose outputs are accessible as context
- `agent` — which agent executes this task

The mandatory **medical disclaimer** string (`⚕️ تنبيه: هذه المعلومات لأغراض تعليمية فقط...`) is embedded in the response task description, ensuring the LLM always appends it.

---

## 7. Interfaces

### `app.py` — Streamlit Web UI

```python
@st.cache_resource
def load_chatbot_v3() -> arabic_chatbot: ...
```

- `st.cache_resource` — Streamlit's singleton decorator; the chatbot object (with loaded FAISS + BM25) is created once per server process and shared across browser sessions.
- **RTL layout** — custom CSS (`direction: rtl; text-align: right`) applied to chat messages.
- **Mode selector** — sidebar radio: `rag | bm25 | hybrid | all | internet`.
- **Live progress** — `st.status()` context manager updated via `_on_progress` callback passed into `bot.run()`. Users see step-by-step status (`Extracting entity...`, `Searching FAISS...`, etc.) while the pipeline runs.
- **Emergency bubble** — if response metadata contains `is_emergency=True`, message box styled with red background and bold border.
- **References box** — `web_sources` from metadata (populated when Serper internet search runs) displayed as a collapsible section.
- `_extract_final_answer(raw)` — handles both plain-string and JSON-wrapped Crew outputs; strips inner `"final_answer"` key if present.

### `api_server.py` — FastAPI REST API

```python
app = FastAPI(title="ArabMedRAG API", version="1.0.0")
```

**Authentication**: `X-API-Key` header validated against keys loaded from `API_KEYS` env variable (comma-separated) or `api_keys.json`. Public endpoints skip auth.

**Endpoints**:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/health` | No | Liveness probe; returns `{status, model, index_loaded}` |
| `GET` | `/api/v1/categories` | Yes | Returns `CATEGORY_ARABIC_LABELS` dict (21 specialties) |
| `POST` | `/api/v1/query` | Yes | Main chat endpoint |

`POST /api/v1/query` request body:
```json
{
  "query": "ما هي أعراض ارتفاع ضغط الدم؟",
  "mode": "hybrid",
  "history": []
}
```

Response:
```json
{
  "answer": "...",
  "meta": { "intent": "symptoms", "elapsed_time": 4.2, ... },
  "disclaimer": "⚕️ تنبيه: هذه المعلومات لأغراض تعليمية فقط..."
}
```

- Mode validated against `{"rag", "bm25", "internet", "hybrid", "all"}` — 422 if invalid.
- Lifespan startup handler (`@asynccontextmanager`) pre-instantiates `arabic_chatbot()` so the first request doesn't pay the index-loading cost.
- CORS configured via `CORS_ORIGINS` env (default: `["*"]`).
- Server: `uvicorn api_server:app --host 0.0.0.0 --port 8000`.

### `telegram_bot/bot.py`

```python
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
```

**Commands**:

| Command | Behaviour |
|---------|-----------|
| `/start` | Welcome message in Arabic |
| `/help` | Shows current mode, cache stats, available commands |
| `/mode` | Sends inline keyboard with 5 mode buttons (`rag`, `bm25`, `hybrid`, `all`, `internet`) |
| `/clear` | Clears per-user conversation history + flushes global query cache |

**Message handling**:
- Free-text messages routed to `message_handler()`.
- Per-user state: `user_state[uid] = {"mode": "hybrid", "history": [...]}`. History capped at **12 messages** (6 turns) to stay within LLM context limits.
- `asyncio.to_thread(crew.run, ...)` — runs the blocking synchronous pipeline in a thread pool without blocking the async Telegram event loop.
- Live progress: initial "processing…" message sent immediately, then edited at each `on_progress` callback to show current step.
- Three-message response pattern:
  1. Main answer body.
  2. References / web sources (if any).
  3. Footer: `mode | elapsed | FAISS used? | BM25 used? | cache hit?`

---

## 8. Infrastructure

### `.github/workflows/deploy.yml`

Triggers on push to `main`/`master` or manual dispatch. Completes in under 30 seconds.

Steps:
1. `appleboy/ssh-action@v1` — SSHes into EC2 using `EC2_HOST` and `EC2_SSH_KEY` secrets.
2. Runs `~/deploy.sh` on the EC2 host.

`~/deploy.sh` does:
```bash
cd ~/graduation_final_git
git pull                          # pull latest code from GitHub
pkill -f api_server.py || true    # stop old processes
pkill -f bot.py || true
screen -S api -X quit || true
screen -S bot -X quit || true
screen -dm -S api bash -c "source venv/bin/activate && python api_server.py"
screen -dm -S bot bash -c "source venv/bin/activate && python telegram_bot/bot.py"
```

Large files (FAISS/BM25 indexes, `.env`, `api_keys.json`) are provisioned once via `scp` and live permanently on the EC2 root volume — they are never committed to git or transferred by the CI pipeline.

---

## 9. End-to-End Request Trace

Walk-through for query: `"ما هي أعراض ارتفاع ضغط الدم؟"` (What are the symptoms of high blood pressure?)

| Step | File | What Happens |
|------|------|--------------|
| 1 | `crew.py:run()` | Query received; `on_progress("Detecting language...")` fired. |
| 2 | `tools/language_detection_tool.py` | `langdetect.detect()` → `"ar"`. `response_language` = `"Arabic"`. |
| 3 | `tools/classifier_tool.py` | No emergency keywords found. `"ضغط الدم"` matches cardiology keywords → `category = "cardiology"`. |
| 4 | `tools/disease_entity_extractor.py` | Pattern `r"أعراض\s+(.+)"` matches → `disease_entity = "ارتفاع ضغط الدم"`, `query_intent = "symptoms"`. |
| 5 | `cache.py:QueryCache.get()` | Key = `"hybrid:symptoms:ما هي اعراض ارتفاع ضغط الدم"` (normalized). Cache miss (first query). |
| 6 | `rag/embedding_pipeline.py` | `embed_query("query: ما هي أعراض ارتفاع ضغط الدم؟")` → 768-d vector (normalized). |
| 7 | `rag/vector_store.py` | FAISS `index.search(vector, k=15)` → 15 cosine scores. Top score = 0.82 (> threshold 0.50). |
| 8 | `tools/hybrid_search_tool.py` | Entity-boosted re-search on `"ارتفاع ضغط الدم"` — relevant chunks multiplied by 1.3. BM25 fallback skipped (top score already > 0.50). Final top-5 chunks returned. |
| 9 | `tools/citation_tool.py` | Chunks formatted into `context_block` (≤2500 chars) and numbered citation list. |
| 10 | `crew.py:run()` | LLM context-evaluator prompt sent to Qwen → relevance score = 8/10 (≥ 6, no internet search needed). |
| 11 | `crew.py:run()` | Final prompt constructed: system role + context_block + user query. Sent to Qwen2.5 (`temperature=0.3`, `max_tokens=1200`). |
| 12 | `tools/hallucination_checker_tool.py` | Generated answer split into sentences; each checked for ≥15% content-word overlap with context. 0 flagged sentences. `hallucination_detected = False`. |
| 13 | `cache.py:QueryCache.put()` | Answer stored in cache for 2 hours. |
| 14 | Caller (app/api/bot) | JSON response with `answer`, `meta`, `disclaimer` returned. |

---

## 10. Key Design Decisions

- **E5 over MiniLM / AraBERT**: Evaluation showed multilingual-E5-base achieves cosine similarity scores 0.80+ on Arabic medical queries vs 0.55-0.72 for AraBERT. E5's multilingual training corpus includes Arabic medical text.
- **Cosine via L2-normalized `IndexFlatIP`**: FAISS's `IndexFlatL2` computes Euclidean distance (smaller = closer), which is less intuitive. Normalizing vectors and using `IndexFlatIP` (inner product) produces cosine similarity directly — scores in [−1, 1] with 1.0 = perfect match, easily thresholded.
- **Hybrid retrieval**: Dense search catches semantic paraphrases ("أعراض" vs "علامات"); sparse BM25 catches exact medical term matches (drug names, rare disease names not in training vocab). Using both improves recall.
- **Emergency bypass before LLM**: Life-threatening queries must return an alert within milliseconds — triggering the full RAG pipeline adds 3–10 seconds of latency. Rule-based detection is deterministic and O(n) string scan.
- **JSON-string tool returns**: CrewAI agents receive tool outputs as strings (the agent prompt includes the tool result as text). Returning JSON strings allows the next agent to `json.loads()` the result while remaining compatible with CrewAI's string-based inter-agent communication.
- **Singletons for embedder, FAISS, BM25**: Loading the E5 model (~438 MB) and FAISS index at module import time would make every test import slow. Singletons load on first use and persist — loading cost paid once per process.
- **`cl100k_base` tokenizer for Arabic chunking**: Arabic BPE tokenization splits unknown Arabic sub-words into UTF-8 byte sequences. While imperfect linguistically, it is consistent and predictable — the 500-token window reliably produces chunks of manageable size for the embedding model's 512-token max sequence length.

---

## 11. Verification

### Run a single query end-to-end (CLI)

```bash
conda activate arabic_chatbot
python src/medical_chatbot/main.py --query "ما هي أعراض ارتفاع ضغط الدم؟"
```

Expected: Arabic answer with numbered citations and disclaimer.

### Check tool imports without indexes

```bash
python -c "
from src.medical_chatbot.tools.disease_entity_extractor import extract_disease_entity
print(extract_disease_entity('ما هي أعراض السكري؟'))
"
```

Expected: `{'disease_entity': 'السكري', 'query_intent': 'symptoms', ...}`

### Start Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Verify RTL layout, mode selector, live progress bar on first query.

### Start FastAPI and probe health

```bash
uvicorn api_server:app --port 8000
curl http://localhost:8000/api/v1/health
```

Expected: `{"status": "ok", "model": "...", "index_loaded": true}`

### Check MLflow runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` — verify pipeline runs appear with params, metrics, and spans when `MLFLOW_ENABLED=true`.

### Compile thesis

```bash
cd thesis_book
pdflatex book.tex && pdflatex book.tex
```

Produces `thesis_book/book.pdf` (200+ pages, 13 chapters across 4 parts).

---

*Document generated for ArabMedRAG — graduation project, Faculty of Computer Science.*  
*Supervisors: Prof. Reham Anany, Prof. Doaa Abo-Elhassan, Prof. Eman Mostafa.*
