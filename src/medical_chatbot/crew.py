"""
crew.py
────────────────────────────────────────────────
CrewAI crew definition for the Arabic Medical Chatbot.
Crew name: arabic_chatbot

Assembles 6 agents in a sequential process:
  1. Language Detection Agent
  2. Medical Classification Agent
  3. Hybrid Retrieval Agent
  4. Citation Grounding Agent
  5. Arabic Medical Response Agent
  6. Hallucination Detection Agent
"""

import logging
import os
import pathlib
import time
import json

import mlflow

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from src.medical_chatbot.tools.language_detection_tool import LanguageDetectionTool
from src.medical_chatbot.tools.classifier_tool import MedicalClassifierTool
from src.medical_chatbot.tools.hybrid_search_tool import HybridSearchTool
from src.medical_chatbot.tools.citation_tool import CitationGroundingTool
from src.medical_chatbot.tools.hallucination_checker_tool import (
    HallucinationCheckerTool,
)
from src.medical_chatbot.tools.disease_entity_extractor import (
    extract_disease_entity,
    DiseaseEntityExtractorTool,
)

logger = logging.getLogger(__name__)

# Fix Windows charmap errors from CrewAI EventsBus emoji logging
import sys, io

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")


# CJK Character Filter — strips Chinese/Japanese/Korean characters from responses
def _clean_arabic_response(text: str) -> str:
    """Remove CJK characters and normalize whitespace in LLM response."""
    import re as _re
    # Remove Chinese characters
    text = _re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f]', '', text)
    # Remove Japanese Hiragana/Katakana
    text = _re.sub(r'[\u3040-\u309f\u30a0-\u30ff]', '', text)
    # Remove Korean
    text = _re.sub(r'[\uac00-\ud7af]', '', text)
    # Normalize horizontal whitespace only — preserve newlines
    text = _re.sub(r'[^\S\n]+', ' ', text)
    # Collapse 3+ consecutive newlines to 2
    text = _re.sub(r'\n{3,}', '\n\n', text)
    # Ensure bullet points each get their own line
    text = _re.sub(r'\s*([•\-])\s+', r'\n\1 ', text)
    return text.strip()

# Always resolve .env relative to project root (two levels above this file)
_ENV_PATH = pathlib.Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH), override=True)


def _setup_mlflow():
    """Initialize MLflow tracking + GenAI tracing if configured."""
    mlflow_enabled = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
    if not mlflow_enabled:
        return None

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "arabic_medical_chatbot")
    mlflow.set_experiment(experiment_name)

    # Enable litellm autologging so every litellm.completion() call is traced
    try:
        from mlflow import litellm as _mlflow_litellm
        _mlflow_litellm.autolog(log_traces=True)
    except Exception as _e:
        logger.debug(f"mlflow.litellm.autolog unavailable: {_e}")

    if mlflow.active_run():
        mlflow.end_run()

    return mlflow.start_run()


# Description: Initializes our main LLM client (like an OpenRouter/Llama endpoint) based on the settings in our `.env` file.
def _build_llm() -> LLM:
    """Build LLM client from environment config."""
    # Re-read .env each time in case of late env changes
    load_dotenv(dotenv_path=str(_ENV_PATH), override=True)
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    logger.info(f"[crew] Building LLM — provider='{provider}'")
    print(f"[crew] LLM_PROVIDER={provider!r}")

    if provider == "openrouter":
        # Use the openrouter/ prefix — LiteLLM routes natively, no base_url needed.
        # Providing base_url alongside the openrouter/ prefix causes malformed requests.
        model = os.getenv("OPENROUTER_MODEL", "openrouter/qwen/qwen-2.5-72b-instruct")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        print(f"[crew] OpenRouter model: {model}")
        return LLM(
            model=model,
            api_key=api_key,
        )

    # Ollama fallback
    model = os.getenv("OLLAMA_MODEL", "ollama/qwen2.5")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"[crew] Ollama model: {model} @ {base_url}")
    return LLM(model=model, base_url=base_url)


# Description: This is the heart of our multi-agent framework. We define our specialized AI crew here, linking each agent to its required tasks.
@CrewBase
class arabic_chatbot:
    """
    Arabic Medical Chatbot Crew

    Multi-agent system for answering medical questions in Arabic
    using hybrid RAG (FAISS + BM25) and SentenceTransformers embeddings.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.llm = _build_llm()

    # ── Tools (instantiated once) ──────────────────────

    def _lang_tool(self) -> LanguageDetectionTool:
        return LanguageDetectionTool()

    def _cls_tool(self) -> MedicalClassifierTool:
        return MedicalClassifierTool()

    def _search_tool(self) -> HybridSearchTool:
        return HybridSearchTool()

    def _citation_tool(self) -> CitationGroundingTool:
        return CitationGroundingTool()

    def _hallucination_tool(self) -> HallucinationCheckerTool:
        return HallucinationCheckerTool()

    # ── Agents ─────────────────────────────────────────

    @agent
    def language_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["language_detection_agent"],
            tools=[self._lang_tool()],
            llm=self.llm,
            verbose=True,
            max_iter=2,
        )

    @agent
    def medical_classification_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["medical_classification_agent"],
            tools=[self._cls_tool()],
            llm=self.llm,
            verbose=True,
            max_iter=2,
        )

    @agent
    def hybrid_retrieval_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["hybrid_retrieval_agent"],
            tools=[self._search_tool()],
            llm=self.llm,
            verbose=True,
            max_iter=3,
        )

    @agent
    def citation_grounding_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["citation_grounding_agent"],
            tools=[self._citation_tool()],
            llm=self.llm,
            verbose=True,
            max_iter=2,
        )

    @agent
    def arabic_medical_response_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["arabic_medical_response_agent"],
            tools=[],  # Pure generation from context – no tools needed
            llm=self.llm,
            verbose=True,
            max_iter=3,
        )

    @agent
    def hallucination_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["hallucination_detection_agent"],
            tools=[self._hallucination_tool()],
            llm=self.llm,
            verbose=True,
            max_iter=2,
        )

    # ── Tasks ──────────────────────────────────────────

    @task
    def language_detection_task(self) -> Task:
        return Task(config=self.tasks_config["language_detection_task"])

    @task
    def medical_classification_task(self) -> Task:
        return Task(config=self.tasks_config["medical_classification_task"])

    @task
    def hybrid_retrieval_task(self) -> Task:
        return Task(config=self.tasks_config["hybrid_retrieval_task"])

    @task
    def citation_grounding_task(self) -> Task:
        return Task(config=self.tasks_config["citation_grounding_task"])

    @task
    def arabic_medical_response_task(self) -> Task:
        return Task(config=self.tasks_config["arabic_medical_response_task"])

    @task
    def hallucination_detection_task(self) -> Task:
        return Task(config=self.tasks_config["hallucination_detection_task"])

    # ── Crew ───────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # auto-populated by @agent decorators
            tasks=self.tasks,  # auto-populated by @task decorators
            process=Process.sequential,
            verbose=True,
            tracing=True,
        )

    # ── Convenience method ─────────────────────────────

    # Description: The main trigger pipe! It fires up the multi-agent task sequence and directly streams responses back to the user interface.
    def run(
        self, query: str, history: list = None, mode: str = "hybrid", on_progress=None
    ) -> str:
        """
        Direct pipeline with retrieval mode and live progress support.
        mode: 'rag' | 'bm25' | 'internet' | 'hybrid' | 'all'
        on_progress: optional callable(step, label, detail) for live UI updates
        """
        mlflow_run = _setup_mlflow()
        t0 = time.time()

        # Start a GenAI root trace span for the full pipeline
        _root_span = None
        try:
            _root_span = mlflow.start_span(
                name="arabic_medical_pipeline",
                span_type="CHAIN",
                inputs={"query": query, "mode": mode},
            )
        except Exception:
            pass

        def _progress(step, label, detail=""):
            if on_progress:
                on_progress(step, label, detail)

        import json
        import litellm

        try:
            mlflow.log_param("query", query[:100])
            mlflow.log_param("mode", mode)
            mlflow.log_param("history_length", len(history) if history else 0)
        except Exception:
            pass

        from src.medical_chatbot.tools.language_detection_tool import (
            LanguageDetectionTool,
        )
        from src.medical_chatbot.tools.classifier_tool import MedicalClassifierTool
        from src.medical_chatbot.tools.hybrid_search_tool import HybridSearchTool
        from src.medical_chatbot.tools.citation_tool import CitationGroundingTool
        from src.medical_chatbot.tools.hallucination_checker_tool import (
            HallucinationCheckerTool,
        )

        # Prepare context-aware search query and history block
        search_query = query
        history_block = ""
        if history:
            last_user_msg = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), None
            )
            if last_user_msg:
                # Prepend the previous question so RAG and Serper understand follow-up pronouns
                search_query = f"{last_user_msg} {query}"

            lines = []
            for m in history[
                -4:
            ]:  # Ensure we only use the last 4 messages to save tokens
                role_ar = "المستخدم" if m["role"] == "user" else "المساعد الطبي"
                lines.append(f"{role_ar}: {m['content']}")
            history_block = "سياق المحادثة السابقة:\n" + "\n".join(lines) + "\n\n"

        # ── 0. LLM Configuration ──────────────────────────────────────────────
        import os
        from dotenv import load_dotenv

        load_dotenv(override=True)
        provider = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()
        if provider == "openrouter":
            model = os.getenv(
                "OPENROUTER_MODEL", "openrouter/qwen/qwen-2.5-72b-instruct"
            )
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            kwargs = {"model": model, "api_key": api_key}
        else:
            model = os.getenv("OLLAMA_MODEL", "ollama/qwen2.5")
            kwargs = {
                "model": model,
                "api_base": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            }

        # ── 0.5. Medical Intent Classification ────────────────────────────────
        _progress("intent", "🤔 تحليل القصد", "التحقق من المحتوى الطبي...")
        intent_prompt = f"هل النص التالي سؤال أو استفسار له علاقة بالطب أو الصحة أو الأمراض أو مجرد أعراض، أم أنه نص غير طبي ورسالة تحية (مثل مرحباً، صباح الخير) أو نص لا علاقة له بالطب؟ أجب بكلمة 'طبي' أو 'تحية' أو 'غير طبي' فقط.\nالنص: {query}"
        try:
            intent_resp = litellm.completion(
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=10,
                temperature=0.0,
                **kwargs,
            )
            intent_ans = (intent_resp.choices[0].message.content or "").strip()
            if "تحية" in intent_ans:
                _progress("intent", "👋 تحية", "توليد الرد للتحية...")
                greet_resp = litellm.completion(
                    messages=[
                        {
                            "role": "user",
                            "content": f"رد على هذه التحية أو الرسالة الودية بلغة المرسل وبشكل مهذب ومختصر كطبيب، بدون إضافة معلومات طبية:\n{query}",
                        }
                    ],
                    max_tokens=100,
                    temperature=0.5,
                    **kwargs,
                )
                result = (greet_resp.choices[0].message.content or "").strip()
                try:
                    mlflow.log_metric("elapsed_time", time.time() - t0)
                    mlflow.set_tag("intent", "greeting")
                    mlflow.end_run()
                except Exception:
                    pass
                try:
                    if _root_span is not None:
                        _root_span.set_outputs({"intent": "greeting"})
                        _root_span.end()
                except Exception:
                    pass
                return result
            elif "غير طبي" in intent_ans:
                _progress("intent", "❌ غير طبي", "تم رفض الطلب")
                result = "عذراً، أنا أعمل كمساعد طبي استشاري ولا يمكنني الإجابة إلا على الأسئلة والاستفسارات المتعلقة بالطب والصحة."
                try:
                    mlflow.log_metric("elapsed_time", time.time() - t0)
                    mlflow.set_tag("intent", "non_medical")
                    mlflow.end_run()
                except Exception:
                    pass
                try:
                    if _root_span is not None:
                        _root_span.set_outputs({"intent": "non_medical"})
                        _root_span.end()
                except Exception:
                    pass
                return result
        except Exception as e:
            print(f"[Pipeline] Intent check failed: {e}")
            pass

        # ── 1. Language detection ────────────────────────────────────────────
        _progress("language", "🌐 اكتشاف اللغة", "تحليل لغة السؤال...")
        lang_json = LanguageDetectionTool()._run(query)
        try:
            lang = json.loads(lang_json)
            detected = lang.get("detected_language", "ar")
            print(f"[Pipeline] language={detected}")
        except Exception:
            print("[Pipeline] language=unknown")

        # ── 2. Classification + emergency fast-path ──────────────────────────
        # We run the logic but hold the _progress update until the end.
        cls_json = MedicalClassifierTool()._run(query)
        try:
            cls = json.loads(cls_json)
        except Exception:
            cls = {}
        cat_display = cls.get("category_arabic") or cls.get("category", "عام")
        print(
            f"[Pipeline] category={cls.get('category')}, emergency={cls.get('is_emergency')}"
        )

        # ── 2.5. Disease Entity Extraction ────────────────────────────────────
        _progress("entity", "🔬 استخراج الكيان المرضي", "تحديد اسم المرض من السؤال...")
        entity_info = extract_disease_entity(query)
        disease_entity = entity_info.get("disease_entity", "")
        query_intent = entity_info.get("query_intent", "general")
        print(f"[Pipeline] entity='{disease_entity}', intent='{query_intent}'")
        _progress("entity", "✅ تم استخراج الكيان", f"المرض: {disease_entity} | القصد: {query_intent}")
        try:
            mlflow.set_tag("disease_entity", disease_entity[:100] if disease_entity else "none")
            mlflow.set_tag("query_intent", query_intent)
            mlflow.set_tag("extraction_method", entity_info.get("extraction_method", "unknown"))
        except Exception:
            pass

        if cls.get("is_emergency"):
            _progress("emergency", "🚨 حالة طارئة", "يُنصح بطلب المساعدة فوراً")
            result = cls.get(
                "emergency_response",
                "⚠️ قد تشير هذه الأعراض إلى حالة طبية طارئة. يرجى طلب المساعدة الطبية فورًا أو الاتصال بخدمات الطوارئ.",
            )
            try:
                mlflow.log_metric("elapsed_time", time.time() - t0)
                mlflow.set_tag("emergency", "true")
                mlflow.set_tag("category", cat_display)
                mlflow.end_run()
            except Exception:
                pass
            try:
                if _root_span is not None:
                    _root_span.set_outputs({"intent": "emergency"})
                    _root_span.end()
            except Exception:
                pass
            return result

        # ── 3. Retrieval (mode-aware) ──────────────────────────────────────────────
        results_count, bm25_used = 0, False
        search_json = None

        if mode in ("rag", "bm25", "hybrid", "all"):
            if mode == "bm25":
                _progress("retrieve", "🔑 BM25 بحث نصي", "تشغيل محرك BM25...")
                import src.medical_chatbot.tools.hybrid_search_tool as _hs_mod

                _orig_thresh = _hs_mod.VECTOR_THRESHOLD
                _hs_mod.VECTOR_THRESHOLD = 1.0
                search_json = HybridSearchTool()._run(search_query)
                _hs_mod.VECTOR_THRESHOLD = _orig_thresh
            elif mode == "rag":
                _progress(
                    "retrieve", "🧠 FAISS سحب دلالي", "بحث FAISS عن الدلالات القريبة..."
                )
                import src.medical_chatbot.tools.hybrid_search_tool as _hs_mod

                _orig_thresh = _hs_mod.VECTOR_THRESHOLD
                _hs_mod.VECTOR_THRESHOLD = 0.0
                search_json = HybridSearchTool()._run(search_query)
                _hs_mod.VECTOR_THRESHOLD = _orig_thresh
            else:
                _progress("retrieve", "⚡ سحب هجين", "FAISS ثم BM25 إن لزم...")
                search_json = HybridSearchTool()._run(search_query)

            try:
                search_data = json.loads(search_json)
                results_count = search_data.get("total_results", 5)
                bm25_used = search_data.get("used_bm25_fallback", False)
            except Exception:
                results_count, bm25_used = 5, False
            bm25_label = "نعم" if bm25_used else "لا"
            _progress(
                "retrieve",
                "✅ اكتمل الاسترجاع",
                f"{results_count} نتيجة • BM25: {bm25_label}",
            )
            print(
                f"[Pipeline] retrieval done — mode={mode} results={results_count}, bm25={bm25_used}"
            )
        else:
            _progress("retrieve", "⏭️ تخطي قاعدة البيانات", "وضع الإنترنت بدلاً من RAG")
            print(f"[Pipeline] skipping RAG/BM25 — mode={mode}")
            search_json = json.dumps(
                {"chunks": [], "total_results": 0, "used_bm25_fallback": False}
            )

        # ── 4. Citation formatting ───────────────────────────────────────────
        cite_json = CitationGroundingTool()._run(search_json)
        try:
            cite = json.loads(cite_json)
        except Exception:
            cite = {}
        context = cite.get("context", "")
        # Hard truncate to keep the LLM prompt under ~2000 tokens
        if len(context) > 2500:
            context = context[:2500] + "\n[...]"
        print(f"[Pipeline] context_len={len(context)}")

        # ── 4a. Context Evaluator (LLM Judge) ────────────────────────────────
        force_internet = False
        if context.strip():
            _progress(
                "evaluator", "⚖️ تقييم النتائج", "فحص مدى ارتباط المصادر بالسؤال..."
            )
            eval_prompt = (
                f"هل المعلومات التالية مفيدة وتتعلق مباشرة بالسؤال الطبي المطروح؟ "
                f"أجب بكلمة 'نعم' أو 'لا' فقط.\n\nالسؤال: {query}\n\nالمعلومات:\n{context[:1000]}"
            )
            try:
                eval_resp = litellm.completion(
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=10,
                    temperature=0.0,
                    **kwargs,
                )
                eval_ans = (eval_resp.choices[0].message.content or "").strip()
                if "لا" in eval_ans or "No" in eval_ans:
                    print("[Pipeline] Context Evaluator REJECTED the retrieved chunks.")
                    _progress(
                        "evaluator",
                        "⚠️ معلومات غير مطابقة",
                        "المصادر المسترجعة قد لا تجيب على السؤال تماماً.",
                    )
                    # We no longer clear the context or search_json. We pass them to the LLM so the user can still see them,
                    # but we tell the LLM that the retrieved context is likely irrelevant.
                    context_rejected = True
                else:
                    print("[Pipeline] Context Evaluator ACCEPTED the retrieved chunks.")
                    context_rejected = False
            except Exception as e:
                print(f"[Pipeline] Context Evaluator failed: {e}")
                context_rejected = False
        else:
            context_rejected = False

        # ── 4b. Internet search via Serper ───────────────────────────────────
        web_sources: list[dict] = []
        serper_count = 0
        load_dotenv(dotenv_path=str(_ENV_PATH), override=True)
        # Serper strictly runs ONLY if mode explicitly asks for internet
        serper_enabled = (
            mode in ("internet", "all")
            and os.getenv("ENABLE_SERPER", "false").lower() == "true"
        )
        if serper_enabled:
            serper_key = os.getenv("SERPER_API_KEY", "")
            if serper_key:
                try:
                    import requests as _req

                    # Use ONLY the clean bare query — not the history-prepended version
                    serper_q = query.strip()
                    resp_s = _req.post(
                        "https://google.serper.dev/search",
                        headers={
                            "X-API-KEY": serper_key,
                            "Content-Type": "application/json",
                        },
                        json={"q": serper_q, "num": 5, "gl": "sa", "hl": "ar"},
                        timeout=10,
                    )
                    raw_json = resp_s.json()
                    hits = raw_json.get("organic", [])[:2]
                    print(
                        f"[Serper] query={serper_q!r} status={resp_s.status_code} organic={len(raw_json.get('organic', []))}"
                    )

                    # If 0 results, retry without Arabic locale (wider net)
                    if not hits:
                        print("[Serper] 0 results with ar locale — retrying globally")
                        resp_s2 = _req.post(
                            "https://google.serper.dev/search",
                            headers={
                                "X-API-KEY": serper_key,
                                "Content-Type": "application/json",
                            },
                            json={"q": serper_q, "num": 5},
                            timeout=10,
                        )
                        hits = resp_s2.json().get("organic", [])[:2]
                        print(f"[Serper] global retry organic={len(hits)}")

                    serper_count = len(hits)
                    for hit in hits:
                        web_sources.append(
                            {
                                "title": hit.get("title", "مصدر طبي"),
                                "url": hit.get("link", ""),
                                "snippet": hit.get("snippet", "")[:300],
                            }
                        )
                    _progress(
                        "serper",
                        "🌐 بحث إنترنت",
                        f"تم استرجاع {serper_count} مصدر إنترنت",
                    )
                    print(f"[Pipeline] serper results={serper_count}")
                except Exception as e:
                    print(f"[Pipeline] serper failed: {e}")
            else:
                print("[Pipeline] SERPER_API_KEY not set — skipping internet search")

        # Append web sources to context with full title+URL for proper citation
        if web_sources:
            web_block = "\n\n── مصادر إنترنت ──"
            for ws in web_sources:
                web_block += f"\n[{ws['title']} | {ws['url']}]\n{ws['snippet']}"
            context += web_block

        # ── 5. Single LLM call for Arabic medical response ───────────────────
        # Note: LLM logic now initialized at top of run() to support intent classifier early out.

        # Build reference list: ONLY web sources (max 2), as Arabic bullet points
        ref_lines: list[str] = []
        for ws in web_sources[:2]:  # already capped at 2 by Serper block
            ref_lines.append(f"  • {ws['title']} — {ws['url']}")
        ref_block = "\n".join(ref_lines) if ref_lines else ""

        # Use the Arabic specialization label from classifier
        cat_label = cls.get("category_arabic") or cls.get("category", "عام")
        # Build entity-focus instruction if a disease was extracted
        entity_focus = ""
        if disease_entity and disease_entity != query:
            entity_focus = (
                f"⚠️ الكيان المرضي المستخرج: «{disease_entity}» (القصد: {query_intent})\n"
                f"ركز إجابتك على هذا المرض/الحالة تحديداً. لا تخلط مع أمراض أخرى.\n\n"
            )
        prompt = (
            f"أنت طبيب عربي متخصص. أجب على السؤال الطبي الحالي بالعربية الفصحى فقط، "
            f"بناءً على السياق الطبي المسترجع أدناه وعلى سياق المحادثة السابقة إن وُجد.\n\n"
            f"{entity_focus}"
            f"{history_block}"
            f"السؤال الحالي: {query}\n\n"
            f"السياق الطبي:\n{context}\n\n"
            f"التعليمات الإلزامية:\n"
            f"0. يجب أن يكون ردك بالعربية الفصحى حصراً. لا تستخدم أبداً أي أحرف صينية أو يابانية أو كورية أو أي لغة غير العربية. إذا وجدت معلومات بلغة غير العربية في السياق، تجاهلها تماماً.\n"
            f"1. إذا كان السؤال لا يتعلق بالطب والصحة (مثل التحيات، أو الرموز التعبيرية، أو مواضيع عامة)، تجاهل جميع التعليمات التالية. فقط اعتذر بلباقة موضحاً أنك مستشار طبي ولا تجيب إلا على الاستفسارات الطبية، وتوقف.\n"
            f"2. ابدأ ردك دائماً بتحية إسلامية مناسبة.\n"
            f"3. التزم بالإجابة المباشرة والموجزة على ما طُلب فقط. إذا سأل المريض 'ما هو المرض'، اشرح ماهيته فقط دون إضافة الأعراض وطرق التشخيص والعلاج إلا إذا طُلب ذلك صراحة.\n"
            f"4. قدّم المعلومات في شكل نقاط واضحة (•) عند الحاجة للتفصيل.\n"
            f"5. كل معلومة يجب أن تكون مدعومة بالسياق المسترجع، ولا تخترع أي معلومة من خارج السياق.\n"
            f"6. إذا لم تجد المعلومة في السياق، قل: «هذه المعلومة غير متوفرة في المصادر المتاحة وتحتاج إلى مراجعة متخصص.»\n"
            f"7. بعد الانتهاء من الإجابة الطبية، اعرض التصنيف وسبب اختياره في هذا الشكل البارز بالضبط:\n"
            f"   ══════════════\n"
            f"   📂 التصنيف الطبي: {cat_label}\n"
            f"   💡 سبب التصنيف: [اكتب هنا باللغة العربية حصراً في سطر واحد مبررًا طبيًا موجزًا يوضح سبب انتماء الحالة لهذا التصنيف]\n"
            f"   ══════════════\n"
            + (
                f"8. **هام جداً:** السياق المسترجع الحالي يُعتبر غير مفيد أو غير كافٍ للرد على سؤال المستخدم. يرجى توضيح ذلك بلباقة للمستخدم واقترح عليه تغيير وضع البحث إلى 'الكل' (All) للبحث في الإنترنت، أو إعادة صياغة السؤال.\n"
                if context_rejected
                else ""
            )
            + (
                f"8. بعد قسم التصنيف وقبل المراجع، أضف قسم '💡 أسئلة مقترحة' يحتوي على 2-3 أسئلة مفيدة يمكن للمستخدم طرحها لمزيد من الاستكشاف.\n"
                f"9. أضف قسم المراجع التالية كنقاط (نقطتان فقط):\n"
                f"   📚 المراجع:\n{ref_block}\n"
                if ref_block
                else f"8. بعد قسم التصنيف، أضف قسم '💡 أسئلة مقترحة' يحتوي على 2-3 أسئلة مفيدة متعلقة بالموضوع.\n"
            )
            + f"10. اختم بهذا التنبيه الطبي بالضبط: "
            + f'"⚕️ تنبيه: هذه المعلومات لأغراض تعليمية فقط ولا تُغني عن استشارة الطبيب المختص."\n\n'
            + f"الإجابة:"
        )

        _progress("classify", "🏥 تصنيف طبي", f"التخصص: {cat_display}")

        _progress(
            "llm",
            f"🤖 توليد الرد [{model.split('/')[-1]}]",
            "إرسال السياق إلى النموذج...",
        )
        print(f"[Pipeline] calling LLM: {model}")
        resp = litellm.completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,
            **kwargs,
        )
        answer = resp.choices[0].message.content or ""
        # Apply CJK character filter to prevent Chinese/Japanese/Korean leakage
        answer = _clean_arabic_response(answer)
        _progress("llm", "🤖 توليد الرد", f"اكتمل الرد ({len(answer)} حرف)")
        print(f"[Pipeline] answer_len={len(answer)}")

        # ── 6. Lightweight hallucination check ───────────────────────────────
        _progress("hallucination", "✅ تدقيق المحتوى", "فحص الادعاءات غير المدعومة...")
        halluc_warning = False
        try:
            halluc_json = HallucinationCheckerTool()._run(answer, context)
            halluc = json.loads(halluc_json)
            if halluc.get("flagged_count", 0) > 0:
                answer += "\n\n⚠️ ملاحظة: قد تحتاج بعض المعلومات إلى مراجعة إضافية."
                halluc_warning = True
        except Exception as e:
            print(f"[Pipeline] hallucination check skipped: {e}")

        # ── 6.5 Append retrieved docs for transparency ────────────────────────
        if search_json:
            try:
                sd = json.loads(search_json)
                chunks = sd.get("chunks", [])
                if chunks:
                    docs_text = "\n\n━━━━━━━━━━━━━━━\n"
                    docs_text += "📑 *المصادر الطبية المسترجعة:*\n"
                    for i, c in enumerate(chunks, 1):
                        q = c.get("question", "").replace("\n", " ").strip()
                        t = c.get("text", "").replace("\n", " ").strip()
                        docs_text += f"\n*{i}. {q}*\n{t[:150]}...\n"
                    answer += docs_text
            except Exception as e:
                print(f"[Pipeline] failed to append docs: {e}")

        # ── 7. Return answer + real metadata as JSON ─────────────────────────
        result = json.dumps(
            {
                "final_answer": answer,
                "meta": {
                    "results": results_count,
                    "bm25_used": bm25_used,
                    "serper_count": serper_count,
                    "halluc_warn": halluc_warning,
                    "web_sources": web_sources,  # full list with title+url for UI
                },
            },
            ensure_ascii=False,
        )

        elapsed = time.time() - t0
        try:
            mlflow.log_metric("elapsed_time", elapsed)
            mlflow.log_metric("results_count", results_count)
            mlflow.log_metric("bm25_used", int(bm25_used))
            mlflow.log_metric("serper_count", serper_count)
            mlflow.log_metric("hallucination_warning", int(halluc_warning))
            mlflow.log_metric("answer_length", len(answer))
            mlflow.set_tag("category", cat_display)
            mlflow.set_tag("language", detected if "detected" in dir() else "unknown")
            mlflow.end_run()
        except Exception:
            pass

        # Close the GenAI root span
        try:
            if _root_span is not None:
                _root_span.set_outputs({
                    "answer_length": len(answer),
                    "results_count": results_count,
                    "elapsed_seconds": round(elapsed, 2),
                })
                _root_span.end()
        except Exception:
            pass

        return result
