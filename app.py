"""
app.py
────────────────────────────────────────────────
Streamlit Web UI for the Arabic Medical Chatbot.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pathlib

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Force-load .env, overriding any stale in-process env vars
_ENV_PATH = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH), override=True)

# ─────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="المساعد الطبي العربي | Arabic Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
#  CSS Styling
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');

  :root {
    --primary:   #1a6b4a;
    --secondary: #0f4d34;
    --accent:    #00c97d;
    --bg-dark:   #0d1117;
    --bg-card:   #161b22;
    --bg-input:  #21262d;
    --border:    #30363d;
    --text:      #e6edf3;
    --text-muted:#8b949e;
    --danger:    #f85149;
    --warning:   #d29922;
  }

  html, body, [class*="css"] {
    font-family: 'Cairo', 'Inter', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
  }

  /* ── Chat messages ── */
  .user-bubble {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    margin: 10px 0 10px 20%;
    color: white;
    font-size: 1.05rem;
    line-height: 1.7;
    box-shadow: 0 4px 15px rgba(0,201,125,0.15);
    direction: auto;
  }

  .bot-bubble {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 18px 24px;
    margin: 10px 20% 10px 0;
    font-size: 1rem;
    line-height: 1.9;
    direction: rtl;
    text-align: right;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }

  .emergency-bubble {
    background: linear-gradient(135deg, #3d1515, #5c1a1a);
    border: 2px solid var(--danger);
    border-radius: 14px;
    padding: 18px 24px;
    margin: 10px 0;
    direction: rtl;
    text-align: right;
    animation: pulse-red 2s infinite;
  }

  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.4); }
    50%       { box-shadow: 0 0 0 10px rgba(248,81,73,0); }
  }

  /* ── Citations box ── */
  .citations-box {
    background: #1c2128;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 12px;
    font-size: 0.88rem;
    color: var(--text-muted);
  }

  /* ── References box (gradient) ── */
  .references-box {
    background: linear-gradient(135deg, #0d2137 0%, #112d1f 50%, #1a1a2e 100%);
    border: 1px solid rgba(0,201,125,0.25);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 8px 20% 4px 0;
    direction: rtl;
    text-align: right;
  }
  .references-box .ref-title {
    color: var(--accent);
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 8px;
    display: block;
  }
  .references-box a {
    color: #58a6ff;
    text-decoration: none;
    font-size: 0.87rem;
  }
  .references-box a:hover { text-decoration: underline; }
  .references-box .ref-item {
    display: block;
    padding: 5px 0;
    color: var(--text-muted);
    font-size: 0.87rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .references-box .ref-item:last-child { border-bottom: none; }

  /* ── Stats badge ── */
  .stat-badge {
    display: inline-block;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 2px;
  }

  .stat-accent { color: var(--accent); }

  /* ── Disclaimer ── */
  .disclaimer {
    background: #1c2128;
    border: 1px solid var(--warning);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: var(--warning);
    margin-top: 10px;
    direction: rtl;
    text-align: right;
  }

  /* ── Input area ── */
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 1rem !important;
    direction: rtl;
  }

  [data-testid="stTextInput"] input:focus,
  [data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,201,125,0.2) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 28px !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,201,125,0.35) !important;
  }

  /* ── Headings ── */
  h1, h2, h3 { color: var(--text) !important; }

  /* ── Hero header ── */
  .hero-header {
    text-align: center;
    padding: 30px 0 20px 0;
  }
  .hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #00a86b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-top: 6px;
  }

  /* ── Spinner ── */
  .thinking-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    color: var(--text-muted);
    font-size: 0.95rem;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-dark); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "crew"          not in st.session_state: st.session_state.crew          = None
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "retrieval_mode" not in st.session_state: st.session_state.retrieval_mode = "rag"


# Description: We cache the CrewAI setup here so Streamlit doesn't reload heavy models on every UI interaction.
@st.cache_resource(show_spinner="جارٍ تحميل نظام الذكاء الاصطناعي...")
def load_chatbot_v3():
    """Load the crew (v3 — busts old cache)."""
    from src.medical_chatbot.crew import arabic_chatbot
    return arabic_chatbot()


# Description: Since CrewAI gives us a JSON structure, we gracefully parse out the final text. If parsing fails, we fallback to returning raw text.
def _extract_final_answer(raw: str) -> tuple[str, list[str], bool, dict]:
    """
    Parse crew.run() JSON output: {final_answer, meta}.
    Returns (answer, citations, is_emergency, meta).
    Falls back gracefully to raw text.
    """
    is_emergency = False
    citations: list[str] = []
    meta: dict = {}

    try:
        data = json.loads(raw)
        answer = data.get("final_answer", raw)
        meta   = data.get("meta", {})
        unsupported = data.get("unsupported_claims", [])
        if unsupported:
            answer += "\n\n⚠️ ملاحظة: بعض الجمل قد تحتاج إلى مراجعة."
    except (json.JSONDecodeError, TypeError):
        answer = raw

    if "طارئة" in answer or "طوارئ" in answer or "فورًا" in answer:
        is_emergency = True

    return answer, citations, is_emergency, meta


# ─────────────────────────────────────────────────────
#  Sidebar  (info only – LLM is configured via .env)
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
      <span style='font-size:2.5rem;'>🏥</span>
      <h2 style='margin:8px 0 4px 0; font-size:1.3rem;'>المساعد الطبي العربي</h2>
      <p style='color:var(--text-muted); font-size:0.85rem;'>Arabic Medical AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── LLM status (read-only, from .env) ─────────────
    _provider = os.getenv("LLM_PROVIDER", "ollama").upper()
    _model    = (
        os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-72b-instruct")
        if _provider == "OPENROUTER"
        else os.getenv("OLLAMA_MODEL", "qwen2.5")
    )
    st.markdown("#### 🤖 النموذج المستخدم")
    st.markdown(
        f"<div style='background:var(--bg-input);border:1px solid var(--border);"
        f"border-radius:10px;padding:10px 14px;font-size:0.88rem;'>"
        f"<b style='color:var(--accent);'>{_provider}</b><br>"
        f"<span style='color:var(--text-muted);'>{_model.split('/')[-1]}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 📊 إحصائيات الجلسة")
    col1, col2 = st.columns(2)
    col1.metric("الأسئلة", st.session_state.total_queries)
    col2.metric("نتائج الذاكرة", "5")

    st.divider()

    st.markdown("#### 📂 الفئات الطبية المتاحة")
    categories = [
        "أمراض النساء والتوليد", "أمراض القلب", "الجهاز الهضمي",
        "الأمراض الجلدية", "العظام والمفاصل", "طب الأطفال",
        "طب العيون", "الأمراض النفسية", "الأورام وغيرها",
    ]
    for cat in categories:
        st.markdown(f"<span class='stat-badge'>• {cat}</span>", unsafe_allow_html=True)

    st.divider()

    # ── Retrieval mode selector ────────────────────────
    st.markdown("#### 🔍 مصدر الاسترجاع")
    mode_labels = {
        "rag":      "🧠 RAG (FAISS فقط)",
        "bm25":     "🔑 BM25 (بحث نصي)",
        "internet": "🌐 الإنترنت فقط (Serper)",
        "hybrid":   "⚡ هجين (RAG + BM25)",
        "all":      "🔥 الكل (RAG + BM25 + إنترنت)",
    }
    chosen_label = st.radio(
        label="",
        options=list(mode_labels.values()),
        index=list(mode_labels.keys()).index(st.session_state.retrieval_mode),
        label_visibility="collapsed",
    )
    # Map label back to key
    label_to_key = {v: k for k, v in mode_labels.items()}
    st.session_state.retrieval_mode = label_to_key[chosen_label]

    st.divider()

    if st.button("ഷ️‍♂️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("""
    <div style='text-align:center; color:var(--text-muted); font-size:0.75rem; margin-top:20px;'>
      Powered by CrewAI · SentenceTransformers · FAISS · BM25
    </div>
    """, unsafe_allow_html=True)




# ─────────────────────────────────────────────────────
#  Main Area
# ─────────────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
  <div class='hero-title'>🏥 المساعد الطبي العربي</div>
  <div class='hero-subtitle'>
    نظام متعدد الوكلاء · استرجاع هجين · SentenceTransformers · Qwen
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick examples ──
EXAMPLE_QUERIES = [
    "ما هي أعراض ارتفاع ضغط الدم؟",
    "ما هو علاج السكري من النوع الثاني؟",
    "أعاني من ألم في الصدر وضيق في التنفس",
    "ما هي أعراض الحمل المبكر؟",
    "كيف أعرف أن لدي التهاباً في الحلق؟",
]

st.markdown("**💡 أسئلة سريعة:**")
cols = st.columns(len(EXAMPLE_QUERIES))
selected_example = None
for i, (col, ex) in enumerate(zip(cols, EXAMPLE_QUERIES)):
    if col.button(ex[:25] + "...", key=f"ex_{i}", help=ex):
        selected_example = ex

st.divider()

# ── Chat history ──
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>🙋 {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            content  = msg["content"]
            is_emerg = msg.get("is_emergency", False)
            if is_emerg:
                st.markdown(
                    f"<div class='emergency-bubble'>🚨 {content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Strip the LLM-written references block from the answer body
                if "📚 المراجع:" in content:
                    answer_body = content.split("📚 المراجع:", 1)[0].strip()
                else:
                    answer_body = content

                st.markdown(
                    f"<div class='bot-bubble'>🤖 {answer_body}</div>",
                    unsafe_allow_html=True,
                )

                # Build references box from meta.web_sources (guaranteed to have URLs)
                web_sources = msg.get("meta", {}).get("web_sources", [])
                if web_sources:
                    ref_items_html = "".join(
                        f"<span class='ref-item'>• <a href='{ws['url']}' target='_blank'>{ws['title']}</a></span>"
                        for ws in web_sources if ws.get("url")
                    )
                    if ref_items_html:
                        st.markdown(
                            "<div class='references-box'>"
                            "<span class='ref-title'>📚 المراجع</span>"
                            + ref_items_html
                            + "</div>",
                            unsafe_allow_html=True,
                        )
            if msg.get("meta"):
                meta  = msg["meta"]
                mode  = meta.get("mode", "hybrid")
                parts = [f"⏱️ زمن الاستجابة: {meta.get('elapsed', '?')}ث"]
                if mode in ("rag", "hybrid", "all"):
                    parts.insert(0, f"📌 نتائج RAG: {meta.get('results', 0)}")
                if mode in ("bm25", "hybrid", "all"):
                    parts.insert(-1, f"🔑 BM25: {'نعم' if meta.get('bm25') else 'لا'}")
                if mode in ("internet", "all") and meta.get("serper"):
                    parts.insert(-1, f"🌐 إنترنت: {meta['serper']}")
                st.markdown(
                    f"<div class='citations-box'>{' | '.join(parts)}</div>",
                    unsafe_allow_html=True,
                )

# ── Input form ──
st.markdown("<br>", unsafe_allow_html=True)
with st.form("query_form", clear_on_submit=True):
    query_input = st.text_input(
        label="اطرح سؤالك الطبي",
        placeholder="مثال: ما هي أعراض التهاب المفاصل؟",
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("🔍 إرسال", use_container_width=True)

# Merge submitted query with example clicks
final_query = (query_input.strip() if submit and query_input.strip() else None) or selected_example

if final_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": final_query})
    st.session_state.total_queries += 1

    # Extract short-term memory (last 4 messages before the current one)
    recent_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-5:-1]
        if m["role"] in ["user", "assistant"]
    ]

    # Live pipeline progress via st.status()
    with st.status("⏳ جارٍ تحليل سؤالك...", expanded=True) as status_box:
        try:
            # Description: This acts as our callback hook. As the agents complete tasks, this function instantly updates the Streamlit UI with live progress indicators.
            def _on_progress(step, label, detail=""):status_box.write(f"**{label}** — {detail}" if detail else f"**{label}**")

            crew = load_chatbot_v3()
            t_start = time.time()
            raw_result = crew.run(
                final_query,
                history=recent_history,
                mode=st.session_state.retrieval_mode,
                on_progress=_on_progress,
            )
            elapsed = round(time.time() - t_start, 1)
            status_box.update(label=f"✅ اكتملت المعالجة ({elapsed}ث)", state="complete", expanded=False)

            answer, citations, is_emergency, pipe_meta = _extract_final_answer(raw_result)

            if pipe_meta:
                st.session_state.messages.append({
                    "role":         "assistant",
                    "content":      answer,
                    "is_emergency": is_emergency,
                    "meta": {
                        "results":     pipe_meta.get("results", 5),
                        "bm25":        pipe_meta.get("bm25_used", False),
                        "serper":      pipe_meta.get("serper_count", 0),
                        "elapsed":     elapsed,
                        "web_sources": pipe_meta.get("web_sources", []),
                        "mode":        st.session_state.retrieval_mode,
                    },
                })
            else:
                st.session_state.messages.append({
                    "role":         "assistant",
                    "content":      answer,
                    "is_emergency": is_emergency,
                })
        except Exception as exc:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"❌ حدث خطأ: {exc}",
            })

    st.rerun()
