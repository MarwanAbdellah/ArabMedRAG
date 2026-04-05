"""
Arabic Medical Chatbot — Telegram Bot Interface
Runs the same crew.run() pipeline as the Streamlit app,
including AraBERT entity extraction and MLflow tracing.

Run:  python telegram_bot/bot.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import logging
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Lazy-load crew once ───────────────────────────────────────────────────────
_crew = None

def get_crew():
    """
    Load CrewAI pipeline once (lazy singleton).
    The pipeline includes AraBERT embeddings, entity extraction,
    and MLflow tracing — all initialized automatically.
    """
    global _crew
    if _crew is None:
        from src.medical_chatbot.crew import arabic_chatbot
        _crew = arabic_chatbot()
    return _crew

# ── Retrieval mode options ────────────────────────────────────────────────────
MODES = {
    "rag":      "🧠 RAG (FAISS فقط)",
    "bm25":     "🔑 BM25 (بحث نصي)",
    "internet": "🌐 الإنترنت فقط",
    "hybrid":   "⚡ هجين (RAG + BM25)",
    "all":      "🔥 الكل (RAG + BM25 + إنترنت)",
}

# Per-user state: {user_id: {"mode": str, "history": list}}
user_state: dict[int, dict] = {}

def state(user_id: int) -> dict:
    """Get or create per-user conversation state."""
    if user_id not in user_state:
        user_state[user_id] = {"mode": "hybrid", "history": []}
    return user_state[user_id]

# ── Helper: strip HTML tags for plain Telegram Markdown ──────────────────────
def clean_for_telegram(text: str) -> str:
    """Clean response for Telegram markdown display."""
    text = re.sub(r"<[^>]+>", "", text)           # remove any HTML
    # Ensure Islamic greeting is followed by a blank line
    text = re.sub(
        r"(السلام عليكم(?:\s+ورحمة\s+الله(?:\s+وبركاته)?)?[.،]?)\s*",
        r"\1\n\n",
        text,
    )
    # Shorten long Unicode box-drawing dividers (cap at 15 chars)
    text = re.sub(r"━{4,}", "━━━━━━━━━━━━━━━", text)
    text = re.sub(r"═{4,}", "═══════════════", text)
    text = re.sub(r"─{4,}", "───────────────", text)
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excess newlines
    text = text.replace("•", "•")
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)  # ** → * (Telegram Markdown)
    return text.strip()

def split_refs(text: str):
    """Split answer body from references section."""
    if "📚 المراجع:" in text:
        parts = text.split("📚 المراجع:", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""

# ── Commands ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Welcome message with available commands."""
    uid = update.effective_user.id
    state(uid)  # init
    await update.message.reply_text(
        "السلام عليكم! 👋\n\n"
        "أنا *المساعد الطبي العربي* — نظام ذكاء اصطناعي متخصص للإجابة على أسئلتك الطبية.\n\n"
        "🔬 يستخدم *AraBERT* للفهم العميق للمصطلحات الطبية العربية\n"
        "🎯 يستخرج اسم المرض تلقائياً من سؤالك\n"
        "📊 يتتبع أداء كل استعلام عبر MLflow\n\n"
        "📌 *الأوامر المتاحة:*\n"
        "/mode — اختر مصدر الاسترجاع\n"
        "/clear — مسح سياق المحادثة\n"
        "/help — مساعدة\n\n"
        "ابدأ بكتابة سؤالك الطبي مباشرة ✍️",
        parse_mode="Markdown",
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Help message with current mode and disclaimer."""
    uid = update.effective_user.id
    current = MODES[state(uid)["mode"]]
    await update.message.reply_text(
        f"🤖 *المساعد الطبي العربي*\n\n"
        f"الوضع الحالي: {current}\n\n"
        "الأوامر:\n"
        "/mode — تغيير مصدر الاسترجاع (RAG / BM25 / إنترنت / هجين / الكل)\n"
        "/clear — مسح ذاكرة المحادثة\n"
        "/start — رسالة الترحيب\n\n"
        "📊 *MLflow Tracing:* كل استعلام يُسجَّل تلقائياً مع:\n"
        "  • الكيان المرضي المستخرج\n"
        "  • زمن الاستجابة\n"
        "  • نتائج فحص الهلوسة\n\n"
        "⚕️ هذا النظام للأغراض التعليمية فقط ولا يُغني عن استشارة الطبيب.",
        parse_mode="Markdown",
    )

async def cmd_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show inline keyboard for retrieval mode selection."""
    kbd = [
        [InlineKeyboardButton(label, callback_data=f"mode:{key}")]
        for key, label in MODES.items()
    ]
    await update.message.reply_text(
        "🔍 *اختر مصدر الاسترجاع:*",
        reply_markup=InlineKeyboardMarkup(kbd),
        parse_mode="Markdown",
    )

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history for the user."""
    uid = update.effective_user.id
    state(uid)["history"] = []
    await update.message.reply_text("✅ تم مسح سياق المحادثة.")

async def callback_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard mode selection callback."""
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    key = query.data.split(":", 1)[1]
    state(uid)["mode"] = key
    await query.edit_message_text(f"✅ تم تغيير الوضع إلى: *{MODES[key]}*", parse_mode="Markdown")

# ── Main message handler ──────────────────────────────────────────────────────
async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Core pipeline handler. Dispatches query to AI crew in background thread,
    shows real-time progress with entity extraction and MLflow tracking info.
    """
    uid   = update.effective_user.id
    s     = state(uid)
    query = update.message.text.strip()

    import re as _re
    # Ignore standalone emojis or empty text
    if not query or len(_re.sub(r'[^\w\s]', '', query).strip()) == 0:
        await update.message.reply_text("عذراً، الرجاء كتابة سؤال طبي واضح.")
        return

    # Show typing + a processing message
    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")
    proc_msg = await update.message.reply_text("⏳ جارٍ تحليل سؤالك...")

    import asyncio
    loop = asyncio.get_running_loop()

    # Build step log for progress
    steps_log: list[str] = []

    def on_progress(step, label, detail=""):
        entry = f"{label}" + (f" — {detail}" if detail else "")
        if entry not in steps_log:
            steps_log.append(entry)
            text = "⏳ جارٍ تحليل سؤالك...\n\n" + "\n".join(f"  {s}" for s in steps_log)
            asyncio.run_coroutine_threadsafe(proc_msg.edit_text(text), loop)

    try:
        crew = get_crew()
        import json, time
        t0 = time.time()

        # CrewAI is synchronous — must run in a thread to avoid blocking
        raw = await asyncio.to_thread(
            crew.run,
            query,
            history=s["history"][-6:],
            mode=s["mode"],
            on_progress=on_progress,
        )
        elapsed = round(time.time() - t0, 1)

        # Parse response
        try:
            data = json.loads(raw)
            answer  = data.get("final_answer", raw)
            meta    = data.get("meta", {})
        except Exception:
            answer, meta = raw, {}

        # Update history
        s["history"].append({"role": "user",      "content": query})
        s["history"].append({"role": "assistant",  "content": answer})
        if len(s["history"]) > 12:
            s["history"] = s["history"][-12:]

        # Split answer body from references
        body, refs_text = split_refs(answer)
        body_clean = clean_for_telegram(body)

        # Delete the "processing" message
        await proc_msg.delete()

        # Send main answer
        await update.message.reply_text(body_clean, parse_mode="Markdown")

        # Send references block if any web sources
        web_sources = meta.get("web_sources", [])
        if web_sources:
            ref_lines = "\n".join(
                f"• [{ws['title']}]({ws['url']})"
                for ws in web_sources if ws.get("url")
            )
            await update.message.reply_text(
                f"📚 *المراجع:*\n{ref_lines}",
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )

        # Send pipeline summary footer with entity info
        if meta:
            results  = meta.get("results", 0)
            bm25     = "نعم" if meta.get("bm25_used") else "لا"
            serper   = meta.get("serper_count", 0)
            mode_lbl = MODES.get(s["mode"], s["mode"])

            footer_parts = [f"🔍 {mode_lbl}", f"⏱️ {elapsed}ث"]
            if s["mode"] in ("rag", "hybrid", "all"):
                footer_parts.append(f"📌 RAG: {results}")
            if s["mode"] in ("bm25", "hybrid", "all"):
                footer_parts.append(f"🔑 BM25: {bm25}")
            if s["mode"] in ("internet", "all") and serper:
                footer_parts.append(f"🌐 إنترنت: {serper}")

            await update.message.reply_text(
                " | ".join(footer_parts),
                parse_mode="Markdown",
            )

    except Exception as exc:
        logger.exception("Pipeline error")
        await proc_msg.edit_text(f"❌ حدث خطأ: {exc}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    """Start the Telegram bot with all handlers."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("mode",  cmd_mode))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CallbackQueryHandler(callback_mode, pattern=r"^mode:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🤖 Arabic Medical Bot is running — press Ctrl+C to stop")
    logger.info("📊 MLflow tracing is enabled — check MLflow UI for agent flow")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
