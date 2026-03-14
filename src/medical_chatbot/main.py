"""
main.py
────────────────────────────────────────────────
CLI entrypoint for the Arabic Medical Chatbot.

Usage:
    # Interactive mode
    python src/medical_chatbot/main.py

    # Single query
    python src/medical_chatbot/main.py --query "ما هي أعراض ارتفاع ضغط الدم؟"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,   # Reduce noise in CLI; set INFO for debugging
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

BANNER = """
╔══════════════════════════════════════════════════════╗
║       🏥  المساعد الطبي العربي بالذكاء الاصطناعي     ║
║          Arabic Medical AI Assistant (CrewAI)        ║
╚══════════════════════════════════════════════════════╝
"""


# Description: A handy helper function to spin up the chatbot for a single question directly from the command line.
def run_query(query: str) -> str:
    """Run the crew pipeline on a single query."""
    from src.medical_chatbot.crew import arabic_chatbot
    bot = arabic_chatbot()
    return bot.run(query)


# Description: The entrypoint for developers! Run this from your terminal to launch the interactive prompt loop.
def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Arabic Medical AI Assistant – CrewAI powered chatbot"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Medical query to answer (if omitted, enters interactive mode)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    print(BANNER)

    if args.query:
        # Single-shot mode
        print(f"📝 السؤال: {args.query}\n")
        print("⏳ جارٍ التفكير...\n")
        answer = run_query(args.query)
        print("─" * 60)
        print(answer)
        print("─" * 60)
    else:
        # Interactive mode
        print("اكتب سؤالك الطبي بالعربية أو الإنجليزية.")
        print("اكتب 'خروج' أو 'exit' للإنهاء.\n")

        while True:
            try:
                query = input("❓ سؤالك: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 وداعاً!")
                break

            if not query:
                continue
            if query.lower() in ("خروج", "exit", "quit", "q"):
                print("👋 وداعاً!")
                break

            print("\n⏳ جارٍ التفكير...\n")
            try:
                answer = run_query(query)
                print("─" * 60)
                print(answer)
                print("─" * 60 + "\n")
            except Exception as exc:
                print(f"❌ خطأ: {exc}\n")


if __name__ == "__main__":
    cli()
