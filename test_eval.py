import os
import sys
import json
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(__file__))

from src.medical_chatbot.tools.hybrid_search_tool import HybridSearchTool
from src.medical_chatbot.tools.citation_tool import CitationGroundingTool
import litellm

def main():
    query = "ما هو مرض السكر"
    
    search_json = HybridSearchTool()._run(query)
    data = json.loads(search_json)
    
    cite_json = CitationGroundingTool()._run(search_json)
    cite = json.loads(cite_json)
    context = cite.get("context", "")
    if len(context) > 2500:
        context = context[:2500] + "\n[...]"
    
    eval_prompt = (
        f"هل المعلومات التالية مفيدة وتتعلق مباشرة بالسؤال الطبي المطروح؟ "
        f"أجب بكلمة 'نعم' أو 'لا' فقط.\n\nالسؤال: {query}\n\nالمعلومات:\n{context[:1000]}"
    )
    
    try:
        eval_resp = litellm.completion(
            model=os.getenv("OPENROUTER_MODEL", "groq/llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=10,
            temperature=0.0
        )
        eval_ans = (eval_resp.choices[0].message.content or "").strip()
    except Exception as e:
        eval_ans = str(e)

    res = {
        "hybrid_search": data,
        "evaluator_prompt": eval_prompt,
        "evaluator_ans": eval_ans
    }
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
