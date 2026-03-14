"""
classifier_tool.py
────────────────────────────────────────────────
Classifies a medical query into a category and detects
emergency symptoms for priority routing.

Classification approach:
  1. Emergency keyword check (always runs first)
  2. Category keyword matching over Arabic + English terms
  3. Fallback to "general_health"
"""

from __future__ import annotations

import json
import logging
import re

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
#  Emergency symptoms (fast path)
# ─────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    # Arabic
    "ألم حاد في الصدر", "ضيق تنفس", "صعوبة التنفس", "فقدان الوعي",
    "نزيف حاد", "سكتة دماغية", "شلل مفاجئ", "ألم شديد في الصدر",
    "ضيق في التنفس", "تشنجات", "احتشاء القلب", "ذبحة صدرية",
    # English (if query is in English)
    "chest pain", "difficulty breathing", "stroke", "severe bleeding",
    "loss of consciousness", "heart attack", "seizure", "unconscious",
]

EMERGENCY_RESPONSE = (
    "⚠️ قد تشير هذه الأعراض إلى حالة طبية طارئة. "
    "يرجى طلب المساعدة الطبية فورًا أو الاتصال بخدمات الطوارئ."
)


# ─────────────────────────────────────────────────────
#  Category keyword table
# ─────────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "dermatology": ["جلد", "طفح", "حساسية", "شعر", "بشرة", "skin", "rash", "dermatology"],
    "stds": ["جنسي", "تناسلي", "إيدز", "سيلان", "زهري", "std", "venereal", "sexually transmitted"],
    "oncology": ["ورم", "سرطان", "خبيث", "حميد", "علاج كيماوي", "cancer", "tumor", "oncology", "malignant"],
    "general_medicine": ["عام", "صحة", "نصيحة", "فيتامين", "تعب", "general", "health", "fatigue"],
    "pediatrics": ["طفل", "أطفال", "رضيع", "نمو طفل", "pediatrics", "child", "infant"],
    "pulmonology": ["تنفس", "رئة", "سعال", "ربو", "كحة", "pulmonology", "lung", "asthma", "cough", "breathing"],
    "gastroenterology": ["معدة", "إسهال", "إمساك", "كبد", "أمعاء", "هضم", "قولون", "بطن", "مغص", "قيء", "استفراغ", "gastroenterology", "stomach", "liver", "digestion", "abdomen", "colic"],
    "hematology": ["دم", "أنيميا", "فقر دم", "نزيف", "تخثر", "hematology", "blood", "anemia", "bleeding"],
    "orthopedics": ["عظم", "عضل", "مفصل", "كسر", "روماتيزم", "هشاشة", "كساح", "غضروف", "ديسك", "نقرس", "انزلاق", "orthopedics", "bone", "muscle", "joint", "fracture", "rickets", "cartilage", "gout", "disc"],
    "ophthalmology": ["عين", "رؤية", "بصر", "شبكية", "نظارة", "مياه بيضاء", "مياه زرقاء", "ophthalmology", "eye", "vision", "cornea", "retina", "glaucoma", "cataract"],
    "endocrinology": ["غدة", "سكري", "هرمون", "درقية", "صماء", "سمنة", "endocrinology", "gland", "diabetes", "hormone", "thyroid", "obesity"],
    "cardiology": ["قلب", "ضغط", "شرايين", "نبض", "كولسترول", "cardiology", "heart", "blood pressure", "artery", "pulse"],
    "urology": ["بول", "كلى", "حالب", "مسانة", "بروستات", "urology", "urine", "kidney", "bladder", "prostate"],
    "internal_medicine": ["باطن", "مناعة", "ألتهاب داخلي", "internal medicine", "immune", "internal"],
    "gynecology": ["دورة شهرية", "حمل", "ولادة", "رحم", "مبيض", "نساء", "gynecology", "pregnancy", "menstruation", "uterus"],
    "psychiatry_neurology": ["نفسي", "عصب", "اكتئاب", "قلق", "دماغ", "صرع", "psychiatry", "neurology", "depression", "anxiety", "brain", "nerves"],
    "ent": ["أذن", "أنف", "حنجرة", "سمع", "شم", "بلع", "ent", "otolaryngology", "ear", "nose", "throat"],
    "plastic_surgery": ["تجميل", "حرق", "تشوه", "شفط دهون", "plastic surgery", "cosmetic", "burns", "liposuction"],
    "general_surgery": ["جراح", "عملية", "تخدير", "خياطة", "استئصال", "surgery", "operation", "incision", "excision"],
    "dentistry": ["أسنان", "ضرس", "لثة", "تسوس", "تقويم", "dentistry", "tooth", "teeth", "gum", "cavity", "braces"],
    "emergency": []   # handled separately
}


# ─────────────────────────────────────────────────────
#  Arabic labels for each category
# ─────────────────────────────────────────────────────

CATEGORY_ARABIC_LABELS: dict[str, str] = {
    "dermatology": "الامراض الجلدية",
    "stds": "الامراض الجنسية",
    "oncology": "الاورام الخبيثة والحميدة",
    "general_medicine": "الطب العام",
    "pediatrics": "امراض الاطفال",
    "pulmonology": "امراض الجهاز التنفسي",
    "gastroenterology": "امراض الجهاز الهضمي",
    "hematology": "امراض الدم",
    "orthopedics": "امراض العضلات والعظام و المفاصل",
    "ophthalmology": "امراض العيون",
    "endocrinology": "امراض الغدد الصماء",
    "cardiology": "امراض القلب و الشرايين",
    "urology": "امراض المسالك البولية والتناسلية",
    "internal_medicine": "امراض باطنية",
    "gynecology": "امراض نسائية",
    "psychiatry_neurology": "امراض نفسية وعصبية",
    "ent": "انف اذن وحنجرة",
    "plastic_surgery": "جراحة تجميل",
    "general_surgery": "جراحة عامة",
    "dentistry": "طب الاسنان",
    "emergency": "حالة طارئة"
}


# Description: Normalizes the raw query text by wiping away Arabic diacritics, ensuring reliable dictionary lookups.
def _normalize(text: str) -> str:
    return re.sub(r"[\u064B-\u065F]", "", text.lower())


# Description: Schema to ensure the tool receives exactly one query parameter.
class ClassifierInput(BaseModel):
    query: str = Field(..., description="The medical query to classify.")


# Description: This tool executes high-speed rule-based classification over user queries, automatically routing things like emergencies off an ultra-fast logic branch.
class MedicalClassifierTool(BaseTool):
    name: str        = "medical_classifier_tool"
    description: str = (
        "Classify a medical query into a category. "
        "Also detects emergency symptoms and triggers priority routing. "
        "Returns JSON with 'category', 'is_emergency', and optionally 'emergency_response'."
    )
    args_schema: type[BaseModel] = ClassifierInput

    # Description: Fires regular expressions over the query! Checks for emergencies first before cascading down into broad categories.
    def _run(self, query: str) -> str:
        normalized_query = _normalize(query)

        # ── Emergency detection ──
        is_emergency = any(
            _normalize(kw) in normalized_query for kw in EMERGENCY_KEYWORDS
        )
        if is_emergency:
            result = {
                "category":          "emergency",
                "category_arabic":   CATEGORY_ARABIC_LABELS["emergency"],
                "is_emergency":      True,
                "emergency_response": EMERGENCY_RESPONSE,
            }
            logger.warning(f"Emergency detected for query: {query[:80]}")
            return json.dumps(result, ensure_ascii=False)

        # ── Category matching ──
        best_category = "general_medicine"
        best_count    = 0

        for category, keywords in CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if _normalize(kw) in normalized_query)
            if count > best_count:
                best_count    = count
                best_category = category

        result = {
            "category":        best_category,
            "category_arabic": CATEGORY_ARABIC_LABELS.get(best_category, best_category),
            "is_emergency":    False,
        }
        logger.info(f"Query classified as: {best_category}")
        return json.dumps(result, ensure_ascii=False)
