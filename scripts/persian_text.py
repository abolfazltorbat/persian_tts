"""Shared, inspectable Persian speech preparation; never feed G2P to Qwen.

Keep vowels/ezafe, Latin text and meaningful ZWNJ. The Pocket G2P still
applies its own training-specific normalization after this common step.
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model" / "v2"))
from normalize_fa import numbers_to_words, number_to_words

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_FORMS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ـ": "", "ۀ": "هٔ"})
_INVISIBLE = re.compile(r"[\u200b\u200d-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")


def prepare_persian(text: str, enabled: bool = True) -> str:
    if not enabled:
        return text.strip()
    text = _INVISIBLE.sub("", unicodedata.normalize("NFC", text)).translate(_FORMS)
    text = text.translate(_DIGITS).replace("٫", ".").replace("٬", ",")
    # Consume complete groups BEFORE punctuation and sentence splitting.
    text = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)",
                  lambda m: m[0].replace(",", ""), text)

    def clock(m):
        hour, minute = map(int, m.groups())
        if hour > 23 or minute > 59:
            return m[0]
        return number_to_words(hour) + (" و " + number_to_words(minute) + " دقیقه" if minute else "")

    text = re.sub(r"(?<![\d:])([0-2]?\d):([0-5]\d)(?![\d:])", clock, text)
    text = re.sub(r"(?<![\w\d])[-−](?=\d)", "منفی ", text)
    text = re.sub(r"(?<![\w\d])\+(?=\d)", "مثبت ", text)
    text = re.sub(r"[%٪]", " درصد ", text)
    text = numbers_to_words(text)
    text = text.translate(str.maketrans({"?": "؟", ",": "،", ";": "؛", "…": "…"}))
    text = re.sub(r"\u200c+", "\u200c", text)
    text = re.sub(r"\s*\u200c\s*", "\u200c", text)
    # Keep newlines as sentence boundaries rather than dropping paragraph pauses.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([،؛؟!:.])", r"\1", text)
    return text.strip()


def text_chunks(text: str, max_chars: int = 220) -> list[str]:
    """Qwen sentence/phrase chunks, preserving punctuation and word boundaries."""
    chunks = []
    for sentence in re.split(r"(?<=[.!؟?])\s+|\n+", text.strip()):
        if not any(c.isalnum() for c in sentence):
            continue
        current = ""
        for phrase in re.split(r"(?<=[،؛,:;])\s+", sentence):
            for word in phrase.split():
                if current and len(current) + len(word) + 1 > max_chars:
                    chunks.append(current)
                    current = ""
                current = (current + " " + word).strip()
            # Prefer a phrase joint once the chunk is reasonably substantial.
            if len(current) >= max_chars // 2:
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)
    return chunks


PERSIAN_CASES = [
    {"id": "everyday", "name": "گفت‌وگوی روزمره", "text": "سلام، حال شما چطور است؟ امیدوارم روز خوبی داشته باشید."},
    {"id": "ezafe", "name": "اضافه و ترکیب‌ها", "text": "کتابِ تازهٔ نویسندهٔ ایرانی روی میزِ چوبی کنار پنجره است."},
    {"id": "numbers", "name": "عدد، ساعت و درصد", "text": "جلسه ساعت ۱۴:۳۰ شروع می‌شود. قیمت بلیت ۱٬۲۵۰٬۰۰۰ تومان است و ۱۲٫۵٪ تخفیف دارد."},
    {"id": "names", "name": "نام‌های فارسی", "text": "پژمان و ژاله از شیراز به اصفهان رفتند و در زاینده‌رود قدم زدند."},
    {"id": "zwnj", "name": "نیم‌فاصله و فعل‌ها", "text": "دانش‌آموزها می‌خواهند کتاب‌هایشان را بخوانند، اما نمی‌توانند تمرکز کنند."},
    {"id": "question", "name": "لحن و مکث", "text": "واقعاً امروز می‌آیی؟ چه خبر خوبی! پس منتظرت می‌مانم؛ لطفاً دیر نکن."},
    {"id": "mixed", "name": "فارسی و انگلیسی", "text": "برای اجرای Python روی Linux، ابتدا API سرور را بررسی کنید."},
    {"id": "negative", "name": "عدد منفی و شماره", "text": "دمای هوا منفی پنج درجه بود و حالا به −۲٫۵ درجه رسیده است. شماره تماس ۰۹۱۲۳۴۵۶۷۸۹ است."},
]
