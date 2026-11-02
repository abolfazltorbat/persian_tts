"""Text front-end: what an answer SAYS, sentence by sentence, in which language.

The acoustic model reads phonemes; the G2P reads Persian words. Everything an
assistant writes that is neither — digits, dates, units, currency signs, ranges,
ordinals, abbreviations, Latin brand names glued together in camelCase — has to
become words first, and the RIGHT words for the sentence it sits in: «۲۵ GB»
inside Persian is «بیست و پنج گیگابایت», "25 GB" inside English is "twenty five
gigabytes". Measured before this module existed (eval/harness.py, Whisper
round trip, 3 seeds): dates came out as digit strings (CER 0.46–0.49), «۴۰°C»
as «چهل سی», «$20» lost its «دلار», «۳-۵» lost its «تا», «−۵» lost its sign,
and English sentences had their numbers read in Persian.

    from frontend import speakable, render
    speakable("جلسه ۱۴۰۳/۰۵/۲۰ ساعت ۱۴:۳۰ است.")
      -> [Sentence(text='جلسه بیستم مرداد هزار و چهارصد و سه ساعت چهارده و سی دقیقه است.', lang='fa')]

Pure python. The order of the passes matters and is documented at `speakable`.
"""
from __future__ import annotations

import importlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

if not __package__:
    # standalone (persian_tts/scripts): siblings, en_phonemes at the repo root
    # and normalize_fa under model/v2 are plain modules on sys.path. Vendored
    # into a package, the same names are package siblings — see _sibling().
    _HERE = Path(__file__).resolve().parent
    for _p in (_HERE, _HERE.parent, _HERE.parent / "model" / "v2"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))


def _sibling(name: str):
    """A sibling module, imported the same way in both homes of this file."""
    if __package__:
        return importlib.import_module("." + name, __package__)
    return importlib.import_module(name)

#: Bump on any change that alters what is spoken — it is part of every audio
#: cache key downstream, so stale audio is never served after a fix.
FRONTEND_VERSION = "2026.11.01"

_FA_LETTERS = "؀-ۿ"
_HAS_FA = re.compile(f"[{_FA_LETTERS}]")
_HAS_LATIN = re.compile(r"[A-Za-z]")

# ══════════════════════════════════════════════════════════════════════════════
#  Characters
# ══════════════════════════════════════════════════════════════════════════════

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_FORMS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ـ": "", "ۀ": "هٔ",
                        " ": " ", " ": " ", " ": " "})
_INVISIBLE = re.compile(r"[​‍-‏‪-‮⁦-⁩﻿­]")


def clean_chars(text: str) -> str:
    """Unicode hygiene: one spelling per letter, ASCII digits, real spaces."""
    t = unicodedata.normalize("NFC", text or "")
    t = _INVISIBLE.sub("", t).translate(_FORMS).translate(_DIGITS)
    t = t.replace("٫", ".").replace("٬", ",")
    t = t.replace("−", "-").replace("–", "-").replace("—", " — ")
    t = re.sub(r"‌+", "‌", t)
    return t


# ══════════════════════════════════════════════════════════════════════════════
#  Numbers → words
# ══════════════════════════════════════════════════════════════════════════════

def fa_number(n: int) -> str:
    return _sibling("normalize_fa").number_to_words(int(n))


def fa_ordinal(n: int, *, suffix: str = "م", date: bool = False) -> str:
    """Persian ordinal: 2 → «دوم» (suffix «مین» → «دومین»); in dates 1 → «یکم»."""
    if n == 1 and not date:
        return "اول" if suffix == "م" else "اولین"
    w = fa_number(n)
    if w.endswith("سه"):
        w = w[:-2] + "سو"
    elif w.endswith("سی"):
        w += "‌ا"
    return w + suffix


_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]
_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth",
           "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def en_number(n: int) -> str:
    n = int(n)
    if n < 0:
        return "minus " + en_number(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        return _EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + _EN_ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _EN_ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + en_number(rest))
    for value, name in ((10**12, "trillion"), (10**9, "billion"),
                        (10**6, "million"), (10**3, "thousand")):
        if n >= value:
            rest = n % value
            return en_number(n // value) + " " + name + ("" if rest == 0 else " " + en_number(rest))
    return str(n)


def en_ordinal(n: int) -> str:
    words = en_number(n).split()
    last = words[-1]
    if last in _EN_ORD:
        words[-1] = _EN_ORD[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def en_year(n: int) -> str:
    """1100–2099 the way years are said: 1999 "nineteen ninety nine",
    2007 "two thousand seven", 2023 "twenty twenty three"."""
    if 2000 <= n <= 2009 or n % 1000 == 0:
        return en_number(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return en_number(hi) + " hundred"
    return en_number(hi) + " " + ("oh " + en_number(lo) if lo < 10 else en_number(lo))


def _digits_fa(s: str) -> str:
    return " ".join("صفر" if d == "0" else fa_number(int(d)) for d in s)


def _digits_en(s: str) -> str:
    return " ".join("oh" if d == "0" else _EN_ONES[int(d)] for d in s)


def number_words(whole: str, frac: str | None, lang: str) -> str:
    """A digit string (with optional fraction) as words. A leading zero or a
    very long run is a code/phone number, read digit by digit."""
    whole = whole.replace(",", "")
    if len(whole) > 12 or (whole.startswith("0") and len(whole) > 1):
        read = _digits_en if lang == "en" else _digits_fa
        out = read(whole)
        if frac:
            out += (" point " if lang == "en" else " ممیز ") + read(frac)
        return out
    n = int(whole or "0")
    if lang == "en":
        out = en_number(n)
        if frac:
            out += " point " + _digits_en(frac)
        return out
    out = fa_number(n)
    if frac:
        frac = frac.rstrip("0")
        if frac == "5":
            return out + " و نیم"
        if frac:
            unit = {1: "دهم", 2: "صدم", 3: "هزارم"}.get(len(frac))
            if unit:
                out += " ممیز " + fa_number(int(frac)) + " " + unit
            else:
                out += " ممیز " + _digits_fa(frac)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Abbreviations (before sentence splitting — their periods are not sentence ends)
# ══════════════════════════════════════════════════════════════════════════════

_ABBR_EN = [
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"), (r"\bvs\.", "versus"), (r"\bv\.s\.", "versus"),
    (r"\bDr\.", "doctor"), (r"\bMr\.", "mister"), (r"\bMrs\.", "missus"),
    (r"\bMs\.", "miz"), (r"\bProf\.", "professor"), (r"\bInc\.", "incorporated"),
    (r"\bLtd\.", "limited"), (r"\bCo\.", "company"), (r"\bapprox\.", "approximately"),
    (r"\bNo\.(?=\s*\d)", "number"), (r"\ba\.m\.", "A M"), (r"\bp\.m\.", "P M"),
    (r"\bU\.S\.A?\.?", "U S"), (r"\bU\.K\.", "U K"), (r"\bSt\.(?=\s+[A-Z])", "saint"),
]
_ABBR_FA = [
    (r"ق\.\s?م\.?", "قبل از میلاد"), (r"ب\.\s?م\.?", "بعد از میلاد"),
    (r"ه\.\s?ش\.?", "هجری شمسی"), (r"ه\.\s?ق\.?", "هجری قمری"),
    (r"\(ص\)", "صلی‌الله علیه و آله"), (r"\(ع\)", "علیه‌السلام"),
    (r"\(ره\)", "رحمت‌الله علیه"), (r"ج\.\s?ا\.\s?ا\.?", "جمهوری اسلامی ایران"),
]


def expand_abbreviations(text: str) -> str:
    for pat, rep in _ABBR_FA:
        text = re.sub(pat, rep, text)
    for pat, rep in _ABBR_EN:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Sentences
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Sentence:
    text: str
    lang: str          # "fa" (incl. Persian with Latin words) or "en"
    para_end: bool = False   # the last sentence before a blank line


# A period followed by whitespace ends a sentence unless it sits between two
# digits (handled: no whitespace there) — and a newline always does.
_SENT_END = re.compile(r"(?<=[.!?؟…])\s+|\s*\n+\s*")


def split_sentences(text: str) -> list[str]:
    out = []
    for s in _SENT_END.split(text.strip()):
        s = s.strip()
        if s and re.search(rf"[{_FA_LETTERS}A-Za-z0-9]", s):
            out.append(s)
    return out


def sentence_lang(s: str) -> str:
    """fa when the sentence has any Persian letter: Latin words inside Persian
    are read by the mixed G2P word by word, but digits and symbols there are
    read in Persian, as a Persian speaker would."""
    return "fa" if _HAS_FA.search(s) else "en"


# ══════════════════════════════════════════════════════════════════════════════
#  Per-sentence expansion
# ══════════════════════════════════════════════════════════════════════════════

_JALALI = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
           "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
_GREG_FA = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت",
            "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]
_GREG_EN = ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]
_MONTHS_FA = _JALALI + _GREG_FA

# unit → (Persian, English). Matched only right after a number.
_UNITS = {
    "km/h": ("کیلومتر بر ساعت", "kilometers per hour"),
    "m/s": ("متر بر ثانیه", "meters per second"),
    "kWh": ("کیلووات ساعت", "kilowatt hours"),
    "mAh": ("میلی‌آمپر ساعت", "milliamp hours"),
    "km": ("کیلومتر", "kilometers"), "cm": ("سانتی‌متر", "centimeters"),
    "mm": ("میلی‌متر", "millimeters"), "m": ("متر", "meters"),
    "kg": ("کیلوگرم", "kilograms"), "mg": ("میلی‌گرم", "milligrams"),
    "g": ("گرم", "grams"), "ml": ("میلی‌لیتر", "milliliters"),
    "mL": ("میلی‌لیتر", "milliliters"), "L": ("لیتر", "liters"),
    "TB": ("ترابایت", "terabytes"), "GB": ("گیگابایت", "gigabytes"),
    "MB": ("مگابایت", "megabytes"), "KB": ("کیلوبایت", "kilobytes"),
    "kB": ("کیلوبایت", "kilobytes"), "GHz": ("گیگاهرتز", "gigahertz"),
    "MHz": ("مگاهرتز", "megahertz"), "Hz": ("هرتز", "hertz"),
    "kW": ("کیلووات", "kilowatts"), "MW": ("مگاوات", "megawatts"),
    "W": ("وات", "watts"), "V": ("ولت", "volts"), "ms": ("میلی‌ثانیه", "milliseconds"),
    "px": ("پیکسل", "pixels"), "MP": ("مگاپیکسل", "megapixels"),
    "fps": ("فریم بر ثانیه", "frames per second"), "mph": ("مایل بر ساعت", "miles per hour"),
    "°C": ("درجهٔ سانتی‌گراد", "degrees Celsius"), "°F": ("درجهٔ فارنهایت", "degrees Fahrenheit"),
    "℃": ("درجهٔ سانتی‌گراد", "degrees Celsius"), "°": ("درجه", "degrees"),
}
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))

_CURRENCY = {"$": ("دلار", "dollars"), "€": ("یورو", "euros"), "£": ("پوند", "pounds"),
             "¥": ("ین", "yen"), "₺": ("لیر", "lira")}
_CURRENCY_CODES = {"USD": ("دلار", "dollars"), "EUR": ("یورو", "euros"),
                   "GBP": ("پوند", "pounds"), "IRR": ("ریال", "rials"),
                   "IRT": ("تومان", "tomans"), "AED": ("درهم", "dirhams")}
_SCALE = {"K": (1, "هزار", "thousand"), "k": (1, "هزار", "thousand"),
          "M": (2, "میلیون", "million"), "B": (3, "میلیارد", "billion"),
          "bn": (3, "میلیارد", "billion"), "T": (4, "تریلیون", "trillion")}

_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"


def _num(s: str, lang: str) -> str:
    whole, _, frac = s.partition(".")
    return number_words(whole, frac or None, lang)


def _date(y: int, m: int, d: int, lang: str) -> str | None:
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    if lang == "en":
        if not 1000 <= y <= 2100:
            return None
        return f"{_GREG_EN[m - 1]} {en_ordinal(d)}, {en_year(y)}"
    if 1200 <= y <= 1499:
        month = _JALALI[m - 1]
    elif 1500 <= y <= 2100:
        month = _GREG_FA[m - 1]
    else:
        return None
    return f"{fa_ordinal(d, date=True)} {month} {fa_number(y)}"


def expand(s: str, lang: str) -> str:
    """Every digit and symbol in one sentence → words in `lang`."""
    en = lang == "en"

    # ── dates ────────────────────────────────────────────────────────────────
    def ymd(m):
        r = _date(int(m[1]), int(m[2]), int(m[3]), lang)
        return f" {r} " if r else m[0]
    # (a sentence-final period after the date is punctuation, not a 4th field)
    s = re.sub(r"(?<![\d/.-])(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})(?![\d/-]|\.\d)", ymd, s)

    def dmy(m):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        d, mo = (a, b) if (a > 12 or not en) else (b, a)    # en: M/D/Y
        r = _date(y, mo, d, lang)
        return f" {r} " if r else m[0]
    s = re.sub(r"(?<![\d/.-])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?![\d/-]|\.\d)", dmy, s)

    if not en:
        # «۲۵ اسفند» → «بیست و پنجم اسفند»
        s = re.sub(rf"(?<![\d.,])(\d{{1,2}})\s+({'|'.join(_MONTHS_FA)})(?![{_FA_LETTERS}])",
                   lambda m: f"{fa_ordinal(int(m[1]), date=True)} {m[2]}"
                   if 1 <= int(m[1]) <= 31 else m[0], s)
    else:
        s = re.sub(rf"\b({'|'.join(_GREG_EN)})\s+(\d{{1,2}})\s?(?:st|nd|rd|th)?\b",
                   lambda m: f"{m[1]} {en_ordinal(int(m[2]))}"
                   if 1 <= int(m[2]) <= 31 else m[0], s)

    # ── clock times ─────────────────────────────────────────────────────────
    def clock(m):
        h, mi = int(m[1]), int(m[2])
        if h > 24 or mi > 59:
            return m[0]
        if en:
            return f" {en_number(h)} " + ("o'clock" if mi == 0 else
                                          ("oh " + en_number(mi) if mi < 10 else en_number(mi))) + " "
        return f" {fa_number(h)}" + (f" و {fa_number(mi)} دقیقه " if mi else " ")
    s = re.sub(r"(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])", clock, s)

    # ── versions: 3.5.2 ─────────────────────────────────────────────────────
    s = re.sub(r"(?<![\d.])(\d+(?:\.\d+){2,})(?![\d.])",
               lambda m: " " + (" point " if en else " نقطه ").join(
                   number_words(p, None, lang) for p in m[1].split(".")) + " ", s)

    # ── currency ────────────────────────────────────────────────────────────
    def money(amount: str, scale: str | None, cur: tuple[str, str]) -> str:
        words = _num(amount, lang)
        if scale:
            words += " " + (_SCALE[scale][2] if en else _SCALE[scale][1])
        return f" {words} {cur[1] if en else cur[0]} "
    sc = "|".join(sorted(_SCALE, key=len, reverse=True))
    s = re.sub(rf"([$€£¥₺])\s?({_NUM})\s?({sc})?\b(?![A-Za-z])",
               lambda m: money(m[2], m[3], _CURRENCY[m[1]]), s)
    s = re.sub(rf"({_NUM})\s?({sc})?\s?([$€£¥₺])",
               lambda m: money(m[1], m[2], _CURRENCY[m[3]]), s)
    codes = "|".join(_CURRENCY_CODES)
    s = re.sub(rf"\b({codes})\s?({_NUM})", lambda m: money(m[2], None, _CURRENCY_CODES[m[1]]), s)
    s = re.sub(rf"({_NUM})\s?({codes})\b", lambda m: money(m[1], None, _CURRENCY_CODES[m[2]]), s)
    # «20K کاربر», "1.5M users"
    s = re.sub(rf"(?<![\w.])({_NUM})\s?(K|M|B|bn)\b(?![A-Za-z])",
               lambda m: f" {_num(m[1], lang)} {_SCALE[m[2]][2] if en else _SCALE[m[2]][1]} ", s)

    # ── percent ─────────────────────────────────────────────────────────────
    s = re.sub(rf"[%٪]\s?({_NUM})", lambda m: f" {'percent' if en else 'درصد'} {m[1]} ", s)  # ٪۵
    s = re.sub(rf"({_NUM})\s?[%٪]", lambda m: f" {m[1]} {'percent' if en else 'درصد'} ", s)

    # ── units (only right after a number) ───────────────────────────────────
    s = re.sub(rf"({_NUM})\s?({_UNIT_RE})(?![A-Za-z{_FA_LETTERS}])",
               lambda m: f" {m[1]} {_UNITS[m[2]][1] if en else _UNITS[m[2]][0]} ", s)

    # ── ranges and signs ────────────────────────────────────────────────────
    s = re.sub(rf"(?<![\w.-])({_NUM})\s?-\s?({_NUM})(?![\w.])",
               lambda m: f" {m[1]} {'to' if en else 'تا'} {m[2]} ", s)
    s = re.sub(rf"(?<![\w.)])-({_NUM})", lambda m: f" {'minus' if en else 'منفی'} {m[1]}", s)
    s = re.sub(rf"(?<![\w.)])\+({_NUM})(?!\d)", lambda m: f" {'plus' if en else 'مثبت'} {m[1]}", s)

    # ── ordinals ────────────────────────────────────────────────────────────
    if en:
        s = re.sub(r"\b(\d+)\s?(st|nd|rd|th)\b", lambda m: en_ordinal(int(m[1])), s)
    else:
        s = re.sub(rf"(\d+)\s?‌?(امین|مین|ام)(?![{_FA_LETTERS}])",
                   lambda m: fa_ordinal(int(m[1]), suffix="مین" if m[2].endswith("ین") else "م"), s)

    # ── symbols ─────────────────────────────────────────────────────────────
    sym = ({"&": " and ", "@": " at ", "=": " equals ", "×": " times ", "÷": " divided by ",
            "≈": " approximately ", "→": " to ", "#": " number "} if en else
           {"&": " و ", "@": " اَت ", "=": " مساوی ", "×": " ضرب در ", "÷": " تقسیم بر ",
            "≈": " تقریباً ", "→": " به ", "#": " شمارهٔ "})
    for k, v in sym.items():
        s = s.replace(k, v)
    s = re.sub(r"(?<=\s)\+(?=\s)", " plus " if en else " به‌علاوه ", s)
    s = re.sub(r"و\s*/\s*یا", "و یا", s)                                  # و/یا
    s = re.sub(rf"(?<=[{_FA_LETTERS}])/(?=[{_FA_LETTERS}])", " یا ", s)       # الف/ب

    # ── remaining numbers ───────────────────────────────────────────────────
    if en:
        def en_num(m):
            raw = m[0]
            if "," not in raw and "." not in raw and len(raw) == 4 and 1100 <= int(raw) <= 2099:
                return f" {en_year(int(raw))} "
            return f" {_num(raw, 'en')} "
        s = re.sub(_NUM, en_num, s)
    else:
        s = re.sub(_NUM, lambda m: f" {_num(m[0], 'fa')} ", s)
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Latin tokens
# ══════════════════════════════════════════════════════════════════════════════

_CAMEL_1 = re.compile(r"(?<=[a-z])(?=[A-Z])")             # ChatGPT → Chat GPT
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")        # GPTModel → GPT Model
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def split_latin(s: str) -> str:
    """Make Latin tokens readable word by word: camelCase and letter-digit
    joints become spaces, hyphen/underscore joints too, and a dotted file
    name reads its dot ("config.json" → "config dot json")."""
    def tok(m):
        w = m[0]
        w = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z]{2,4}\b)", " dot ", w)
        w = w.replace("_", " ")
        w = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", " ", w)
        w = _CAMEL_1.sub(" ", w)
        w = _CAMEL_2.sub(" ", w)
        w = _LETTER_DIGIT.sub(" ", w)
        return w
    return re.sub(r"[A-Za-z][A-Za-z0-9_.\-]*[A-Za-z0-9]|[A-Za-z]", tok, s)


# ══════════════════════════════════════════════════════════════════════════════
#  The whole front-end
# ══════════════════════════════════════════════════════════════════════════════

def _tidy(s: str, lang: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([،,.!?؟:;؛])", r"\1", s)
    s = re.sub(r"([،,])\s*([.!?؟])", r"\2", s)
    s = s.strip()
    if s and s[-1] not in ".!?؟…:":
        s += "."
    return s


def speakable(text: str) -> list[Sentence]:
    """Written text → sentences of words, each tagged with its language.

    Order: characters → abbreviations (their periods are not sentence ends)
    → sentence split (newline is a sentence end) → per sentence: language,
    digits/symbols in that language, Latin token splitting.
    """
    t = expand_abbreviations(clean_chars(text))
    out = []
    for para in re.split(r"\n[ \t]*\n", t):
        for raw in split_sentences(para):
            lang = sentence_lang(raw)
            # Latin tokens first: "GPT-4" must become "GPT 4" before digits are
            # read, or the hyphen survives as a dangling "GPT- چهار".
            s = split_latin(raw)
            s = expand(s, lang)
            s = _tidy(s, lang)
            if re.search(rf"[{_FA_LETTERS}A-Za-z]", s):
                out.append(Sentence(s, lang))
        if out:
            out[-1].para_end = True
    if out:
        out[-1].para_end = False          # nothing follows the last one
    return out


def speakable_text(text: str) -> str:
    return " ".join(s.text for s in speakable(text))


# ══════════════════════════════════════════════════════════════════════════════
#  Mixed-script G2P
# ══════════════════════════════════════════════════════════════════════════════

# Persian never sets two vowels side by side inside a word: a suffix after the
# silent-h «e» starts with a glottal stop («داده‌اند» dAde?and, «خانه‌ام»
# xAne?am, «نامه‌ای» nAme?i). The G2P writes it only sometimes — «شده‌اند»
# alone is «Sode ?and», inside a sentence «Sodeand». Inserted (2026-10-24,
# eval/fluency.py, 7 cases × 5 seeds): UTMOS +0.047 (95 % CI +0.010..+0.084),
# CER unchanged; «برنامه‌اش» stopped coming out as «برنامهش». Written without
# a space: «dAde ?and» split the verb (heard «شده. انکه») for no CER gain.
# Before «i» only a word-final suffix counts (-ei, -eim, -eid): «xeili» is
# the G2P's spelling of the diphthong in «خیلی», not two vowels.
_HIATUS = re.compile(r"(?<=[^\s?])e(?=[aAou]|i[md]?(?:\s|$))")


def hiatus(ph: str) -> str:
    """«dAdeand» -> «dAde?and». Persian segments only: English phonemes come
    from en_phonemes and never pass through here."""
    return _HIATUS.sub("e?", ph)


class MixedG2P:
    """OnnxG2P + English, with the same `phonemise` signature, so the engine's
    punctuation-aware phrase planning is reused unchanged. English words that
    en_phonemes can resolve are phonemised directly (keeping the vowels Persian
    script would drop); everything else goes to the Persian G2P.

    ONE copy, shared by speak.py, the demo server and the chatbot runtime: the
    chatbot's own copy once treated a lone «،» left between two English words
    as text — and the G2P raises on punctuation-only input ("normalisation
    emptied the text"), failing the whole chunk."""

    def __init__(self, inner):
        self.inner = inner

    def phonemise(self, text, keep_ezafe=False):
        out = []
        for kind, seg in _sibling("en_phonemes").split(text):
            if kind == "en":
                out.append(seg)
            elif any(ch.isalnum() for ch in seg):     # punctuation debris is not text
                ph = self.inner.phonemise(seg, keep_ezafe=keep_ezafe)
                if ph:
                    out.append(hiatus(ph))
        return " ".join(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Rendering
# ══════════════════════════════════════════════════════════════════════════════

def plan(engine, text: str, *, pack: bool = False):
    """→ (sentences, per-sentence phrase plans) for `engine.synthesize_many`."""
    tts_onnx = _sibling("tts_onnx")
    pack_phrases, plan_phrases = tts_onnx.pack_phrases, tts_onnx.plan_phrases

    sents, plans = [], []
    for s in speakable(text):
        p = plan_phrases(s.text, engine._g2p, engine.sp)
        if pack:
            p = pack_phrases(p)
        if p:
            sents.append(s)
            plans.append(p)
    return sents, plans


def join(engine, sents, audios, pace: float = 1.0, *, para_start: bool = True,
         para_end: bool = True):
    """Per-sentence audio -> one answer: the sentence pause between sentences,
    the paragraph pause after a paragraph's last one. ONE copy for every
    caller (speak.py, the demo server, the chatbot runtime), so the pauses
    cannot drift apart between them.

    With `engine.intonation` > 0 each sentence also gets the prosody of its
    written type (scripts/intonation.py). Version 2 adds a loudness plan over
    the text, which needs to know where paragraphs start and end: blank lines
    inside the text say so, and `para_start` / `para_end` say whether this
    text's first sentence opens a paragraph and its last one closes it (True
    for a whole text; a caller speaking a text in pieces passes what it knows)."""
    import numpy as np

    tts_onnx = _sibling("tts_onnx")
    strength = float(getattr(engine, "intonation", 0.0) or 0.0)
    into = _sibling("intonation") if strength > 0 else None
    v2 = into is not None and getattr(into, "VERSION", 1) >= 2
    ceiling = getattr(engine, "limiter_ceiling", None)
    n = len(sents)
    parts = []
    opens = para_start                 # does the next sentence open a paragraph?
    for i, (s, a) in enumerate(zip(sents, audios)):
        first, opens = opens, bool(s.para_end)
        if not len(a):
            continue
        if parts:
            parts.append(engine.pause(gap / max(pace, 0.1)))
        a = np.asarray(a, dtype=np.float32)
        if v2:
            # paragraph place only means something in a text of several sentences
            closes = bool(s.para_end) or (i == n - 1 and para_end)
            multi = n > 1
            a = into.process(a, engine.sample_rate, s.text, strength,
                             para_first=multi and first and not closes,
                             para_last=multi and closes and not first,
                             peak_guard=ceiling is None)
        elif into is not None:
            # the written mark is still on the sentence; the audio is not
            kind = into.sentence_kind(s.text)
            if kind in into.MODIFIED:
                a = into.apply(a, engine.sample_rate, kind, strength, s.text)
        parts.append(a)
        trace = getattr(engine, "join_trace", None)
        if trace is not None:            # research: per-sentence audio as joined
            trace.append((s.text, a))
        gap = tts_onnx.PARAGRAPH_GAP if s.para_end else tts_onnx.SENTENCE_GAP
    out = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    if ceiling:
        # a louder level target only works if peaks are limited, not whole
        # sentences scaled down (see intonation.limit)
        out = _sibling("intonation").limit(out, engine.sample_rate, float(ceiling))
    return out


def styles_for(engine, sents) -> list | None:
    """Per sentence, the prosody-styled prompt its written type asks for
    (intonation.STYLE_OF_KIND; e.g. an exclamation is generated from a livelier
    prompt), or None when the engine has no styles / intonation is off."""
    if not (float(getattr(engine, "intonation", 0.0) or 0.0) > 0 and getattr(engine, "prompt_styles", None)):
        return None
    into = _sibling("intonation")
    table = getattr(into, "STYLE_OF_KIND", {})
    out = [table.get(into.sentence_plan(s.text)[0]) if s.lang != "en" else None for s in sents]
    return out if any(out) else None


def render(engine, text: str, voice_wav: str, *, seed: int | None = None,
           pace: float = 1.0, pack: bool = False, para_start: bool = True,
           para_end: bool = True):
    """Text → (float32 audio, phoneme summary). Every sub-chunk of every
    sentence is generated in ONE parallel batch (synthesize_many), so a
    multi-sentence chunk no longer waits sentence by sentence. `para_start` /
    `para_end`: see join (a caller speaking one text in pieces passes them)."""
    import numpy as np

    if seed is not None:
        engine.rng = np.random.default_rng(seed)
    sents, plans = plan(engine, text, pack=pack)
    phon = " | ".join(" ".join(p for p, _ in pl) for pl in plans)
    if not plans:
        return np.zeros(0, np.float32), phon
    kw = {}
    styles = styles_for(engine, sents)
    if styles:
        kw["styles"] = styles
    audios = engine.synthesize_many(plans, voice_wav, pace=pace,
                                    langs=[s.lang for s in sents], **kw)
    return join(engine, sents, audios, pace, para_start=para_start, para_end=para_end), phon


if __name__ == "__main__":
    for line in sys.argv[1:] or [
        "جلسه در تاریخ ۱۴۰۳/۰۵/۲۰ ساعت ۱۴:۳۰ برگزار شد.",
        "این سرویس ماهانه $20 هزینه دارد و ۸ GB رم دارد.",
        "The company grew by 25% in 2023 and now has 1,200 employees.",
        "مدل GPT-4 و ChatGPT در سال 2023 منتشر شدند.",
    ]:
        for s in speakable(line):
            print(f"[{s.lang}] {s.text}")
