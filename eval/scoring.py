"""Transcript scoring: fold away spelling variation, keep pronunciation errors.

Whisper writes the same spoken words many ways — «۱۴۰۲» / «1402» / «هزار و
چهارصد و دو», «می‌شود» / «میشود» / «می شود», «Python» / «پایتون», Arabic ي/ك
for Persian ی/ک. None of that is a TTS error, so both the reference and the
transcript are folded onto one canonical form before comparison:

  * Arabic letter forms → Persian, diacritics and tatweel dropped, and the
    letters that spell one Persian sound folded together (ط→ت, ص/ث→س, …)
  * every digit run → words (Persian words for fa/mixed, English for en)
  * Latin words → Persian script (mixed only), spelled from the engine's own
    English pronunciation, so «اپل» and "Apple" meet in the middle
  * punctuation, ZWNJ and SPACES removed — word segmentation in Persian is a
    spelling convention, not something a listener hears

The primary metric is CER on that space-free string. WER (on the folded but
space-kept string) is reported alongside for readability.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                  # en_phonemes
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model" / "v2"))

_FOLD = str.maketrans({
    "ي": "ی", "ى": "ی", "ۍ": "ی", "ك": "ک", "ڪ": "ک", "ة": "ه", "ۀ": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا", "ؤ": "و", "ئ": "ی", "ء": "", "ـ": "",
    "آ": "ا",
    # letters that spell the same Persian sound: a listener cannot hear «باطری»
    # from «باتری», and Whisper picks either when it spells a word it misheard
    "ط": "ت", "ص": "س", "ث": "س", "ذ": "ز", "ض": "ز", "ظ": "ز", "ح": "ه",
    "غ": "ق", "ع": "ا",
})
_DIACRITICS = re.compile(r"[ً-ٰٕۖ-ۭ]")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# ── numbers → words ─────────────────────────────────────────────────────────
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]


def en_number(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        return _EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + _EN_ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _EN_ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + en_number(rest))
    for value, name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        if n >= value:
            rest = n % value
            return en_number(n // value) + " " + name + ("" if rest == 0 else " " + en_number(rest))
    return str(n)


def _en_year(n: int) -> str:
    """1900–2099 the way they are said: "twenty twenty three"."""
    if 2000 <= n <= 2009:
        return en_number(n)
    hi, lo = divmod(n, 100)
    return en_number(hi) + " " + ("hundred" if lo == 0 else
                                  ("oh " + en_number(lo) if lo < 10 else en_number(lo)))


def _fa_numbers(text: str) -> str:
    from normalize_fa import numbers_to_words

    text = re.sub(r"(\d)[,٬](?=\d{3}\b)", r"\1", text)
    # Whisper writes a spoken clock time as «14.30» or «14:30»; the reader said
    # «چهارده و سی دقیقه». Fold both sides onto the spoken form.
    text = re.sub(r"(ساعت\s*)(\d{1,2})[:.](\d{2})\b",
                  lambda m: f"{m[1]}{int(m[2])} و {int(m[3])} دقیقه" if int(m[3]) else f"{m[1]}{int(m[2])}",
                  text)
    # «۴۲.۵» and «۴۲ و نیم» are the same reading; fold both to the latter
    text = re.sub(r"(\d+)\.5\b", r"\1 و نیم", text)
    text = re.sub(r"(\d+)\s*[%٪]", r"\1 درصد", text)
    return numbers_to_words(text)


def _en_numbers(text: str) -> str:
    text = re.sub(r"(\d),(?=\d{3}\b)", r"\1", text)
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)

    def one(m: re.Match) -> str:
        n = int(m.group(0))
        if 1900 <= n <= 2099 and len(m.group(0)) == 4:
            return " " + _en_year(n) + " "
        return " " + en_number(n) + " "
    return re.sub(r"\d+", one, text)


# ── canonical forms ─────────────────────────────────────────────────────────
def _translit(text: str) -> str:
    from g2p_onnx import transliterate_text

    return transliterate_text(text)


# phoneme → the letter Persian writes for it; the short vowels a/e are not
# written, as in «اپل», «داکر», «مشین لرنینگ»
_PH_FA = {"b": "ب", "p": "پ", "t": "ت", "d": "د", "k": "ک", "g": "گ", "q": "ق",
          "C": "چ", "j": "ج", "S": "ش", ";": "ژ", "x": "خ", "h": "ه", "z": "ز",
          "s": "س", "f": "ف", "v": "و", "y": "ی", "r": "ر", "l": "ل", "m": "م",
          "n": "ن", "A": "ا", "i": "ی", "o": "و", "u": "و"}


def _ph_to_fa(ph: str) -> str:
    out = []
    for w in ph.split():
        lead = ""
        if w.startswith("?"):                 # word-initial vowel: «ا», «ای», «او»
            w = w[1:]
            lead = {"i": "ای", "o": "او", "u": "او"}.get(w[:1], "ا")
            if w[:1] in "aAeiou":
                w = w[1:]
        out.append(lead + "".join(_PH_FA.get(c, "") for c in w))
    return " ".join(out)


def _latin_to_fa(text: str) -> str:
    """English words → Persian script spelled from their PRONUNCIATION (the
    engine's own en_phonemes), so a correct reading of "Apple" matches the
    «اپل» Whisper writes. Letter-by-letter transliteration gave «اپپل»,
    «ی فاون» for iPhone and «ینستالل» for install, and scored every correctly
    spoken English word in a Persian sentence as an error. Words en_phonemes
    cannot resolve keep the letter transliteration."""
    from en_phonemes import word

    def one(m: re.Match) -> str:
        ph = word(m.group(0))
        return _ph_to_fa(ph) if ph else m.group(0)
    return _translit(re.sub(r"[A-Za-z][A-Za-z']*", one, text))


def fold(text: str, lang: str) -> str:
    """Canonical words (space-separated) for scoring."""
    t = unicodedata.normalize("NFC", text or "")
    t = t.translate(_DIGITS)
    if lang == "en":
        t = _en_numbers(t.lower())
        t = re.sub(r"[^a-z\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()
    t = _DIACRITICS.sub("", t).translate(_FOLD)
    t = _fa_numbers(t)
    if lang == "mix":
        t = re.sub(r"(?<=[A-Za-z])-(?=\d)", " ", t)
        # "ChatGPT" is said «چت جی‌پی‌تی»: split camelCase before
        # transliterating, or the correct reading scores as an error
        from frontend import split_latin
        t = _latin_to_fa(split_latin(t))
    t = t.replace("\u200c", " ")
    # A spoken ezafe after a silent ه is written «نویسنده‌ی» / «نویسنده ای» /
    # «نویسندهٔ» / «نویسنده» depending on the writer — one pronunciation.
    t = re.sub(r"(?<=ه)\s*(?:ای|ی)(?=\s|$)", "", t)
    # Persian punctuation sits INSIDE the Arabic block the next line keeps: a
    # transcript's «،» or «؟» (a pause Whisper heard) used to count as an error
    t = re.sub(r"[،؛؟٪٫٬۔]", " ", t)
    t = re.sub(r"[^؀-ۿA-Za-z\s]", " ", t)
    t = _DIACRITICS.sub("", t).translate(_FOLD)
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def _edits(a, b) -> int:
    """Levenshtein distance over sequences (chars or words)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(hyp: str, ref: str, lang: str) -> dict:
    """CER (space-free) and WER of `hyp` against `ref`, both folded."""
    h, r = fold(hyp, lang), fold(ref, lang)
    hc, rc = h.replace(" ", ""), r.replace(" ", "")
    cer = _edits(hc, rc) / max(1, len(rc))
    wer = _edits(h.split(), r.split()) / max(1, len(r.split()))
    return {"cer": round(min(cer, 1.5), 4), "wer": round(min(wer, 1.5), 4),
            "hyp_fold": h, "ref_fold": r}
