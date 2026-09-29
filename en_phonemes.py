"""English words -> the pocket-tts Persian phoneme alphabet, offline.

The TTS model only knows Persian phonemes (a A e i o u, b C d f g h j k l m
n p q r s S t v x y z ?), so English comes out Persian-accented by design —
there is no /th/, /w/ or /ng/ to render. What this avoids is the engine's
default path, which spells an English word in Persian letters and asks the
G2P to read it back: Persian script leaves short vowels unwritten, so
"hello" -> "هلو" -> "holu". Going straight to phonemes keeps the vowels.
"""
import re

_ARPA = {
    "AA": "A", "AE": "a", "AO": "o", "AW": "Av", "AY": "Ay", "EH": "e",
    "ER": "er", "EY": "ey", "IH": "i", "IY": "i", "OW": "o", "OY": "oy",
    "UH": "u", "UW": "u",
    "B": "b", "CH": "C", "D": "d", "DH": "d", "F": "f", "G": "g", "HH": "h",
    "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng", "P": "p",
    "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "s", "V": "v", "W": "v",
    "Y": "y", "Z": "z", "ZH": "j",
}
# spoken letter names, Persian convention (GPU -> jipiyu)
_LETTERS = {
    "a": "?ey", "b": "bi", "c": "si", "d": "di", "e": "?i", "f": "?ef",
    "g": "ji", "h": "?eyC", "i": "?Ay", "j": "jey", "k": "key", "l": "?el",
    "m": "?em", "n": "?en", "o": "?o", "p": "pi", "q": "kiyu", "r": "?Ar",
    "s": "?es", "t": "ti", "u": "yu", "v": "vi", "w": "dablyu",
    "x": "?eyks", "y": "vAy", "z": "zed",
}
# words CMUdict pronounces in a way Persian speakers do not use
_OVERRIDE = {
    "python": "pAyton", "windows": "vindoz", "linux": "linuks",
    "google": "gugel", "server": "server", "email": "?imeyl",
    "video": "vidiyo", "audio": "?odiyo", "menu": "meno",
    "chat": "Cat", "bot": "bot", "api": "?ey pi ?Ay",
}
_VOWELS = set("aAeiou")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")

_dict = None


def _lookup(word):
    global _dict
    if _dict is None:
        import cmudict
        _dict = cmudict.dict()
    return _dict.get(word)


def word(w):
    """One English word -> phonemes, or None if it cannot be resolved
    (caller should then fall back to the engine's own transliteration)."""
    lw = w.lower().strip("'")
    if lw in _OVERRIDE:
        return _OVERRIDE[lw]
    if w.isupper() and 2 <= len(w) <= 5 and w.isalpha():
        return "".join(_LETTERS[c] for c in lw)      # GPT -> jipiti
    pron = _lookup(lw)
    if not pron:
        return None
    out = []
    for p in pron[0]:
        sym, stress = p.rstrip("012"), p[-1] if p[-1].isdigit() else ""
        if sym == "AH":
            out.append("e" if stress == "0" else "a")   # schwa vs strut
        else:
            out.append(_ARPA.get(sym, ""))
    s = "".join(out).replace("ngg", "ng")
    if not s:
        return None
    if s[0] in _VOWELS:
        s = "?" + s                                  # Persian onset rule
    elif s[0] == "s" and len(s) > 1 and s[1] not in _VOWELS:
        s = "?e" + s                                 # state -> ?esteyt
    return s


def split(text):
    """[('en', phonemes) | ('fa', text), ...] in original order. Latin words
    that cannot be resolved stay inside the 'fa' text, so the engine's own
    transliteration still handles names and jargon."""
    parts, last = [], 0
    for m in _WORD.finditer(text):
        ph = word(m.group(0))
        if ph is None:
            continue
        parts.append(("fa", text[last:m.start()]))
        parts.append(("en", ph))
        last = m.end()
    parts.append(("fa", text[last:]))
    return parts
