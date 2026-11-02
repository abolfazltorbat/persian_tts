"""English words -> the pocket-tts Persian phoneme alphabet, offline.

The TTS model only knows Persian phonemes (a A e i o u, b C d f g h j k l m
n p q r s S t v x y z ? ;), so English comes out Persian-accented by design —
there is no /th/, /w/ or /ng/ to render. What this avoids is the engine's
default path, which spells an English word in Persian letters and asks the
G2P to read it back: Persian script leaves short vowels unwritten, so
"hello" -> "هلو" -> "holu". Going straight to phonemes keeps the vowels.

Choices that follow the way Persian speakers actually say English (and that
were checked on the Whisper round trip, eval/harness.py):

* /θ/ -> t, /ð/ -> d ("three" = tri, "the" = de) — the convention of Persian
  loanwords (تئوری، تریلر، آتن). Mapping /θ/ to s made "three" ?esri.
* a word NOT in CMUdict is not handed straight to Persian transliteration:
  an inflection of a known word ("embeddings", "tokenizer") or a compound of
  two ("frontend", "dataset") is assembled from its parts first, and a short
  table covers the tech names an assistant actually says.
"""
import re

_ARPA = {
    "AA": "A", "AE": "a", "AO": "o", "AW": "Av", "AY": "Ay", "EH": "e",
    "ER": "er", "EY": "ey", "IH": "i", "IY": "i", "OW": "o", "OY": "oy",
    "UH": "u", "UW": "u",
    "B": "b", "CH": "C", "D": "d", "DH": "d", "F": "f", "G": "g", "HH": "h",
    "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng", "P": "p",
    "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "t", "V": "v", "W": "v",
    "Y": "y", "Z": "z", "ZH": ";",
}
# spoken letter names, Persian convention (GPU -> ji pi yu)
_LETTERS = {
    "a": "?ey", "b": "bi", "c": "si", "d": "di", "e": "?i", "f": "?ef",
    "g": "ji", "h": "?eyC", "i": "?Ay", "j": "jey", "k": "key", "l": "?el",
    "m": "?em", "n": "?en", "o": "?o", "p": "pi", "q": "kiyu", "r": "?Ar",
    "s": "?es", "t": "ti", "u": "yu", "v": "vi", "w": "dablyu",
    "x": "?eyks", "y": "vAy", "z": "zed",
}
# words CMUdict pronounces in a way Persian speakers do not use, or does not
# have at all — the names an assistant says most
_OVERRIDE = {
    "python": "pAyton", "windows": "vindoz", "linux": "linuks",
    "google": "gugel", "server": "server", "email": "?imeyl",
    "video": "vidiyo", "audio": "?odiyo", "menu": "meno",
    "chat": "Cat", "bot": "bot", "api": "?ey pi ?Ay",
    "json": "jeyson", "anthropic": "?antropik", "claude": "kelod",
    "kubernetes": "kubernetis", "openai": "?open ?ey ?Ay", "github": "git hAb",
    "gitlab": "git lab", "ubuntu": "?ubuntu", "nvidia": "?envidiyA",
    "pytorch": "pAy torC", "tensorflow": "tensor flo", "numpy": "nAm pAy",
    "javascript": "jAvA ?eskript", "typescript": "tAyp ?eskript",
    "django": "jango", "vue": "viyu", "wifi": "vAy fAy", "whatsapp": "vAts ?ap",
    "tiktok": "tik tAk", "youtube": "yu tiyub", "instagram": "?instAgram",
    "telegram": "telegrAm", "spotify": "?espAtifAy", "iphone": "?Ay fon",
    "ipad": "?Ay pad", "macos": "mak ?o ?es", "ios": "?Ay ?o ?es",
    "android": "?andeoyd", "llama": "lAmA", "gemini": "jeminAy",
    "deepseek": "dip sik", "qwen": "kiven", "mistral": "mistrAl",
    "huggingface": "hAging feys", "ai": "?ey ?Ay", "ok": "?okey",
    "okay": "?okey", "online": "?AnlAyn", "offline": "?AflAyn",
    "dana": "dAnA", "sql": "?es kiyu ?el", "html": "?eyC ti ?em ?el",
    "css": "si ?es ?es", "url": "yu ?Ar ?el", "pdf": "pi di ?ef",
    "gpu": "ji pi yu", "cpu": "si pi yu", "ram": "ram", "usb": "yu ?es bi",
    "sms": "?es ?em ?es",
    # acronyms said as words, not spelled (ناسا، ناتو، اوپک، کووید، یونسکو)
    "nasa": "nAsA", "nato": "nAto", "opec": "?opek", "covid": "kovid",
    "unesco": "yunesko", "unicef": "yunisef",
    # the article in its strong form. CMUdict's schwa came out as a lone
    # «?e», which Whisper heard as "the" in 3 of 3 takes ("with the chance")
    "a": "?ey",
}
_VOWELS = set("aAeiou")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")

# Inflectional endings tried on a word CMUdict lacks: (ending, phonemes added
# after the stem, stem spellings to try).
_SUFFIXES = (
    ("ization", "Ayzeysen", ("ize", "")), ("izer", "Ayzer", ("ize", "")),
    ("izers", "Ayzerz", ("ize", "")), ("ings", "ingz", ("", "e")),
    ("ing", "ing", ("", "e")), ("ers", "erz", ("", "e")), ("er", "er", ("", "e")),
    ("ed", "d", ("", "e")), ("ly", "li", ("",)), ("ness", "nes", ("",)),
    ("ment", "ment", ("",)), ("able", "Abel", ("", "e")), ("es", "iz", ("",)),
    ("s", "s", ("",)),
)

_PREFIXES = (("re", "ri"), ("pre", "pri"), ("un", "?an"), ("non", "nAn"),
             ("multi", "malti"), ("sub", "sab"), ("super", "super"),
             ("inter", "?inter"), ("over", "?over"), ("under", "?ander"),
             ("auto", "?Ato"), ("co", "ko"), ("de", "di"), ("micro", "mAykro"))

_VOICELESS = set("ptkfsSC")


def _voiced_ending(end: str, ph_end: str, stem_ph: str) -> str:
    """-s / -ed follow the stem's last sound, as in English: cats/dogs,
    stopped/played/wanted."""
    last = stem_ph[-1:]
    if end == "s":
        return "s" if last in _VOICELESS else "z"
    if end == "ed":
        if last in ("t", "d"):
            return "id"
        return "t" if last in _VOICELESS else "d"
    return ph_end


_dict = None


def _lookup(word):
    """CMUdict is OPTIONAL: without it every word reads as None and the caller
    falls back to the engine's Latin->Persian transliteration — a Persian-
    accented reading rather than a failed synthesis."""
    global _dict
    if _dict is None:
        try:
            import cmudict
            _dict = cmudict.dict()
        except Exception:
            _dict = {}
    return _dict.get(word)


def _arpa_to_ph(pron) -> str:
    out = []
    for p in pron:
        sym, stress = p.rstrip("012"), p[-1] if p[-1].isdigit() else ""
        if sym == "AH":
            out.append("e" if stress == "0" else "a")   # schwa vs strut
        else:
            out.append(_ARPA.get(sym, ""))
    return "".join(out).replace("ngg", "ng")


def _onset(s: str) -> str:
    """Persian phonotactics: no vowel-initial and no s+consonant onset."""
    if not s:
        return s
    if s[0] in _VOWELS:
        return "?" + s                                   # Persian onset rule
    if s[0] == "s" and len(s) > 1 and s[1] not in _VOWELS:
        return "?e" + s                                  # state -> ?esteyt
    return s


def _dict_ph(lw: str):
    pron = _lookup(lw)
    return _arpa_to_ph(pron[0]) if pron else None


def _oov(lw: str):
    """A word CMUdict lacks, assembled from parts it has."""
    for end, ph_end, stems in _SUFFIXES:
        if lw.endswith(end) and len(lw) - len(end) >= 3:
            base = lw[: -len(end)]
            for extra in stems:
                stem = base + extra
                ph = _OVERRIDE.get(stem) or _dict_ph(stem)
                if ph:
                    return ph + _voiced_ending(end, ph_end, ph)
            # doubled consonant before -ing/-ed/-er ("running", "stopped")
            if end in ("ing", "ed", "er", "ers") and len(base) >= 4 and base[-1] == base[-2]:
                ph = _dict_ph(base[:-1])
                if ph:
                    return ph + _voiced_ending(end, ph_end, ph)
    for pre, ph_pre in _PREFIXES:
        if lw.startswith(pre) and len(lw) - len(pre) >= 4:
            rest = lw[len(pre):]
            ph = _OVERRIDE.get(rest) or _dict_ph(rest) or _oov(rest)
            if ph:
                return ph_pre + " " + _onset(ph)
    if len(lw) >= 6:
        for i in range(3, len(lw) - 2):
            a, b = _OVERRIDE.get(lw[:i]) or _dict_ph(lw[:i]), _OVERRIDE.get(lw[i:]) or _dict_ph(lw[i:])
            if a and b and len(lw[:i]) >= 3 and len(lw[i:]) >= 3:
                return a + " " + _onset(b)
    return None


def spell(letters: str) -> str:
    """An acronym letter by letter, one word per letter: LLM -> «?el ?el ?em»,
    as a Persian speaker says «ال ال ام». Glued into one word («?el?el?em») the
    tokenizer cuts it on a mid-word «?» the model seldom saw, and LLM came out
    as «عللم» / «LNM»."""
    return " ".join(_LETTERS[c] for c in letters.lower())


def word(w):
    """One English word -> phonemes, or None if it cannot be resolved
    (caller should then fall back to the engine's own transliteration)."""
    lw = w.lower().strip("'")
    if lw in _OVERRIDE:
        return _OVERRIDE[lw]
    if w.isupper() and 2 <= len(w) <= 5 and w.isalpha():
        return spell(w)                                   # GPT -> ji pi ti
    if len(w) >= 3 and w.endswith("s") and w[:-1].isupper() and w[:-1].isalpha():
        # a plural acronym (LLMs, APIs, GPUs): the -s goes on the last letter
        base = _OVERRIDE.get(lw[:-1]) or (spell(w[:-1]) if len(w) <= 6 else None)
        if base:
            return base + ("iz" if base[-1] in "szSC;j" else _voiced_ending("s", "s", base))
    s = _dict_ph(lw)
    if not s:
        s = _oov(lw)
    if not s:
        return None
    return " ".join(_onset(part) for part in s.split())


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
