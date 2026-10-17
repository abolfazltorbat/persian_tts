"""ASR self-check for generated speech: did this take say what it was asked to?

The acoustic model is stochastic and fails in ways no energy/duration check
can see: it sometimes replays the REFERENCE VOICE's own sentence instead of the
text (female_narration's clip says «خوب، حالا فهمه و آزاده کجا رفتن؟», and that
phrase surfaced at the start of 1 in ~10 baseline renders), drops a word, or
says a different one. Those takes are loud, the right length, and wrong. So
each generated unit is transcribed and compared with its target; a bad take is
regenerated (tts_onnx._generate_with_retry), the best of the attempts kept.

WHAT IS COMPARED. The unit's target exists only as phonemes, the transcript
only as script — Persian script leaves short vowels unwritten, so the honest
common ground is the CONSONANT SKELETON: every consonant mapped to a sound
class (ت/ط → t, س/ص/ث → s, ز/ذ/ض/ظ → z, ق/غ → q, ح/ه → h …), vowels, glottal
stops, و/ی/ا (vowel-or-consonant letters) and a word-final silent ه dropped on
both sides. "monta;eret" vs «منتظرت» then differ in exactly the consonant the
G2P got wrong, while «می‌شود» / «میشود» / «می شود» all agree. English (Latin
transcript) goes through the same CMUdict mapping the engine used to speak it.

The score is a character error rate on skeletons, so it is comparable across
takes of the same unit — that, not an absolute bar, is what picks the best one.
"""
from __future__ import annotations

import importlib
import re
import threading

import numpy as np


def _sibling(name: str):
    """A sibling module, imported the same way standalone and vendored."""
    if __package__:
        return importlib.import_module("." + name, __package__)
    return importlib.import_module(name)

# phoneme alphabet (model) → sound class; vowels, ?, y, v dropped — and h:
# Persian /h/ is weak and routinely elided («پرسش‌ها» = porseSA), while the
# letter ه also writes the vowel e («به», «خانه»). Comparing it only adds noise.
_PH_CLASS = {c: c for c in "bptsjCxdzr;SqfkglmnZ"}
_PH_CLASS["Z"] = ";"
# Persian letters → sound class
_FA_CLASS = {
    "ب": "b", "پ": "p", "ت": "t", "ط": "t", "ث": "s", "س": "s", "ص": "s",
    "ج": "j", "چ": "C", "خ": "x", "د": "d", "ذ": "z",
    "ز": "z", "ض": "z", "ظ": "z", "ر": "r", "ژ": ";", "ش": "S", "ق": "q",
    "غ": "q", "ف": "f", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m",
    "ن": "n",
}
_FA_WORD = re.compile(r"[؀-ۿ‌]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def phoneme_skeleton(ph: str) -> str:
    return "".join(_PH_CLASS.get(c, "") for c in ph.replace("1", ""))


def _fa_word_skeleton(w: str) -> str:
    return "".join(_FA_CLASS.get(c, "") for c in w)


def _latin_word_skeleton(w: str) -> str:
    try:
        ph = _sibling("en_phonemes").word(w)
    except Exception:                                     # pragma: no cover
        ph = None
    if ph is None:
        return _fa_word_skeleton(_sibling("g2p_onnx").transliterate_text(w))
    return phoneme_skeleton(ph)


def text_skeleton(text: str, lang: str = "fa") -> str:
    t = _sibling("frontend").speakable_text(text) if re.search(r"\d", text) else text
    out = []
    for m in re.finditer(r"[؀-ۿ‌]+|[A-Za-z][A-Za-z']*", t):
        w = m.group(0)
        out.append(_latin_word_skeleton(w) if w[0].isascii() else _fa_word_skeleton(w))
    return "".join(out)


def _edits(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _align_unit_errors(ref: str, hyp: str, owner: list[int], n_units: int) -> list[int]:
    """Edit-distance alignment of `hyp` onto `ref`, with each edit charged to
    the unit that owns the reference position it touches. An insertion is
    charged to the unit of the NEXT reference character — so a phrase replayed
    at the start of a unit (the leak) lands on that unit, not its neighbour."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = cost of ref[:i] vs hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != hyp[j - 1]))
    errs = [0] * n_units
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                errs[owner[i - 1]] += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            errs[owner[i - 1]] += 1                       # deleted (not said)
            i -= 1
        else:
            errs[owner[min(i, n - 1)] if n else 0] += 1   # inserted (extra)
            j -= 1
    return errs


def unit_errors(unit_phonemes: list[str], transcript: str, lang: str = "fa") -> tuple[float, list[float]]:
    """(sentence skeleton CER, per-unit error rate) of a transcript of the
    stitched units. A unit's rate is its charged edits over its own length,
    floored at 4 consonants so a two-consonant unit is not condemned by one
    noisy character."""
    sks = [phoneme_skeleton(u) for u in unit_phonemes]
    ref = "".join(sks)
    owner = [k for k, sk in enumerate(sks) for _ in sk]
    hyp = text_skeleton(transcript, lang)
    if not ref:
        return 0.0, [0.0] * len(sks)
    errs = _align_unit_errors(ref, hyp, owner, len(sks))
    total = min(1.5, sum(errs) / len(ref))
    return total, [e / max(4, len(sk)) if e >= 2 else 0.0 for e, sk in zip(errs, sks)]


def skeleton_cer(target_phonemes: str, transcript: str, lang: str = "fa") -> float:
    ref = phoneme_skeleton(target_phonemes)
    hyp = text_skeleton(transcript, lang)
    if not ref:
        return 0.0
    return min(1.5, _edits(hyp, ref) / len(ref))


class AsrVerifier:
    """Callable(audio, sample_rate, target_phonemes, lang) -> skeleton CER.

    `model` is a loaded faster_whisper.WhisperModel — on a GPU one call costs
    tens of milliseconds, which is why this is worth doing per unit. Greedy
    decoding (beam 1), no prompt: a prompt containing the target would talk
    the recogniser into hearing it.
    """

    #: A unit whose charged error rate is above this is regenerated.
    threshold = 0.34
    #: A sentence whose skeleton CER is at or below this is accepted whole,
    #: without looking at its units — the recogniser's own noise floor.
    sentence_ok = 0.10

    def __init__(self, model, *, threshold: float | None = None, beam_size: int = 1):
        self.model = model
        self.beam_size = beam_size
        if threshold is not None:
            self.threshold = threshold
        self.calls = 0
        self.seconds = 0.0
        self._lock = threading.Lock()
        self._suppress = None

    def suppress_tokens(self) -> list[int]:
        """Whisper writes a spoken number as digits («14.30», «۲۵۰») — while
        the target is words («چهارده و سی دقیقه»), and a clock time and a
        decimal look the same in digits. Forbidding every token that contains
        a digit makes it spell numbers out, which is what is compared."""
        if self._suppress is None:
            ids = [-1]
            try:
                tok = self.model.hf_tokenizer
                for _piece, i in tok.get_vocab().items():
                    if any(ch.isdigit() for ch in tok.decode([i])):
                        ids.append(i)
            except Exception:                              # pragma: no cover
                pass
            self._suppress = ids
        return self._suppress

    def transcribe(self, audio: np.ndarray, sr: int, lang: str) -> str:
        from scipy.signal import resample_poly

        a = np.asarray(audio, dtype=np.float32)
        if sr != 16000:
            g = np.gcd(int(sr), 16000)
            a = resample_poly(a, 16000 // g, int(sr) // g).astype(np.float32)
        segs, _ = self.model.transcribe(
            a, language="en" if lang == "en" else "fa", beam_size=self.beam_size,
            temperature=0.0, condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True, suppress_tokens=self.suppress_tokens())
        return " ".join(s.text.strip() for s in segs).strip()

    def check_sentence(self, audio: np.ndarray, sr: int, unit_phonemes: list[str],
                       lang: str = "fa") -> tuple[float, list[float]]:
        """One transcription of a whole stitched sentence → (sentence error,
        per-unit error). One recogniser call per SENTENCE rather than per
        unit: whisper pads every call to a 30 s window, so checking 2-second
        units one at a time cost ~10x more for the same information."""
        import time

        t0 = time.perf_counter()
        hyp = self.transcribe(audio, sr, lang)
        total, per_unit = unit_errors(unit_phonemes, hyp, lang)
        with self._lock:
            self.calls += 1
            self.seconds += time.perf_counter() - t0
        return total, per_unit

    def __call__(self, audio: np.ndarray, sr: int, phonemes: str, lang: str = "fa") -> float:
        total, _ = self.check_sentence(audio, sr, [phonemes], lang)
        return total
