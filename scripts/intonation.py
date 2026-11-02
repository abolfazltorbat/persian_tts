"""Sentence-type intonation for a punctuation-blind TTS model.

pocket-tts-farsi-v2 was trained on phonemes with every punctuation mark
removed (the G2P drops them; «؟» was stripped because it shares a symbol with
the glottal stop). «رفتی؟» and «رفتی.» therefore reach the model as the SAME
input, and every sentence comes out with the falling contour of a statement.
The text still has its punctuation, so the sentence type is known here, after
synthesis, and the contour can be restored on the audio:

  qyn  yes/no question  the last ~0.45 s of voicing rises to end above the
                        speaker's median pitch (a final rise, H%)
  qwh  wh-question      UNCHANGED: the question word is in the text and the
                        model already accents it and falls at the end, as
                        Persian wh-questions do (a rule only over-accented)
  exc  exclamation      wider pitch range around the median, raised level,
                        a louder first half
  st   statement        unchanged

Targets come from real Persian speech (GPTInformal-Persian, research/
PROGRESS.md Study C): statements end ~4.3 st below the speaker's median,
questions near it; yes/no questions rise at the very end.

Pitch is changed by TD-PSOLA (pitch-synchronous overlap-add) on a numpy YIN
track: time scale and the spectral envelope (formants, so the voice) stay; only
the regions the rule touches are replaced, with a crossfade. numpy + scipy
only — it runs in every interpreter the engine runs in.
"""
from __future__ import annotations

import re

import numpy as np

_WH = re.compile(r"(?:^|[\s‌«(])(چی|چیه|چه|چرا|کجا|کجاست|کی|کیه|چطور|چطوری|چگونه|چند|چقدر|کدوم|کدام|"
                 r"چیست|چیکار|چکار|چندم|کو)(?=$|[\s‌؟?!.،,»)])")
_EN_WH = re.compile(r"^\s*(what|where|when|why|who|whom|whose|which|how)\b", re.I)


def sentence_kind(text: str) -> str:
    """'qyn' | 'qwh' | 'exc' | 'st' from the written sentence (punctuation kept)."""
    t = text.strip().rstrip(" ‌»\"')")
    if t.endswith(("؟", "?")):
        body = t.rstrip("؟?!. ")
        if re.search(r"(^|\s)آیا(\s|$)", body):
            return "qyn"
        if _WH.search(body) or _EN_WH.search(body):
            return "qwh"
        return "qyn"
    if t.endswith("!"):
        return "exc"
    return "st"


def wh_position(text: str) -> float | None:
    """Where the (last) question word sits in the sentence, as the fraction of
    the sentence's letters before its middle — mapped onto the voiced span it
    places the wh-word's nuclear accent without any alignment."""
    t = text.strip()
    ms = list(_WH.finditer(t))
    if not ms:
        m = _EN_WH.search(t)
        if not m:
            return None
        lo, hi = m.span(1)
    else:
        lo, hi = ms[-1].span(1)
    letters = lambda u: sum(ch.isalpha() for ch in u)
    total = letters(t)
    if not total:
        return None
    return (letters(t[:lo]) + 0.5 * letters(t[lo:hi])) / total


# ── pitch tracking: YIN (de Cheveigné & Kawahara), vectorised over frames ──

def yin(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 450.0,
        hop_s: float = 0.005, thresh: float = 0.15):
    """-> (f0 Hz per frame, 0 where unvoiced; hop in samples).
    Two passes: the first finds the speaker's median F0, the second searches
    only -10 .. +12 st around it — the octave-down errors YIN makes on creaky
    sentence ends (half the true F0) fall outside that window."""
    f0, hop = _yin(x, sr, fmin, fmax, hop_s, thresh)
    v = f0 > 0
    if v.sum() < 10:
        return f0, hop
    med = float(np.median(f0[v]))
    lo, hi = max(fmin, med * 2 ** (-10 / 12)), min(fmax * 1.3, med * 2 ** (12 / 12))
    # second pass: the window already rules out octave errors, so weaker
    # periodicity (creaky, decaying sentence ends) may count as voiced
    f0, hop = _yin(x, sr, lo, hi, hop_s, thresh + 0.07, floor_db=40.0)
    return f0, hop


def _yin(x: np.ndarray, sr: int, fmin: float, fmax: float, hop_s: float, thresh: float,
         floor_db: float = 30.0):
    hop = max(1, int(hop_s * sr))
    tmax = int(sr / fmin)
    tmin = max(2, int(sr / fmax))
    W = 2 * tmax
    if len(x) < W + hop:
        return np.zeros(0), hop
    pad = np.pad(x.astype(np.float64), (W // 2, W))
    n = 1 + (len(x) - 1) // hop
    idx = np.arange(n) * hop
    fr = np.lib.stride_tricks.sliding_window_view(pad, W + tmax)[idx]        # [n, W+tmax]
    a = fr[:, :W]
    nfft = 1 << int(np.ceil(np.log2(W + tmax + W)))
    corr = np.fft.irfft(np.fft.rfft(fr, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)[:, :tmax + 1]
    sq = np.cumsum(np.pad(fr ** 2, ((0, 0), (1, 0))), axis=1)
    e0 = sq[:, W]
    et = sq[:, W:W + tmax + 1] - sq[:, :tmax + 1]
    d = e0[:, None] + et - 2 * corr
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1) / np.maximum(cm, 1e-12)
    energy = 10 * np.log10(e0 / W + 1e-12)
    floor = np.percentile(energy, 95) - floor_db
    f0 = np.zeros(n)
    for i in range(n):
        if energy[i] < floor:
            continue
        row = dn[i, tmin:tmax]
        below = np.flatnonzero(row < thresh)
        if len(below):
            k = below[0]
            while k + 1 < len(row) and row[k + 1] < row[k]:
                k += 1
        else:
            k = int(np.argmin(row))
            if row[k] > 0.35:
                continue
        t = k + tmin
        if 1 <= t < tmax:          # parabolic refinement
            y0, y1, y2 = dn[i, t - 1], dn[i, t], dn[i, t + 1]
            den = y0 - 2 * y1 + y2
            t = t + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
        f0[i] = sr / t
    return _clean_track(f0), hop


def _clean_track(f0: np.ndarray) -> np.ndarray:
    """Drop octave jumps and isolated frames: a voiced frame > 6 st from the
    median of its 9-frame neighbourhood, or voiced runs shorter than 25 ms."""
    out = f0.copy()
    v = out > 0
    if v.sum() < 5:
        return np.zeros_like(out)
    st = np.where(v, 12 * np.log2(np.where(v, out, 1) / np.median(out[v])), np.nan)
    for i in np.flatnonzero(v):
        nb = st[max(0, i - 4): i + 5]
        if abs(st[i] - np.nanmedian(nb)) > 6:
            out[i] = 0
    v = out > 0
    i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            if j - i < 5:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


# ── TD-PSOLA ───────────────────────────────────────────────────────────────

def _marks(x: np.ndarray, f0s: np.ndarray, sr: int, uv_period: int) -> tuple[np.ndarray, np.ndarray]:
    """Analysis marks (sample positions) and the local period at each.
    Voiced: one mark per glottal period, on the waveform peak of a smoothed
    copy (one polarity, so grains align). Unvoiced: fixed spacing."""
    from scipy.signal import butter, sosfiltfilt
    sm = sosfiltfilt(butter(2, 900, btype="low", fs=sr, output="sos"), x)
    pos = sm if np.max(sm) >= -np.min(sm) else -sm
    marks, periods = [], []
    t = 0
    N = len(x)
    while t < N:
        f = f0s[min(t, N - 1)]
        if f > 0:
            P = int(round(sr / f))
            lo, hi = t, min(N, t + P)
            if marks and periods[-1] > 0:        # search near the expected next mark
                c = marks[-1] + P
                lo, hi = max(marks[-1] + int(0.7 * P), c - int(0.3 * P)), min(N, c + int(0.3 * P))
            if hi - lo < 2:
                break
            m = lo + int(np.argmax(pos[lo:hi]))
            marks.append(m)
            periods.append(P)
            t = m + max(1, int(0.7 * P))
        else:
            marks.append(t)
            periods.append(-uv_period)           # negative = unvoiced
            t += uv_period
    return np.asarray(marks), np.asarray(periods)


def psola(x: np.ndarray, sr: int, f0_samples: np.ndarray, ratio_samples: np.ndarray) -> np.ndarray:
    """Pitch-scale x by ratio(t) (>1 raises), time scale preserved."""
    uv = int(0.005 * sr)
    am, ap = _marks(x, f0_samples, sr, uv)
    if len(am) < 3:
        return x.copy()
    N = len(x)
    # local spacing from the marks themselves (not the f0 track): with ratio 1
    # the output marks then land exactly on the analysis marks and the
    # overlap-add of the asymmetric Hann grains reconstructs the input
    left = np.diff(am, prepend=am[0] - abs(ap[0]))
    right = np.diff(am, append=am[-1] + abs(ap[-1]))
    left = np.maximum(left, 2)
    right = np.maximum(right, 2)
    y = np.zeros(N + 4 * sr // 50)
    wsum = np.zeros_like(y)
    o = float(am[0])
    k = 0
    while o < N:
        # analysis mark nearest the output time (time scale preserved)
        while k + 1 < len(am) and abs(am[k + 1] - o) <= abs(am[k] - o):
            k += 1
        voiced = ap[k] > 0
        r = float(ratio_samples[min(int(o), N - 1)]) if voiced else 1.0
        c, L, R = int(am[k]), int(left[k]), int(right[k])
        lo, hi = max(0, c - L), min(N, c + R)
        win = np.concatenate([np.hanning(2 * L + 1)[:L], np.hanning(2 * R + 1)[R:]])
        win = win[L - (c - lo): L + (hi - c)]
        g = x[lo:hi]
        s = int(round(o)) - (c - lo)
        if s >= 0 and s + len(g) <= len(y):
            y[s:s + len(g)] += g * win
            wsum[s:s + len(g)] += win
        step = R / max(r, 0.25) if voiced else R
        o += max(step, 1.0)
    y = y[:N]
    w = wsum[:N]
    out = np.where(w > 0.05, y / np.maximum(w, 0.05), x)
    return out.astype(np.float32)


# ── the rules ──────────────────────────────────────────────────────────────

def _smooth(v: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(v) < n:
        return v
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    return np.convolve(np.pad(v, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


# Only the types the model cannot infer from the words are changed. A
# wh-question carries its question word, and the model already renders the
# Persian pattern for it (nuclear accent on the wh-word, +3.4 st, then a fall
# like a statement; real speech +3.8 st) — a rule there only over-accents
# (+5.9 st; research/PROGRESS.md, study C). A yes/no question differs from its
# statement twin ONLY by the mark, and so does an exclamation's delivery.
MODIFIED = ("qyn", "exc")

# per-type targets, semitones re the sentence's median F0
RISE_END_ST = 5.0          # yes/no question: last voiced frame ends this high (dev-tuned)
RISE_LEN_S = 0.40          # ... over this much of the final voicing
WH_LIFT_ST = 3.0           # wh-question: raised nuclear accent on the question word
WH_DEACCENT = 0.4          # wh-question: excursions after it shrunk by this share
EXC_RANGE = 1.25           # exclamation: pitch excursions scaled by this
EXC_LIFT_ST = 1.5          # exclamation: raised level
EXC_END_ST = -1.0          # exclamation: final fall suspended to here
EXC_GAIN_DB = 3.0          # exclamation: first half louder by this
MAX_SHIFT_ST = 9.0         # no frame is moved further than this (PSOLA stays clean)


def shift_contour(f0: np.ndarray, hop_s: float, kind: str, strength: float = 1.0,
                  wh_pos: float | None = None) -> np.ndarray:
    """Per-frame pitch shift in semitones that turns the rendered contour into
    `kind`'s. Zero outside the touched region and on unvoiced frames' ends."""
    v = np.flatnonzero(f0 > 0)
    sh = np.zeros(len(f0))
    if len(v) < 10 or kind == "st":
        return sh
    med = np.median(f0[v])
    st = np.full(len(f0), np.nan)
    st[v] = 12 * np.log2(f0[v] / med)
    # a smooth version of the rendered contour (interpolated over gaps)
    from scipy.signal import medfilt
    stv = medfilt(st[v], 7) if len(v) >= 7 else st[v]
    sti = np.interp(np.arange(len(f0)), v, stv)
    sts = _smooth(sti, max(3, int(0.06 / hop_s)))
    a, b = v[0], v[-1]
    span = b - a
    t = np.arange(len(f0))
    if kind == "qyn":
        L = min(int(RISE_LEN_S / hop_s), max(int(0.6 * span), 1))
        r0 = b - L
        u = np.clip((t - r0) / max(L, 1), 0, 1)
        start = sts[max(r0, a)]
        target = start + (RISE_END_ST - start) * u ** 1.6
        sh = np.where(t >= r0, target - sts, 0.0)
        sh = np.clip(sh, -1.0, MAX_SHIFT_ST)
    elif kind == "qwh":
        # Persian wh-questions FALL like statements; the wh-word carries the
        # nuclear accent and what follows is deaccented (Sadat-Tehrani 2011).
        # So: a raised accent on the wh-word, the material after it pulled
        # down toward the final level, the ending itself left alone.
        pos = 0.25 if wh_pos is None else wh_pos
        c = a + pos * span
        sig = max(0.10 / hop_s, 0.06 * span)
        bump = WH_LIFT_ST * np.exp(-0.5 * ((t - c) / sig) ** 2)
        after = np.clip((t - (c + 1.5 * sig)) / max(sig, 1), 0, 1)
        end_level = np.median(sts[max(a, b - int(0.15 / hop_s)):b + 1])
        deacc = -WH_DEACCENT * np.clip(sts - end_level, 0, None) * after
        sh = np.where((t >= a) & (t <= b), bump + deacc, 0.0)
    elif kind == "exc":
        # raised level and wider excursions over the whole sentence, and — as
        # in real exclamations — no deep final fall (ends near the median)
        sh = (EXC_RANGE - 1.0) * np.clip(sts, -3, 8) + EXC_LIFT_ST
        L = min(int(0.40 / hop_s), max(int(0.5 * span), 1))
        r0 = b - L
        u = np.clip((t - r0) / max(L, 1), 0, 1)
        sh = sh + np.where(t >= r0, np.clip(EXC_END_ST - sts, 0.0, MAX_SHIFT_ST) * u ** 1.2, 0.0)
        sh = np.where((t >= a) & (t <= b + int(0.05 / hop_s)), np.minimum(sh, MAX_SHIFT_ST), 0.0)
    sh = _smooth(sh * strength, max(3, int(0.04 / hop_s)))
    return sh


def apply(x: np.ndarray, sr: int, kind: str, strength: float = 1.0, text: str | None = None) -> np.ndarray:
    """Rendered sentence audio -> the same audio with `kind`'s intonation.
    `text` (the written sentence) places the wh-word accent of a wh-question."""
    x = np.asarray(x, dtype=np.float32)
    if kind not in MODIFIED or strength <= 0 or len(x) < int(0.25 * sr):
        return x
    f0, hop = yin(x, sr)
    if not len(f0) or (f0 > 0).sum() < 10:
        return x
    sh = shift_contour(f0, hop / sr, kind, strength, wh_position(text) if (text and kind == "qwh") else None)
    if not np.any(np.abs(sh) > 0.05):
        return x
    # per-sample f0 and ratio (voiced f0 held across short gaps for marking)
    ts = np.arange(len(x)) / hop
    fr = np.arange(len(f0))
    f0s = np.where(np.interp(ts, fr, (f0 > 0).astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
    ratio = 2.0 ** (np.interp(ts, fr, sh) / 12.0)
    y = psola(x.astype(np.float64), sr, f0s, ratio)
    # splice: original where the rule changed nothing, crossfaded 30 ms
    touched = np.abs(np.interp(ts, fr, sh)) > 0.05
    m = _smooth(touched.astype(float), int(0.03 * sr)) if touched.any() else touched.astype(float)
    m = np.clip(m * 1.5, 0, 1)
    out = (1 - m) * x + m * y
    if kind == "exc" and EXC_GAIN_DB * strength:
        v = np.flatnonzero(f0 > 0)
        a, b = v[0] * hop, v[-1] * hop
        mid = a + (b - a) // 2
        g = np.ones(len(x))
        ramp = np.clip((np.arange(len(x)) - mid) / max(1, int(0.25 * sr)), 0, 1)
        g = 10 ** ((EXC_GAIN_DB * strength * (1 - ramp)) / 20)
        out = out * g
        pk = np.max(np.abs(out))
        if pk > 0.99:
            out *= 0.99 / pk
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  v2 (research/PROGRESS.md, study D): expressive sentence prosody
# ══════════════════════════════════════════════════════════════════════════════
#
# Study C restored the final rise of yes/no questions and found its cost was the
# signal processing. Listening (owner, 11-01): questions, surprise and loudness
# still do not change enough through a text. v2 changes four things:
#
#   * more sentence types from the text: surprised / echo questions («جدی؟»,
#     «؟!»), alternative questions («چای یا قهوه؟», which FALL), wh-questions
#     with the suspended ending of real speech, exclamations with an intensity
#     from their marks and interjections;
#   * a question is more than a pitch rise: the final syllable is also
#     lengthened and its energy kept up (a declarative ending decays 20-30 dB);
#   * one PSOLA pass changes pitch, timing and level together (zero-frequency-
#     filtering pitch marks were tried and lost to study C's waveform peaks);
#   * a sentence-level loudness plan (type + position in the paragraph) instead
#     of every sentence at the same level.

VERSION = 2            # 1 = the study C rule exactly (apply_v1)
APPLY = True           # research: False keeps the styled prompts but skips the audio rule
                       # (rule variants are then applied offline, research/rule_grid.py)

_ECHO_WORDS = {"واقعاً", "واقعا", "جدی", "جدا", "راستی", "راست", "نه", "مطمئنی", "حتماً", "حتما",
               "یعنی", "همین", "همینه", "اِ", "عه", "اه", "آره", "بله", "چی", "کی", "کجا", "چرا",
               "really", "seriously", "what", "sure", "right"}
_INTERJ = re.compile(r"^\s*(?:وای|ای\s*وای|آخ|اوه|اوخ|عجب|به\s*به|به‌به|آفرین|هورا|ایول|آه|خدایا|"
                     r"وای\s*خدا|اه|آی|wow|oh|great|amazing)(?=$|[\s‌،,!؟?.])", re.I)
_EXCLAMATIVE = re.compile(r"^\s*(?:چه|چقدر|چقد|عجب|چه‌قدر)\s")
_ALT = re.compile(r"\sیا\s|\sor\s", re.I)
_TAG = re.compile(r"(?:یا\s+نه|مگه\s+نه|مگر\s+نه|،\s*نه|, *right|, *no)\s*$", re.I)


def sentence_plan(text: str) -> tuple[str, float]:
    """(kind, intensity) from the written sentence. Kinds: st, qyn, qwh, qalt,
    qecho (surprised / echo question), exc. Intensity 1 = an ordinary mark;
    «!!», an interjection or an exclamative «چه ...!» raise it."""
    t = text.strip().rstrip(" ‌»\"')")
    m = re.search(r"[.!?؟…]+$", t)
    marks = m.group(0) if m else ""
    body = t[: len(t) - len(marks)].strip()
    words = [w for w in re.split(r"[\s‌،,]+", body) if w]
    has_q = "؟" in marks or "?" in marks
    has_x = "!" in marks
    if has_q and has_x:
        return "qecho", 1.3
    if has_q:
        w0 = words[0].strip("«»").lower() if words else ""
        if (len(words) == 1 and w0 in _ECHO_WORDS) or (len(words) == 2 and w0 in _ECHO_WORDS
                                                       and not _WH.search(" " + w0)):
            return "qecho", 1.0
        if re.search(r"(^|\s)آیا(\s|$)", body):
            return ("qalt" if _ALT.search(body) and not _TAG.search(body) else "qyn"), 1.0
        if _TAG.search(body):
            return "qyn", 1.0
        if _WH.search(body) or _EN_WH.search(body):
            return "qwh", 1.0
        if _ALT.search(body):
            return "qalt", 1.0
        return "qyn", 1.0
    if has_x:
        k = 1.0 + 0.3 * (marks.count("!") > 1) + 0.25 * bool(_INTERJ.search(body) or _EXCLAMATIVE.search(body))
        return "exc", min(k, 1.5)
    return "st", 0.0


# ── pitch marks: zero-frequency filtering (Murty & Yegnanarayana 2008) ─────

def zff_epochs(x: np.ndarray, sr: int, period_s: float) -> np.ndarray:
    """Glottal closure instants: positive zero crossings of the zero-frequency
    filtered signal. The two 0-Hz resonators are double integrators; the local
    mean (window ~1.5 pitch periods) is removed after each, which is the same
    LTI chain as removing it at the end but stays inside float64 range."""
    s = np.diff(np.asarray(x, dtype=np.float64), prepend=float(x[0]))
    n = max(3, int(round(1.5 * period_s * sr)) | 1)
    k = np.ones(n) / n

    def detrend(y):
        for _ in range(3):
            y = y - np.convolve(np.pad(y, (n // 2, n // 2), mode="edge"), k, mode="valid")
        return y

    y = s
    for _ in range(2):
        y = detrend(np.cumsum(np.cumsum(y)))
    return np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0)) + 1


def marks_v2(x: np.ndarray, sr: int, f0s: np.ndarray, uv_s: float = 0.005):
    """Analysis marks + voiced flag. Voiced runs: ZFF epochs, cleaned to one per
    period (too-close ones dropped, gaps filled at the local period); runs where
    ZFF finds too few epochs fall back to uniform marks at the local period.
    Unvoiced: uniform `uv_s` spacing."""
    N = len(x)
    v = f0s > 0
    if not v.any():
        m = np.arange(0, N, max(1, int(uv_s * sr)))
        return m, np.zeros(len(m), bool)
    P_med = sr / float(np.median(f0s[v]))
    ep = zff_epochs(x, sr, P_med / sr)
    uv = max(1, int(uv_s * sr))
    marks, flags = [], []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))          # voiced [start, end)
    pos = 0
    for a, b in runs:
        marks += list(range(pos, a, uv))
        flags += [False] * len(range(pos, a, uv))
        e = ep[(ep >= a) & (ep < b)]
        run = []
        for c in e:
            P = sr / f0s[c]
            if run and c - run[-1] < 0.5 * P:
                continue
            while run and c - run[-1] > 1.8 * P:        # a missed epoch: fill
                run.append(run[-1] + int(round(P)))
            run.append(int(c))
        if len(run) < max(3, 0.5 * (b - a) / P_med):    # ZFF failed here
            run, c = [], a
            while c < b:
                run.append(int(c))
                c += max(2, int(round(sr / f0s[min(int(c), N - 1)])))
        marks += run
        flags += [True] * len(run)
        pos = (run[-1] + max(2, int(round(sr / f0s[run[-1]])))) if run else b
        pos = max(pos, b)
    marks += list(range(pos, N, uv))
    flags += [False] * len(range(pos, N, uv))
    m = np.asarray(marks)
    keep = np.concatenate([[True], np.diff(m) > 0])
    return m[keep], np.asarray(flags)[keep]


#: pitch marks for psola_v2: "peak" = study C's waveform-peak marks, "zff" =
#: zero-frequency-filtering epochs. ZFF LOST on UTMOS (flat +2 st: -0.37 vs
#: -0.27; +4 st: -0.67 vs -0.52, 324 renders) — epochs on codec speech jitter.
MARKS = "peak"


def psola_v2(x: np.ndarray, sr: int, f0s: np.ndarray, ratio: np.ndarray,
             beta: np.ndarray | None = None, gain: np.ndarray | None = None):
    """Pitch x ratio(t), local duration x beta(t) (>1 lengthens), level x gain(t),
    in ONE overlap-add pass; all three curves are per analysis sample.
    -> (y, tau): y the output, tau[i] the analysis sample output sample i came
    from (for splicing). ratio = beta = gain = 1 reproduces x exactly."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if MARKS == "zff":
        am, vf = marks_v2(x, sr, f0s)
    else:
        am, ap = _marks(x, f0s, sr, int(0.005 * sr))
        vf = ap > 0
    if len(am) < 3:
        return x.copy(), np.arange(N, dtype=np.float64)
    beta = np.ones(N) if beta is None else beta
    gain = np.ones(N) if gain is None else gain
    if MARKS == "zff":
        left = np.maximum(np.diff(am, prepend=am[0] - (am[1] - am[0])), 2)
        right = np.maximum(np.diff(am, append=am[-1] + (am[-1] - am[-2])), 2)
    else:                                   # exactly as psola()
        left = np.maximum(np.diff(am, prepend=am[0] - abs(ap[0])), 2)
        right = np.maximum(np.diff(am, append=am[-1] + abs(ap[-1])), 2)
    cap = int(N * max(1.0, float(beta.max()))) + sr // 10
    y = np.zeros(cap)
    wsum = np.zeros(cap)
    o = float(am[0])
    tau = float(am[0])
    k = 0
    om, tm = [0.0, o], [0.0, tau]
    while tau < N:
        while k + 1 < len(am) and abs(am[k + 1] - tau) <= abs(am[k] - tau):
            k += 1
        ti = min(int(tau), N - 1)
        r = float(ratio[ti]) if vf[k] else 1.0
        c, L, R = int(am[k]), int(left[k]), int(right[k])
        lo, hi = max(0, c - L), min(N, c + R)
        win = np.concatenate([np.hanning(2 * L + 1)[:L], np.hanning(2 * R + 1)[R:]])
        win = win[L - (c - lo): L + (hi - c)]
        s = int(round(o)) - (c - lo)
        if s >= 0 and s + (hi - lo) <= cap:
            y[s:s + hi - lo] += x[lo:hi] * win * float(gain[c if c < N else N - 1])
            wsum[s:s + hi - lo] += win
        step = max(1.0, (R / max(r, 0.25)) if vf[k] else float(R))
        o += step
        tau += step / max(float(beta[ti]), 0.25)
        om.append(o)
        tm.append(tau)
    n_out = int(round(np.interp(N, tm, om)))
    n_out = min(max(n_out, 1), cap)
    tau_o = np.interp(np.arange(n_out), om, tm)
    src = x[np.clip(np.round(tau_o).astype(int), 0, N - 1)]
    w = wsum[:n_out]
    out = np.where(w > 0.05, y[:n_out] / np.maximum(w, 0.05), src)
    return out, tau_o


# ── where the last syllable starts (energy nuclei; no aligner) ─────────────

def _band_env_db(x: np.ndarray, sr: int, hop: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    b = sosfiltfilt(butter(2, [300, 2500], btype="band", fs=sr, output="sos"), x)
    w = int(0.025 * sr)
    e = np.convolve(b ** 2, np.hanning(w) / np.hanning(w).sum(), mode="same")
    return 10 * np.log10(e[::hop][: 1 + (len(x) - 1) // hop] + 1e-10)


def last_syllable(x: np.ndarray, sr: int, f0: np.ndarray, hop: int,
                  min_s: float = 0.16, max_s: float = 0.45) -> int:
    """Frame where the final syllable starts: the energy valley before the last
    vowel nucleus inside the voiced span, kept 0.16-0.45 s before voicing ends."""
    from scipy.signal import find_peaks
    v = np.flatnonzero(f0 > 0)
    a, b = int(v[0]), int(v[-1])
    e = _band_env_db(x, sr, hop)[: len(f0)]
    if len(e) < b + 1:
        e = np.pad(e, (0, b + 1 - len(e)), mode="edge")
    seg = e[a:b + 1]
    hop_s = hop / sr
    pk, _ = find_peaks(seg, prominence=3.0, distance=max(1, int(0.07 / hop_s)))
    lo_f, hi_f = b - int(max_s / hop_s), b - int(min_s / hop_s)
    if len(pk) >= 2:
        p1, p0 = a + pk[-1], a + pk[-2]
        on = p0 + int(np.argmin(e[p0:p1 + 1]))
    elif len(pk) == 1:
        on = a + pk[-1] - int(0.08 / hop_s)
    else:
        on = b - int(0.30 / hop_s)
    return int(np.clip(on, max(a, lo_f), max(a, hi_f)))


# ── the v2 rules ───────────────────────────────────────────────────────────
# dev-tuned on odd-numbered prosody-set sentences (research/PROGRESS.md, study D).
# Register and liveliness come from the styled prompt (STYLE_OF_KIND) — the model
# speaks them itself; the rule only adds what a prompt cannot: the ending.
Q_END_ST = 9.0         # yes/no question: voicing ends this far above the median (measured end ~ +0.5 st = real)
Q_SHAPE = 1.3          # rise shape u**Q_SHAPE over the final syllable
Q_LEN = 0.15           # final syllable lengthened by this share
Q_SUSTAIN_DB = 0.0     # final syllable's energy decay undone up to this much (UTMOS -0.11 at 6 dB: off)
Q_REG_ST = 0.0         # whole question raised by this much (PSOLA; the prompt does it better)
ECHO_END_ST = 8.0      # surprised / echo question (the lively prompt already raises it ~3-4 st)
ECHO_LEN = 0.15
ECHO_REG_ST = 0.0
WH_END_ST = -1.0       # wh-question: the final fall stops here (real speech -0.94)
WH_LEN = 0.0
X_RANGE = 1.0          # exclamation: excursions x this (1 = left to the lively prompt)
X_LIFT_ST = 0.0        # ... level raised (x intensity)
X_END_ST = -1.0        # ... final fall suspended
X_GAIN_DB = 0.0        # ... first half louder (study C: +3 dB) — real excited speech does the
                       #     opposite (ShEMO: energy kept up to the end), see RAMP_DB
MAX_SHIFT_V2 = 11.0

#: prosody-styled prompt per sentence type (engine.prompt_styles names;
#: frontend.styles_for). The model then itself speaks higher and livelier —
#: no processing of the output (research/style_probe.py).
STYLE_OF_KIND = {"exc": "lively", "qecho": "lively"}
# ("ask" (+2 st, x1.25, +2 dB) for yes/no and wh-questions was tried and dropped on held-out
# data: the wider range deepened the fall before the final rise — yes/no «؟» heard .12 -> .07,
# question-statement end gap +3.6 -> +1.8 st — and made wh-questions fall deeper, away from
# real speech; research/PROGRESS.md, study D)
#: the styles the product uses: name -> (register st, range scale, level dB)
PROMPT_STYLES = {"lively": (4.0, 1.6, 4.0)}

#: second half of the sentence raised by up to this (dB; a plain gain envelope,
#: no PSOLA): the rendered speech decays ~5.5-6 dB from its first half to its
#: second; real expressive speech (ShEMO) decays 4.0 dB when neutral, 1.8 when
#: surprised, 0.6 when happy, 3.1 in neutral yes/no questions.
RAMP_DB = {"qyn": 2.0, "qecho": 3.5, "exc": 3.5}

#: sentence loudness offsets in dB (frontend.join), by type and paragraph place
GAIN_DB = {"st": 0.0, "qyn": 0.5, "qwh": 0.5, "qalt": 0.0, "qecho": 2.0, "exc": 3.0}
PARA_FIRST_DB = 1.0    # a paragraph's first sentence (high onset of a new topic)
PARA_LAST_DB = -1.5    # a paragraph's last sentence (winding down)


def _sts(f0: np.ndarray, hop_s: float):
    from scipy.signal import medfilt
    v = np.flatnonzero(f0 > 0)
    med = np.median(f0[v])
    st = 12 * np.log2(f0[v] / med)
    stv = medfilt(st, 7) if len(v) >= 7 else st
    sti = np.interp(np.arange(len(f0)), v, stv)
    return _smooth(sti, max(3, int(0.06 / hop_s))), v


def contour_v2(x, sr, f0, hop, kind: str, intensity: float, strength: float):
    """-> per-frame (shift st, stretch beta, gain dB) for `kind`."""
    n = len(f0)
    sh, be, gd = np.zeros(n), np.ones(n), np.zeros(n)
    v = np.flatnonzero(f0 > 0)
    if len(v) < 10 or kind in ("st", "qalt"):
        return sh, be, gd
    hop_s = hop / sr
    sts, v = _sts(f0, hop_s)
    a, b = int(v[0]), int(v[-1])
    t = np.arange(n)
    if kind in ("qyn", "qecho"):
        echo = kind == "qecho"
        end = (ECHO_END_ST if echo else Q_END_ST)
        on = last_syllable(x, sr, f0, hop)
        L = max(1, b - on)
        u = np.clip((t - on) / L, 0, 1)
        start = sts[on]
        target = start + (end - start) * u ** Q_SHAPE
        sh = np.where(t >= on, target - sts, 0.0)
        sh = np.clip(sh, -1.5, MAX_SHIFT_V2)
        sh += (ECHO_REG_ST if echo else Q_REG_ST) * (t >= a)
        # the final syllable is longer and does not decay
        bump = np.clip((t - on) / max(1, int(0.05 / hop_s)), 0, 1) * (t <= b + int(0.04 / hop_s))
        be = 1.0 + (ECHO_LEN if echo else Q_LEN) * bump
        e = _band_env_db(x, sr, hop)[:n]
        e = np.pad(e, (0, max(0, n - len(e))), mode="edge")
        ref = np.percentile(e[v], 70) - 3.0
        es = _smooth(e, max(3, int(0.08 / hop_s)))
        gd = np.clip(ref - es, 0, Q_SUSTAIN_DB) * bump
    elif kind == "qwh":
        L = min(int(0.35 / hop_s), max(int(0.5 * (b - a)), 1))
        r0 = b - L
        u = np.clip((t - r0) / L, 0, 1)
        sh = np.where(t >= r0, np.clip(WH_END_ST - sts, 0.0, 6.0) * u ** 1.2, 0.0)
        be = 1.0 + WH_LEN * np.clip((t - (b - int(0.2 / hop_s))) / max(1, int(0.05 / hop_s)), 0, 1) * (t <= b)
    elif kind == "exc":
        k = intensity
        sh = (X_RANGE - 1.0) * k * np.clip(sts, -3, 8) + X_LIFT_ST * k
        L = min(int(0.40 / hop_s), max(int(0.5 * (b - a)), 1))
        r0 = b - L
        u = np.clip((t - r0) / L, 0, 1)
        sh = sh + np.where(t >= r0, np.clip(X_END_ST - sts, 0.0, MAX_SHIFT_V2) * u ** 1.2, 0.0)
        sh = np.where((t >= a) & (t <= b + int(0.05 / hop_s)), np.minimum(sh, MAX_SHIFT_V2), 0.0)
        mid = a + (b - a) // 2
        gd = X_GAIN_DB * k * np.clip(1 - (t - mid) / max(1, int(0.25 / hop_s)), 0, 1) * (t >= a)
    sm = max(3, int(0.04 / hop_s))
    return (_smooth(sh * strength, sm), 1 + _smooth((be - 1) * strength, sm),
            _smooth(gd * strength, sm))


def _second_half_ramp(y: np.ndarray, sr: int, db: float) -> np.ndarray:
    """Raise the second half of the speech by up to `db` (raised-cosine from the
    speech midpoint to its end, held after): energy kept up, as in excited speech."""
    e = 10 * np.log10(np.convolve(y.astype(np.float64) ** 2, np.ones(480) / 480, mode="same") + 1e-12)
    sp = np.flatnonzero(e > e.max() - 35)
    if len(sp) < sr // 10:
        return y
    a, b = int(sp[0]), int(sp[-1])
    mid = (a + b) // 2
    u = np.clip((np.arange(len(y)) - mid) / max(1, b - mid), 0, 1)
    return (y * 10 ** (db * (0.5 - 0.5 * np.cos(np.pi * u)) / 20)).astype(np.float32)


def limit(x: np.ndarray, sr: int, ceiling: float = 0.95, window_ms: float = 20.0,
          smooth_ms: float = 10.0) -> np.ndarray:
    """Peak limiter with look-ahead and no overshoot: the gain each sample needs
    (ceiling / |x|, at most 1) is minimum-filtered over `window_ms`, then
    smoothed over at most half that window, so the smoothed gain never exceeds
    what any sample under it needs. Only the rare peaks above the ceiling are
    touched — unlike scaling a whole sentence down, which undid the loudness
    plan and made a louder target pointless (a third of renders hit 0.98)."""
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x)
    if not len(x) or float(a.max()) <= ceiling:
        return x
    from scipy.ndimage import minimum_filter1d
    need = np.minimum(1.0, ceiling / np.maximum(a.astype(np.float64), 1e-9))
    w = max(3, int(window_ms * sr / 1000) | 1)
    g = minimum_filter1d(need, size=w, mode="nearest")
    L = max(3, min(int(smooth_ms * sr / 1000), w // 2) | 1)
    k = np.hanning(L + 2)[1:-1]
    k /= k.sum()
    g = np.convolve(np.pad(g, (L // 2, L // 2), mode="edge"), k, mode="valid")
    return np.clip(x * g, -ceiling, ceiling).astype(np.float32)


def process(x: np.ndarray, sr: int, text: str, strength: float = 1.0,
            para_first: bool = False, para_last: bool = False, peak_guard: bool = True) -> np.ndarray:
    """One rendered sentence -> the same sentence with its type's prosody (pitch,
    final timing, level) and its loudness offset in the text. strength 0 = x."""
    x = np.asarray(x, dtype=np.float32)
    if strength <= 0 or not len(x) or not APPLY:
        return x
    kind, inten = sentence_plan(text)
    g_db = GAIN_DB.get(kind, 0.0) * (inten if kind == "exc" else 1.0)
    g_db += PARA_FIRST_DB * para_first + PARA_LAST_DB * para_last
    y = x
    if kind not in ("st", "qalt") and len(x) >= int(0.25 * sr):
        f0, hop = yin(x, sr)
        if len(f0) and (f0 > 0).sum() >= 10:
            sh, be, gd = contour_v2(x, sr, f0, hop, kind, inten, strength)
            touched = (np.abs(sh) > 0.05) | (np.abs(be - 1) > 0.005) | (np.abs(gd) > 0.05)
            if touched.any():
                ts = np.arange(len(x)) / hop
                fr = np.arange(len(f0))
                f0s = np.where(np.interp(ts, fr, (f0 > 0).astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
                ratio = 2.0 ** (np.interp(ts, fr, sh) / 12.0)
                beta = np.interp(ts, fr, be)
                gain = 10 ** (np.interp(ts, fr, gd) / 20.0)
                z, tau = psola_v2(x, sr, f0s, ratio, beta, gain)
                m = _smooth(np.interp(ts, fr, touched.astype(float)), int(0.03 * sr))
                m = np.clip(m * 1.5, 0, 1)
                mo = np.interp(tau, np.arange(len(x)), m)
                src = x[np.clip(np.round(tau).astype(int), 0, len(x) - 1)]
                y = ((1 - mo) * src + mo * z).astype(np.float32)
    ramp = RAMP_DB.get(kind, 0.0) * (inten if kind == "exc" else 1.0) * strength
    if ramp > 0:
        y = _second_half_ramp(np.asarray(y, dtype=np.float32), sr, ramp)
    g = 10 ** (strength * g_db / 20.0)
    y = y * g
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak_guard and pk > 0.99:            # without a limiter after join: scale the sentence
        y = y * (0.99 / pk)
    return y.astype(np.float32)
