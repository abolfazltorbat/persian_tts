"""Study D prosody layer (scripts/intonation.py v2, prompt styles, join/render hints)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import intonation as I  # noqa: E402

SR = 24000


def _voiced(seconds=1.4, f_start=140.0, f_end=110.0, sr=SR):
    """A falling glottal-pulse-like tone with a decaying envelope: a statement ending."""
    t = np.arange(int(seconds * sr)) / sr
    f = np.linspace(f_start, f_end, len(t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, 12))
    env = np.clip(t / 0.05, 0, 1) * np.clip((seconds - t) / 0.08, 0, 1)
    return (0.2 * x * env).astype(np.float32)


@pytest.mark.parametrize("text,kind", [
    ("فایل را ذخیره کردید؟", "qyn"),
    ("کجا رفتی؟", "qwh"),
    ("واقعاً؟", "qecho"),
    ("نه بابا؟!", "qecho"),
    ("چای می‌خوری یا قهوه؟", "qalt"),
    ("می‌آیی یا نه؟", "qyn"),
    ("هوا خوب است، مگه نه؟", "qyn"),
    ("عالی شد!", "exc"),
    ("این خوب است.", "st"),
])
def test_sentence_plan_kinds(text, kind):
    assert I.sentence_plan(text)[0] == kind


def test_exclamation_intensity_from_marks_and_interjections():
    assert I.sentence_plan("وای، باورم نمی‌شود!")[1] > I.sentence_plan("باورم نمی‌شود!")[1]
    assert I.sentence_plan("عالی شد!!")[1] > I.sentence_plan("عالی شد!")[1]


def test_psola_v2_identity_and_pitch_and_time():
    x = _voiced()
    f0, hop = I.yin(x, SR)
    ts = np.arange(len(x)) / hop
    fr = np.arange(len(f0))
    f0s = np.where(np.interp(ts, fr, (f0 > 0).astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
    y, _ = I.psola_v2(x, SR, f0s, np.ones(len(x)))
    assert np.abs(y - x).max() < 1e-6                       # ratio = beta = gain = 1 reproduces x
    y2, _ = I.psola_v2(x, SR, f0s, np.ones(len(x)), beta=np.full(len(x), 1.25))
    assert abs(len(y2) / len(x) - 1.25) < 0.02               # time scale follows beta
    a = I.psola(x.astype(np.float64), SR, f0s, np.full(len(x), 2 ** (2 / 12)))
    b, _ = I.psola_v2(x, SR, f0s, np.full(len(x), 2 ** (2 / 12)))
    assert np.abs(a - b[:len(a)]).max() < 1e-5               # same marks as study C's PSOLA


def test_process_statement_untouched_and_strength_zero():
    x = _voiced()
    assert np.array_equal(I.process(x, SR, "این خوب است."), x)
    assert np.array_equal(I.process(x, SR, "این خوب است؟", strength=0.0), x)


def test_question_ends_higher_than_statement():
    x = _voiced()
    q = I.process(x, SR, "این خوب است؟")
    fq, _ = I.yin(q, SR)
    fs, _ = I.yin(x, SR)
    end = lambda f: np.median(f[np.flatnonzero(f > 0)[-10:]])  # noqa: E731
    assert 12 * np.log2(end(fq) / end(fs)) > 3.0
    assert len(q) > len(x)                                   # final syllable lengthened


def test_exclamation_louder_and_second_half_kept_up():
    x = _voiced()
    y = I.process(x, SR, "عالی است!")
    h = len(x) // 2
    rms = lambda v: float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))  # noqa: E731
    assert rms(y) > rms(x)
    assert rms(y[h:]) / rms(x[h:]) > rms(y[:h]) / rms(x[:h])


def test_paragraph_offsets():
    x = _voiced()
    first = I.process(x, SR, "این خوب است.", para_first=True)
    last = I.process(x, SR, "این خوب است.", para_last=True)
    assert np.max(np.abs(first)) > np.max(np.abs(x)) > np.max(np.abs(last))


def test_styles_are_only_named_for_known_kinds():
    assert set(I.STYLE_OF_KIND.values()) <= set(I.PROMPT_STYLES)
    assert "st" not in I.STYLE_OF_KIND and "qalt" not in I.STYLE_OF_KIND


def test_limiter_touches_only_peaks():
    rng = np.random.default_rng(0)
    x = (0.2 * rng.standard_normal(SR)).astype(np.float32)
    x[1000:1010] = 1.4                                       # a few hot samples
    y = I.limit(x, SR, 0.95)
    assert np.abs(y).max() <= 0.95 + 1e-6
    far = np.r_[0:500, 3000:SR]
    assert np.array_equal(y[far], x[far])                    # away from the peak: untouched
    quiet = 0.5 * x / np.abs(x).max()
    assert np.array_equal(I.limit(quiet, SR, 0.95), quiet)   # nothing above the ceiling: identity
