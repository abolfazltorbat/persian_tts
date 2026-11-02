"""Voice-referenced spectral correction ("clarity", research/PROGRESS.md study D).

The model renders some voices with a different long-term spectrum than their
reference clip: the default male voice comes out 10-11 dB weaker at 2.5-6.3 kHz
(presence, consonant bursts and frication) and ~7 dB boxier at 0.6-1.2 kHz than
the clip it imitates — "muffled". The correction is the smoothed difference of
the two long-term average spectra (LTAS, speech frames only), level-neutral over
300-3000 Hz, clamped, and applied as one linear-phase FIR to the finished audio.
Bands where the reference itself has no energy (a clip recorded at 16 kHz) are
left alone, so a band limit is never copied.

numpy + scipy only. `curve(ref, tts, sr)` -> (freqs, gain dB); `apply(x, sr, curve)`.
"""
from __future__ import annotations

import numpy as np

NFFT = 2048
MAX_BOOST_DB = 6.0
MAX_CUT_DB = 6.0
F_LO, F_HI = 70.0, 10000.0


def ltas_db(x: np.ndarray, sr: int, floor_db: float = 30.0) -> np.ndarray:
    """Mean power spectrum (dB) over the speech frames of x (within `floor_db`
    of the loudest frame), Hann frames of NFFT, half overlap."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[:: NFFT // 2]
    e = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)
    fr = fr[e > e.max() - floor_db]
    S = (np.abs(np.fft.rfft(fr * np.hanning(NFFT), axis=1)) ** 2).mean(axis=0)
    return 10 * np.log10(S + 1e-20)


def _octave_smooth(f: np.ndarray, s: np.ndarray, frac: float = 1 / 3) -> np.ndarray:
    out = np.empty_like(s)
    for i, fc in enumerate(f):
        if fc <= 0:
            out[i] = s[i]
            continue
        lo, hi = fc * 2 ** (-frac / 2), fc * 2 ** (frac / 2)
        m = (f >= lo) & (f <= hi)
        out[i] = 10 * np.log10(np.mean(10 ** (s[m] / 10))) if m.any() else s[i]
    return out


def curve(ref: np.ndarray | list, tts: np.ndarray | list, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Correction (freqs Hz, gain dB) that moves `tts`'s LTAS toward `ref`'s.
    Each argument is one signal or a list of signals (their LTAS are averaged
    in the power domain)."""
    def avg(xs):
        xs = xs if isinstance(xs, (list, tuple)) else [xs]
        return 10 * np.log10(np.mean([10 ** (ltas_db(x, sr) / 10) for x in xs], axis=0))
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    R = _octave_smooth(f, avg(ref))
    T = _octave_smooth(f, avg(tts))
    d = R - T
    mid = (f >= 300) & (f <= 3000)
    d -= np.mean(d[mid])
    # never copy a reference's band limit or its rumble: only where the clip
    # has energy (no more than 35 dB below its 300-3000 Hz level) and inside F_LO..F_HI
    ok = (R > np.mean(R[mid]) - 35) & (f >= F_LO) & (f <= F_HI)
    d = np.where(ok, d, 0.0)
    d = np.clip(d, -MAX_CUT_DB, MAX_BOOST_DB)
    k = np.hanning(17) / np.hanning(17).sum()          # ~100 Hz: no sharp edges at the clamps
    d = np.convolve(np.pad(d, 8, mode="edge"), k, mode="valid")
    return f, d


def fir(f: np.ndarray, gain_db: np.ndarray, sr: int, strength: float = 1.0, taps: int = 513) -> np.ndarray:
    from scipy.signal import firwin2
    g = 10 ** (strength * gain_db / 20)
    ff = np.clip(f / (sr / 2), 0, 1)
    ff[0], ff[-1] = 0.0, 1.0
    return firwin2(taps, ff, g)


def apply(x: np.ndarray, sr: int, h: np.ndarray) -> np.ndarray:
    """Linear-phase FIR `h` (odd length) applied with no delay; peak-guarded."""
    from scipy.signal import fftconvolve
    x = np.asarray(x, dtype=np.float32)
    if not len(x) or h is None or len(h) < 3:
        return x
    y = fftconvolve(x.astype(np.float64), h, mode="full")[len(h) // 2: len(h) // 2 + len(x)]
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    if pk > 0.99:
        y *= 0.99 / pk
    return y.astype(np.float32)
