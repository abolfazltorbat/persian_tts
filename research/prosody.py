"""Study C — sentence-type prosody: measurements (no speech recogniser).

Per clip: an F0 track (librosa pYIN, 10 ms hop) and a frame-energy track over
the speech region, reduced to the numbers that separate a question from a
statement and an exclamation from narration:

  end_st     mean F0 of the last 150 ms of voicing, in semitones re the clip's
             median F0 (> 0: the clip ends high — a rise)
  end_slope  F0 slope over the last 300 ms of voicing, semitones / second
  range_st   F0 range, 95th - 5th percentile, semitones
  sd_st      F0 standard deviation, semitones
  f0_med     median F0, Hz
  e_range    frame-energy range over speech, 95th - 20th percentile, dB
  e_sd       frame-energy standard deviation over speech, dB
  peak_pos   where the F0 maximum sits, 0 = start of voicing, 1 = end

Runs in .venv-qwen (librosa). Library use: `measure(x, sr)`; CLI:

    .venv-qwen/bin/python research/prosody.py DIR_OR_WAV... [--out file.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HOP_S = 0.010


def f0_track(x: np.ndarray, sr: int, fmin: float = 65.0, fmax: float = 520.0):
    """-> (f0 Hz with NaN where unvoiced, frame times s). 16 kHz internally."""
    import librosa
    if sr != 16000:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    hop = int(HOP_S * sr)
    f0, vflag, _ = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=hop,
                                fill_na=np.nan)
    f0 = np.where(vflag, f0, np.nan)
    return f0, np.arange(len(f0)) * HOP_S


def energy_db(x: np.ndarray, sr: int, win_s: float = 0.025) -> np.ndarray:
    hop, win = int(HOP_S * sr), int(win_s * sr)
    n = max(1, 1 + (len(x) - win) // hop)
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, max(0, win - len(x)))), win)[::hop][:n]
    return 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-10)


def _octave_clean(st: np.ndarray) -> np.ndarray:
    """pYIN still halves/doubles on creak and breath: a voiced frame more than
    9 st from its 7-frame neighbourhood median is dropped."""
    out = st.copy()
    v = ~np.isnan(st)
    idx = np.flatnonzero(v)
    for k, i in enumerate(idx):
        nb = st[idx[max(0, k - 3): k + 4]]
        if abs(st[i] - np.nanmedian(nb)) > 9:
            out[i] = np.nan
    return out


def measure(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f0, t = f0_track(x, sr)
    st = 12 * np.log2(f0 / np.nanmedian(f0)) if np.any(~np.isnan(f0)) else f0
    st = _octave_clean(st)
    v = np.flatnonzero(~np.isnan(st))
    e = energy_db(x, sr)
    sp = e > (np.percentile(e, 95) - 35)                    # speech frames
    out = {"dur": round(len(x) / sr, 3), "voiced_s": round(len(v) * HOP_S, 3)}
    if len(v) < 20:
        out.update(end_st=np.nan, end_slope=np.nan, range_st=np.nan, sd_st=np.nan,
                   f0_med=np.nan, peak_pos=np.nan)
    else:
        sv = st[v]
        last = v[-15:]
        tail = v[v >= v[-1] - int(0.30 / HOP_S)]
        slope = np.polyfit(t[tail], st[tail], 1)[0] if len(tail) >= 8 else np.nan
        out.update(end_st=float(np.mean(st[last])), end_slope=float(slope),
                   range_st=float(np.percentile(sv, 95) - np.percentile(sv, 5)),
                   sd_st=float(np.std(sv)), f0_med=float(np.nanmedian(f0)),
                   peak_pos=float((v[np.argmax(sv)] - v[0]) / max(1, v[-1] - v[0])))
    es = e[sp] if sp.sum() >= 10 else e
    out.update(e_range=float(np.percentile(es, 95) - np.percentile(es, 20)), e_sd=float(np.std(es)))
    return {k: (round(val, 4) if isinstance(val, float) and not np.isnan(val) else val)
            for k, val in out.items()}


def contour(x: np.ndarray, sr: int, n: int = 20) -> np.ndarray:
    """Time-normalised F0 contour (semitones re median) over the voiced span,
    n points — for averaging the shape of a sentence type."""
    f0, _ = f0_track(x, sr)
    st = _octave_clean(12 * np.log2(f0 / np.nanmedian(f0)))
    v = np.flatnonzero(~np.isnan(st))
    if len(v) < 20:
        return np.full(n, np.nan)
    pos = (v - v[0]) / max(1, v[-1] - v[0])
    edges = np.linspace(0, 1, n + 1)
    return np.array([np.nanmean(st[v][(pos >= a) & (pos <= b)]) if np.any((pos >= a) & (pos <= b)) else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])


def main(argv: list[str]) -> int:
    import soundfile as sf
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        out, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    wavs = []
    for a in argv:
        p = Path(a)
        wavs += sorted(p.glob("*.wav")) if p.is_dir() else [p]
    rows = []
    for w in wavs:
        x, sr = sf.read(str(w), dtype="float32")
        r = {"wav": str(w), **measure(x, sr)}
        rows.append(r)
        if out is None:
            print(json.dumps(r, ensure_ascii=False))
    if out:
        Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print("wrote", len(rows), "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
