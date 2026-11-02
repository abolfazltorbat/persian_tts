"""Study C — wh-questions: pitch ON the question word and at the end.

For each wh-question render: map the wh-word's text position (intonation.wh_position)
onto the pYIN voiced span and report the mean F0 (semitones re the clip median)
in +-80 ms around it ("acc"), plus end_st. Real GPTInformal wh-questions too.

    .venv-qwen/bin/python research/wh_check.py base=output/eval/flu_c_base new=output/research/int_v3/wav [--split dev]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
import intonation as I  # noqa: E402
from prosody import f0_track, _octave_clean  # noqa: E402
from prosody_table import REAL, WH, num  # noqa: E402


def acc_end(x, sr, pos):
    f0, t = f0_track(x, sr)
    st = _octave_clean(12 * np.log2(f0 / np.nanmedian(f0)))
    v = np.flatnonzero(~np.isnan(st))
    if len(v) < 20 or pos is None:
        return np.nan, np.nan
    c = v[0] + pos * (v[-1] - v[0])
    win = v[np.abs(v - c) <= 8]
    return (float(np.mean(st[win])) if len(win) else np.nan), float(np.mean(st[v[-15:]]))


def main() -> int:
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--split", default="all")
    a = ap.parse_args()
    texts = {}
    for line in (ROOT / "research/sets/prosody_set.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            cid, _l, cat, t = line.split("\t", 3)
            if cat == "qwh":
                texts[cid] = t
    for spec in a.systems:
        name, d = spec.split("=", 1)
        acc, end = [], []
        for w in sorted(Path(d).glob("w*.wav")):
            cid = w.stem.split("__")[0]
            if cid not in texts or a.split == "dev" and num(cid) % 2 == 0 or a.split == "test" and num(cid) % 2 == 1:
                continue
            x, sr = sf.read(str(w), dtype="float32")
            p, e = acc_end(x, sr, I.wh_position(texts[cid]))
            acc.append(p)
            end.append(e)
        print(f"{name:6s} n={len(acc):3d} wh-word accent {np.nanmean(acc):+.2f} st   end {np.nanmean(end):+.2f} st")
    acc, end = [], []
    for line in (REAL / "clips.tsv").read_text(encoding="utf-8").splitlines():
        cid, k, _d, t = line.split("\t", 3)
        if k == "q" and WH.search(t):
            x, sr = sf.read(str(REAL / "wav" / f"{cid}.wav"), dtype="float32")
            p, e = acc_end(x, sr, I.wh_position(t))
            acc.append(p)
            end.append(e)
    print(f"REAL   n={len(acc):3d} wh-word accent {np.nanmean(acc):+.2f} st   end {np.nanmean(end):+.2f} st")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
