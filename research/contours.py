"""Study C — mean time-normalised F0 contours per sentence type, for the paper
figure: real speech (GPTInformal) vs renders of one or more bench tags, plus
the F0 tracks of one minimal pair (statement / yes-no question).

    .venv-qwen/bin/python research/contours.py base=c_base ours=c_int --split test \
        --pair p02 --pair-voice male_hello_clean --out output/research/prosody/contours.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from prosody import contour, f0_track, _octave_clean  # noqa: E402
from prosody_table import load_set_cats, num, CAT_OF, WH, REAL  # noqa: E402

N = 24


def boot(m: np.ndarray, rng) -> tuple[list, list]:
    ok = ~np.all(np.isnan(m), axis=1)
    m = m[ok]
    bs = np.array([np.nanmean(m[rng.integers(0, len(m), len(m))], axis=0) for _ in range(1000)])
    return np.nanpercentile(bs, 2.5, axis=0).tolist(), np.nanpercentile(bs, 97.5, axis=0).tolist()


def main() -> int:
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="name=bench_tag")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pair", default="p02")
    ap.add_argument("--pair-voice", default="male_hello_clean")
    ap.add_argument("--pair-seed", default="s1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cats = load_set_cats()
    rng = np.random.default_rng(0)
    out = {"n_points": N, "split": a.split, "mean": {}, "lo": {}, "hi": {}, "n": {}, "pair": {}}
    groups = collections.defaultdict(list)
    for spec in a.systems:
        name, tag = spec.split("=", 1)
        for w in sorted((ROOT / "output/eval" / f"flu_{tag}").glob("*.wav")):
            case = w.stem.split("__")[0]
            if case not in cats:
                continue
            if a.split == "test" and num(case) % 2 == 1 or a.split == "dev" and num(case) % 2 == 0:
                continue
            x, sr = sf.read(str(w), dtype="float32")
            groups[(name, cats[case])].append(contour(x, sr, N))
    real_meta = {}
    for line in (REAL / "clips.tsv").read_text(encoding="utf-8").splitlines():
        cid, k, _d, t = line.split("\t", 3)
        real_meta[cid] = "qwh" if k == "q" and WH.search(t) else CAT_OF[k]
    for cid, cat in real_meta.items():
        x, sr = sf.read(str(REAL / "wav" / f"{cid}.wav"), dtype="float32")
        groups[("real", cat)].append(contour(x, sr, N))
    for (name, cat), cs in groups.items():
        m = np.array(cs)
        key = f"{name}|{cat}"
        out["mean"][key] = np.nanmean(m, axis=0).tolist()
        out["lo"][key], out["hi"][key] = boot(m, rng)
        out["n"][key] = int(len(m))
    # one minimal pair, absolute F0 tracks (Hz) over time
    for spec in a.systems:
        name, tag = spec.split("=", 1)
        for suf in ("s", "q"):
            w = ROOT / "output/eval" / f"flu_{tag}" / f"{a.pair}{suf}__{a.pair_voice}__{a.pair_seed}.wav"
            if not w.exists():
                continue
            x, sr = sf.read(str(w), dtype="float32")
            f0, t = f0_track(x, sr)
            out["pair"][f"{name}|{suf}"] = {"t": t.tolist(), "f0": [None if np.isnan(v) else float(v) for v in f0]}
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print("wrote", a.out, {k: v for k, v in out["n"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
