"""Consensus (min of Whisper and wav2vec2 CER) paired comparison of two bench tags,
Persian rows only, with a case-bootstrap CI.   python research/cons_stats.py BASE NEW"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[1] / "output" / "eval"


def load(t):
    h = {(r["case"], r["seed"]): r for r in json.loads((EVAL / f"flu_{t}" / "report.json").read_text(encoding="utf-8"))["rows"]}
    w = {(r["case"], r["seed"]): r for r in json.loads((EVAL / f"flu_{t}" / "report_w2v.json").read_text(encoding="utf-8"))["rows"]}
    return {k: min(h[k]["cer"], w[k]["cer"]) for k in w}


b, e = load(sys.argv[1]), load(sys.argv[2])
keys = [k for k in b if k in e]
a = np.array([b[k] for k in keys])
c = np.array([e[k] for k in keys])
by = collections.defaultdict(list)
for k, x, y in zip(keys, a, c):
    by[k[0]].append(y - x)
rng = np.random.default_rng(0)
cs = list(by)
bs = [np.mean([v for j in rng.integers(0, len(cs), len(cs)) for v in by[cs[j]]]) for _ in range(4000)]
print(f"cons {sys.argv[2]} vs {sys.argv[1]}: n={len(keys)} {a.mean():.4f} (fails {(a > .25).sum()}) -> "
      f"{c.mean():.4f} (fails {(c > .25).sum()})  diff {np.mean(c - a):+.4f} "
      f"CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  better/worse {(c < a).sum()}/{(c > a).sum()}")
