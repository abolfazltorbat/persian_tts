"""Paired statistics between fluency-bench tags (same case × voice × seed).

    python research/paired_stats.py BASE TAG [TAG ...] [--cats a,b] [--metrics cer,utmos,spk]

For each metric: means, the paired mean difference with a 95 % bootstrap CI
that resamples CASES (renders of one case are not independent), a Wilcoxon
signed-rank p over render pairs, and for CER the failure count (> 0.25, the
bench's definition). Writes output/research/paired_<BASE>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[1] / "output" / "eval"
OUT = Path(__file__).resolve().parents[1] / "output" / "research"
LOWER_BETTER = {"cer", "wer", "join_sil_ms", "reset_st", "fall_mid_st", "jump_db", "rate_cv"}


JUDGE = "whisper"


def load(tag: str) -> dict:
    name = "report_w2v.json" if JUDGE == "w2v" else "report.json"
    rows = json.loads((EVAL / f"flu_{tag}" / name).read_text(encoding="utf-8"))["rows"]
    return {(r["case"], r["voice"], r["seed"]): r for r in rows}


def boot_ci(diffs_by_case: dict, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cases = list(diffs_by_case)
    if not cases:
        return float("nan"), float("nan")
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(cases), len(cases))
        vals = [v for i in pick for v in diffs_by_case[cases[i]]]
        means.append(np.mean(vals))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(base: dict, other: dict, metrics: list[str], cats: set | None) -> dict:
    from scipy.stats import wilcoxon

    keys = [k for k in base if k in other and (cats is None or base[k]["cat"] in cats)]
    out = {"pairs": len(keys), "cases": len({k[0] for k in keys})}
    for m in metrics:
        ks = [k for k in keys if base[k].get(m) is not None and other[k].get(m) is not None]
        if not ks:
            continue
        a = np.array([base[k][m] for k in ks], float)
        b = np.array([other[k][m] for k in ks], float)
        by_case = defaultdict(list)
        for k, x, y in zip(ks, a, b):
            by_case[k[0]].append(y - x)
        lo, hi = boot_ci(by_case)
        try:
            p = float(wilcoxon(b, a).pvalue) if np.any(a != b) else 1.0
        except ValueError:
            p = float("nan")
        d = {"n": len(ks), "base": round(float(a.mean()), 4), "new": round(float(b.mean()), 4),
             "diff": round(float((b - a).mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
             "p_wilcoxon": round(p, 5)}
        if m == "cer":
            d["fails_base"] = int((a > 0.25).sum())
            d["fails_new"] = int((b > 0.25).sum())
            d["better"] = int((b < a - 1e-9).sum())
            d["worse"] = int((b > a + 1e-9).sum())
        out[m] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--cats", default="")
    ap.add_argument("--judge", default="whisper", choices=["whisper", "w2v"])
    ap.add_argument("--metrics", default="cer,utmos,spk,join_sil_ms,reset_st,fall_mid_st,rate_cv,audio_s,synth_s")
    a = ap.parse_args()
    global JUDGE
    JUDGE = a.judge
    cats = set(a.cats.split(",")) if a.cats else None
    base = load(a.base)
    res = {}
    for t in a.tags:
        res[t] = compare(base, load(t), a.metrics.split(","), cats)
        r = res[t]
        print(f"== {t} vs {a.base}: {r['pairs']} pairs, {r['cases']} cases" + (f" cats={a.cats}" if cats else ""))
        for m, d in r.items():
            if not isinstance(d, dict):
                continue
            extra = f"  fails {d['fails_base']}->{d['fails_new']}  better/worse {d['better']}/{d['worse']}" if m == "cer" else ""
            print(f"   {m:<12} {d['base']:.4f} -> {d['new']:.4f}  diff {d['diff']:+.4f}  CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_wilcoxon']:.4g}{extra}")
    name = f"paired_{a.base}" + (f"_{a.cats.replace(',', '-')}" if cats else "") + ("_w2v" if a.judge == "w2v" else "") + ".json"
    (OUT / name).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
