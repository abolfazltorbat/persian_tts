"""Tables for prompt probes judged by both recognisers (rows_judged.jsonl).

For every voice × prompt kind: mean CER and failure rate (> .3) under Whisper,
wav2vec2 and the consensus label, plus the paired difference against the
baseline kind (same case, unit and take seed) with a 95 % bootstrap CI that
resamples CASES. Several probe tags can be pooled (same seeds → paired).

    python research/probe_table.py prompt_ext3 prompt_ext4 --base ref
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output" / "research"
LABELS = ("cer_whisper", "cer_w2v", "cer_cons")


def boot(diffs: dict, n: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    cases = list(diffs)
    vals = [np.mean([v for i in rng.integers(0, len(cases), len(cases)) for v in diffs[cases[i]]])
            for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--base", default="ref")
    ap.add_argument("--label", default="cer_cons")
    a = ap.parse_args()
    rows = []
    for t in a.tags:
        p = OUT / t / "rows_judged.jsonl"
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[(r["voice"].split("/")[-1], r["kind"])][(r["case"], r["unit"], r["take"])] = r
    out = {}
    for (voice, kind), d in sorted(by.items()):
        base = by.get((voice, a.base), {})
        line = {"n": len(d)}
        for lab in LABELS:
            v = [x[lab] for x in d.values()]
            line[lab] = (round(st.mean(v), 3), round(sum(x > 0.3 for x in v) / len(v), 3))
        if kind != a.base and base:
            keys = [k for k in d if k in base]
            diffs = defaultdict(list)
            for k in keys:
                diffs[k[0]].append(d[k][a.label] - base[k][a.label])
            dm = float(np.mean([x for v in diffs.values() for x in v]))
            lo, hi = boot(diffs)
            fb = sum(base[k][a.label] > 0.3 for k in keys)
            fn = sum(d[k][a.label] > 0.3 for k in keys)
            line["vs_base"] = {"pairs": len(keys), "diff": round(dm, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "fails": [fb, fn]}
        out[f"{voice}|{kind}"] = line
        vb = line.get("vs_base")
        extra = (f"  Δ{a.label[4:]} {vb['diff']:+.3f} [{vb['ci95'][0]:+.3f},{vb['ci95'][1]:+.3f}] "
                 f"fails {vb['fails'][0]}→{vb['fails'][1]}") if vb else ""
        print(f"{voice:24s} {kind:14s} n={line['n']:3d} " +
              "  ".join(f"{lab[4:]}={line[lab][0]:.3f}/{line[lab][1]:.2f}" for lab in LABELS) + extra)
    name = "probe_table_" + "_".join(t.replace("prompt_", "") for t in a.tags) + ".json"
    (OUT / name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
