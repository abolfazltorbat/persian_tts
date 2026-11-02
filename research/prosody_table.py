"""Study C — per-type prosody table: real speech vs one or more systems.

    .venv-qwen/bin/python research/prosody_table.py base=output/research/prosody/c_base.jsonl \
        int=output/research/int_v1/prosody.jsonl [--split dev|test|all]

Rows: sentence type; columns: end_st / end_slope / range_st / e_range (mean
over renders). `qyn - st` is the PAIRED gap: the same words with «؟» vs «.»,
same voice and seed — 0 for the unmodified engine by construction.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = Path.home() / ".cache/persian_tts_ft/data_gptinformal_types"
WH = re.compile(r"(^|\s)(چی|چه|چرا|کجا|کی|چطور|چطوری|چگونه|چند|چقدر|کدوم|کدام|کیه|چیه|چیکار|چکار)")
CAT_OF = {"q": "qyn", "x": "exc", "s": "st"}
METRICS = ("end_st", "end_slope", "range_st", "e_range")


def load_set_cats() -> dict:
    out = {}
    for line in (ROOT / "research/sets/prosody_set.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            cid, _l, cat, _t = line.split("\t", 3)
            out[cid] = cat
    return out


def num(cid: str) -> int:
    m = re.search(r"\d+", cid)
    return int(m.group()) if m else 0


def load_sys(path: str, split: str) -> dict:
    cats = load_set_cats()
    rows = collections.defaultdict(dict)        # cat -> {(case, voice, seed): row}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        stem = Path(r["wav"]).stem
        case, voice, seed = stem.split("__")[:3]
        if case not in cats:
            continue
        if split == "dev" and num(case) % 2 == 0 or split == "test" and num(case) % 2 == 1:
            continue
        rows[cats[case]][(case, voice, seed)] = r
    return rows


def load_real() -> dict:
    meta = {}
    for line in (REAL / "clips.tsv").read_text(encoding="utf-8").splitlines():
        cid, k, _d, t = line.split("\t", 3)
        cat = CAT_OF[k]
        if k == "q" and WH.search(t):
            cat = "qwh"
        meta[cid] = cat
    rows = collections.defaultdict(dict)
    p = ROOT / "output/research/prosody/real_types.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cid = Path(r["wav"]).stem
        rows[meta[cid]][cid] = r
    return rows


def mean(rows: dict, m: str) -> float:
    v = [r[m] for r in rows.values() if isinstance(r.get(m), (int, float)) and r[m] == r[m]]
    return float(np.mean(v)) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="name=prosody.jsonl")
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    systems = {s.split("=", 1)[0]: load_sys(s.split("=", 1)[1], a.split) for s in a.systems}
    real = load_real()
    out = {"split": a.split, "real": {}, "sys": {}}
    cats = ["st", "qyn", "qshort", "qwh", "exc"]
    print(f"split={a.split}")
    print(f"{'type':7s} {'system':8s} {'n':>4s} " + " ".join(f"{m:>9s}" for m in METRICS))
    for c in cats:
        if c in real:
            vals = {m: mean(real[c], m) for m in METRICS}
            out["real"][c] = {"n": len(real[c]), **vals}
            print(f"{c:7s} {'REAL':8s} {len(real[c]):4d} " + " ".join(f"{vals[m]:+9.2f}" for m in METRICS))
        for name, rows in systems.items():
            if c not in rows:
                continue
            vals = {m: mean(rows[c], m) for m in METRICS}
            out["sys"].setdefault(name, {})[c] = {"n": len(rows[c]), **vals}
            print(f"{c:7s} {name:8s} {len(rows[c]):4d} " + " ".join(f"{vals[m]:+9.2f}" for m in METRICS))
    # paired question-statement gap
    print("\npaired gap qyn - st (same words, voice, seed): mean end_st / end_slope, 95% CI over sentence pairs")
    for name, rows in systems.items():
        q, s = rows.get("qyn", {}), rows.get("st", {})
        by = collections.defaultdict(list)
        for (case, v, sd), r in q.items():
            k = (case[:-1] + "s", v, sd)
            if k in s and r["end_st"] == r["end_st"] and s[k]["end_st"] == s[k]["end_st"]:
                by[case].append((r["end_st"] - s[k]["end_st"], r["end_slope"] - s[k]["end_slope"]))
        if not by:
            continue
        d = np.array([x for v in by.values() for x in v])
        rng = np.random.default_rng(0)
        keys = list(by)
        bs = [np.mean([x[0] for j in rng.integers(0, len(keys), len(keys)) for x in by[keys[j]]]) for _ in range(2000)]
        out["sys"].setdefault(name, {})["gap"] = {"n": len(d), "end_st": float(d[:, 0].mean()),
                                                   "end_slope": float(np.nanmean(d[:, 1])),
                                                   "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print(f"  {name:8s} n={len(d):3d} end_st {d[:, 0].mean():+.2f} st [{np.percentile(bs, 2.5):+.2f}, "
              f"{np.percentile(bs, 97.5):+.2f}]  end_slope {np.nanmean(d[:, 1]):+.1f} st/s")
    if "qyn" in real and "st" in real:
        print(f"  REAL (unpaired) end_st {mean(real['qyn'], 'end_st') - mean(real['st'], 'end_st'):+.2f} st")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
