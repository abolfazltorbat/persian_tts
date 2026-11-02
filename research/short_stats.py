"""Paired consensus statistics for short-sentence render directories (review points 3/5):
    python research/short_stats.py BASE_DIR NEW_DIR [--set research/sets/short_test.tsv]
Each directory holds <case>__<voice>__s<seed>.wav plus whisper.jsonl (research/punct_judge.py) and
w2v_hyps.jsonl (research/w2v_judge.py). Persian rows only. Consensus CER = min(Whisper, wav2vec2);
a render FAILS when its consensus CER > 0.25. Differences per voice and pooled, with a 95%
case-bootstrap CI; UTMOS / speaker similarity changes when utmos.jsonl / spk.jsonl exist."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "research"))


def load(d: Path, refs: dict) -> dict:
    from scoring import score
    wh = {json.loads(l)["name"]: json.loads(l) for l in (d / "whisper.jsonl").read_text().splitlines() if l.strip()}
    w2 = {json.loads(l)["id"] + ".wav": json.loads(l)["hyp_w2v"] for l in (d / "w2v_hyps.jsonl").read_text().splitlines() if l.strip()}
    ut = {}
    for f, k in (("utmos.jsonl", "utmos"), ("spk.jsonl", "spk")):
        if (d / f).exists():
            for l in (d / f).read_text().splitlines():
                r = json.loads(l)
                ut.setdefault(r["wav"], {})[k] = r[k]
    rows = {}
    for name, r in wh.items():
        case = r["case"]
        if case not in refs or refs[case]["lang"] != "fa" or name not in w2:
            continue
        c_w2 = score(w2[name], refs[case]["ref"], "fa")["cer"]
        rows[name] = {"case": case, "voice": name.split("__")[1], "cons": min(r["cer"], c_w2), **ut.get(name, {})}
    return rows


def boot(by_case: dict, n=4000):
    rng = np.random.default_rng(0)
    cs = list(by_case)
    bs = [np.mean([v for j in rng.integers(0, len(cs), len(cs)) for v in by_case[cs[j]]]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    from core import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--set", default=str(ROOT / "research/sets/short_test.tsv"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    refs = {c["id"]: c for c in load_set(a.set)}
    b, n = load(Path(a.base), refs), load(Path(a.new), refs)
    keys = sorted(k for k in b if k in n)
    voices = sorted({b[k]["voice"] for k in keys})
    out = {}
    for v in voices + ["pooled"]:
        ks = [k for k in keys if v == "pooled" or b[k]["voice"] == v]
        by = collections.defaultdict(list)
        for k in ks:
            by[b[k]["case"]].append(n[k]["cons"] - b[k]["cons"])
        lo, hi = boot(by)
        res = {"n": len(ks), "cons_base": np.mean([b[k]["cons"] for k in ks]), "cons_new": np.mean([n[k]["cons"] for k in ks]),
               "fails_base": int(sum(b[k]["cons"] > 0.25 for k in ks)), "fails_new": int(sum(n[k]["cons"] > 0.25 for k in ks)),
               "d_cons": [float(np.mean([n[k]["cons"] - b[k]["cons"] for k in ks])), lo, hi]}
        for m in ("utmos", "spk"):
            d = [n[k][m] - b[k][m] for k in ks if m in n[k] and m in b[k]]
            if d:
                res[f"d_{m}"] = float(np.mean(d))
        out[v] = res
        print(f"{v:24s} n={res['n']:4d} cons {res['cons_base']:.3f} -> {res['cons_new']:.3f} "
              f"[{res['d_cons'][0]:+.3f} ({lo:+.3f}, {hi:+.3f})] fails {res['fails_base']} -> {res['fails_new']}"
              + "".join(f"  d{m} {res[f'd_{m}']:+.3f}" for m in ("utmos", "spk") if f"d_{m}" in res))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
