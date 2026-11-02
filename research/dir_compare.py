"""Study D — compare render directories judged by research/dir_eval.sh.

    python research/dir_compare.py BASE_DIR NEW_DIR [NEW_DIR ...] [--cats qyn,qwh,...] [--split test|dev|all]
           [--json out.json]

Rows are joined by file name (<case>__<voice>__s<seed>.wav). Per category:
consensus CER (min of Whisper and wav2vec2, Persian), failed renders (> .25),
Whisper "?" rate (does the recogniser hear a question?), UTMOS, speaker
similarity, and the prosody measures of research/prosody.py. Paired
differences against BASE get a case-bootstrap 95 % CI. Minimal pairs (pNNq vs
pNNs, same voice and seed) give the question-statement final-F0 gap.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "research"))
SETS = [ROOT / "research/sets/prosody_set.tsv", ROOT / "research/sets/prosody_set2.tsv"]


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def load_dir(d: Path, refs: dict) -> dict[str, dict]:
    from scoring import score
    rows: dict[str, dict] = collections.defaultdict(dict)
    for r in _jsonl(d / "whisper.jsonl"):
        rows[r["name"]].update(cer_wh=r.get("cer"), mark=r["mark"])
    for r in _jsonl(d / "w2v_hyps.jsonl"):
        case = r["id"].split("__")[0]
        if case in refs:
            rows[r["id"] + ".wav"]["cer_w2v"] = score(r["hyp_w2v"], refs[case]["ref"], "fa")["cer"]
    for r in _jsonl(d / "utmos.jsonl"):
        rows[r["wav"]]["utmos"] = r["utmos"]
    for r in _jsonl(d / "spk.jsonl"):
        rows[r["wav"]]["spk"] = r["spk"]
    for r in _jsonl(d / "dnsmos.jsonl"):
        rows[r["wav"]]["sig"] = r["sig"]
    import soundfile as sf
    for name in list(rows):
        w = d / name
        if w.exists():
            x, sr = sf.read(str(w), dtype="float32")
            rows[name]["half_db"] = half_db(x)
            rows[name]["level_db"] = level_db(x)
    for r in _jsonl(d / "prosody.jsonl"):
        rows[Path(r["wav"]).name].update({k: r[k] for k in ("end_st", "end_slope", "range_st", "e_range",
                                                            "f0_med", "dur") if k in r})
    for name, r in rows.items():
        case, voice, seed = Path(name).stem.split("__")
        r.update(case=case, voice=voice, seed=seed, cat=refs.get(case, {}).get("cat", "?"))
        if r.get("cer_wh") is not None and r.get("cer_w2v") is not None:
            r["cons"] = min(r["cer_wh"], r["cer_w2v"])
        if "mark" in r:
            r["q"] = float(r["mark"] == "?")
        if isinstance(r.get("f0_med"), float) and r["f0_med"] > 0:
            r["f0_st"] = 12 * np.log2(r["f0_med"])
    return rows


def _env_db(x):
    return 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, np.ones(480) / 480, mode="same") + 1e-12)


def half_db(x) -> float:
    """second-half minus first-half speech energy (dB): how much a sentence decays"""
    e = _env_db(x)
    sp = np.flatnonzero(e > e.max() - 35)
    if len(sp) < 100:
        return float("nan")
    h = sp[len(sp) // 2]
    return float(10 * np.log10(np.mean(x[h:sp[-1]] ** 2) + 1e-12) - 10 * np.log10(np.mean(x[sp[0]:h] ** 2) + 1e-12))


def level_db(x) -> float:
    e = _env_db(x)
    sp = e > e.max() - 35
    return float(10 * np.log10(np.mean(x[sp] ** 2) + 1e-12))


def split_ok(case: str, split: str) -> bool:
    if split == "all":
        return True
    num = int("".join(ch for ch in case if ch.isdigit()) or 0)
    return (num % 2 == 0) == (split == "test")


def boot(diffs_by_case: dict, n: int = 3000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    cs = list(diffs_by_case)
    if not cs:
        return (np.nan, np.nan)
    bs = [np.mean([v for j in rng.integers(0, len(cs), len(cs)) for v in diffs_by_case[cs[j]]]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


METRICS = ["cons", "cer_wh", "q", "utmos", "sig", "spk", "end_st", "end_slope", "range_st", "e_range", "dur",
           "half_db", "level_db", "f0_med"]


def summarize(rows: dict, base: dict | None, cats: list[str], split: str) -> dict:
    out = {}
    for cat in cats:
        sel = {k: r for k, r in rows.items() if (cat == "all" or r["cat"] == cat) and split_ok(r["case"], split)}
        if not sel:
            continue
        res = {"n": len(sel)}
        for m in METRICS:
            v = [r[m] for r in sel.values() if r.get(m) is not None and not (isinstance(r[m], float) and np.isnan(r[m]))]
            if v:
                res[m] = round(float(np.mean(v)), 4)
        res["fails"] = int(sum(r.get("cons", 0) > 0.25 for r in sel.values()))
        if base is not None:
            for m in ("cons", "q", "utmos", "sig", "spk", "end_st", "range_st", "e_range", "half_db", "level_db", "f0_st"):
                by = collections.defaultdict(list)
                for k, r in sel.items():
                    b = base.get(k)
                    if b and r.get(m) is not None and b.get(m) is not None and not np.isnan(r[m]) and not np.isnan(b[m]):
                        by[r["case"]].append(r[m] - b[m])
                if by:
                    d = [x for v in by.values() for x in v]
                    lo, hi = boot(by)
                    res[f"d_{m}"] = [round(float(np.mean(d)), 4), round(lo, 4), round(hi, 4)]
        out[cat] = res
    # minimal pairs: question - statement final F0 (same voice, seed)
    gaps = collections.defaultdict(list)
    for k, r in rows.items():
        if r["case"].startswith("p") and r["case"].endswith("q") and split_ok(r["case"], split):
            s = rows.get(k.replace(r["case"], r["case"][:-1] + "s", 1))
            ok = lambda v: isinstance(v, (int, float)) and not np.isnan(v)  # noqa: E731
            if s and ok(r.get("end_st")) and ok(s.get("end_st")):
                gaps[r["case"]].append(r["end_st"] - s["end_st"])
    if gaps:
        d = [x for v in gaps.values() for x in v]
        out["pair_gap"] = [round(float(np.mean(d)), 3), *[round(x, 3) for x in boot(gaps)]]
    return out


def main() -> int:
    from core import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--cats", default="all,st,qyn,qshort,qwh,qecho,qalt,qtag,exc,dyn")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    refs = {}
    for s in SETS:
        refs.update({c["id"]: c for c in load_set(s)})
    dirs = [Path(d) for d in a.dirs]
    base = load_dir(dirs[0], refs)
    cats = a.cats.split(",")
    report = {}
    for d in dirs:
        rows = load_dir(d, refs) if d != dirs[0] else base
        report[str(d)] = summarize(rows, base if d != dirs[0] else None, cats, a.split)
    for d, res in report.items():
        print(f"\n## {d}  (split {a.split})")
        for cat, r in res.items():
            if cat == "pair_gap":
                print(f"  pair gap q-s end F0: {r[0]:+.2f} st [{r[1]:+.2f}, {r[2]:+.2f}]")
                continue
            line = f"  {cat:7s} n={r['n']:3d}"
            for m, f in (("cons", ".3f"), ("q", ".2f"), ("utmos", ".2f"), ("sig", ".2f"), ("spk", ".3f"),
                         ("end_st", "+.2f"), ("range_st", ".2f"), ("half_db", "+.1f")):
                if m in r:
                    line += f" {m}={format(r[m], f)}"
            line += f" fails={r['fails']}"
            for m in ("cons", "q", "utmos", "sig", "spk", "f0_st", "level_db"):
                if f"d_{m}" in r:
                    x = r[f"d_{m}"]
                    line += f" | d{m} {x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]"
            print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
