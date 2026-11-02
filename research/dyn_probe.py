"""Study D — does prosody CHANGE through a text? Multi-sentence texts rendered
with several engine configurations; every sentence's audio as joined is traced
(engine.join_trace) and measured.

    python research/dyn_probe.py --out dyn1 --configs prod,v2,v2s [--voices ...] [--seeds 1,2]
       [--set research/sets/prosody_set2.tsv --cats dyn]

Per sentence: written type, speech level (dB), F0 median (Hz), F0 range (st),
final F0 (st re the sentence median, last 150 ms of voicing). Per text: the
spread of sentence level and register across the text, and the contrasts
question/exclamation vs statement. Full renders are kept for listening:
output/research/<out>/<config>/<case>__<voice>__s<seed>.wav
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))

BASE = {"prompt_ext_dir": str(ROOT / "output/research/voice_ext_m"), "prompt_ext_en": False,
        "max_attempts": 1}
CONFIGS = {
    "none": {**BASE, "intonation": 0.0},
    "prod": {**BASE, "intonation": 1.0, "intonation.VERSION": 1},        # the shipped study C rule
    "v2": {**BASE, "intonation": 1.0, "intonation.VERSION": 2},
    "v2s": {**BASE, "intonation": 1.0, "intonation.VERSION": 2, "styles": True},
    "v2se": {**BASE, "intonation": 1.0, "intonation.VERSION": 2, "styles": True, "clarity": 1.0},
    "final": {**BASE, "intonation": 1.0, "intonation.VERSION": 2, "styles": True},   # = the product
}


def sentence_measures(x: np.ndarray, sr: int) -> dict:
    import intonation as I
    e = 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, np.ones(480) / 480, mode="same") + 1e-12)
    sp = e > e.max() - 35
    out = {"level_db": float(10 * np.log10(np.mean(x[sp] ** 2) + 1e-12)), "dur": len(x) / sr}
    f0, hop = I.yin(x, sr)
    v = np.flatnonzero(f0 > 0)
    if len(v) >= 10:
        st = 12 * np.log2(f0[v] / np.median(f0[v]))
        n_end = max(3, int(0.15 / (hop / sr)))
        out.update(f0_med=float(np.median(f0[v])), range_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
                   end_st=float(np.mean(st[-n_end:])))
    return out


def apply_config(eng, cfg: dict) -> None:
    import importlib
    import intonation as I
    importlib.reload(I)
    eng.prompt_styles = None
    eng.clarity = 0.0
    for k, v in cfg.items():
        if k == "styles":
            eng.prompt_styles = dict(I.PROMPT_STYLES) if v else None
        elif k.startswith("intonation."):
            setattr(I, k.split(".", 1)[1], v)
        else:
            setattr(eng, k, v)


def main() -> int:
    import soundfile as sf
    import harness
    import frontend
    from core import load_set
    import intonation as I

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dyn1")
    ap.add_argument("--configs", default="prod,v2,v2s")
    ap.add_argument("--voices", default="male_hello_clean,female_narration_clean,male_news")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--set", default=str(ROOT / "research/sets/prosody_set2.tsv"))
    ap.add_argument("--cats", default="dyn")
    a = ap.parse_args()
    box = harness.Engine()
    eng = box.eng
    cases = [c for c in load_set(a.set) if c["cat"] in a.cats.split(",")]
    out = ROOT / "output/research" / a.out
    rows = []
    for cname in a.configs.split(","):
        apply_config(eng, CONFIGS[cname])
        d = out / cname
        d.mkdir(parents=True, exist_ok=True)
        for c in cases:
            for v in a.voices.split(","):
                for s in [int(x) for x in a.seeds.split(",")]:
                    eng.join_trace = []
                    audio, _ = frontend.render(eng, c["text"], box.voice(v), seed=s)
                    sf.write(d / f"{c['id']}__{v}__s{s}.wav", audio, eng.sample_rate)
                    for i, (text, x) in enumerate(eng.join_trace):
                        rows.append({"config": cname, "case": c["id"], "voice": v, "seed": s, "i": i,
                                     "text": text, "kind": I.sentence_plan(text)[0],
                                     **sentence_measures(np.asarray(x), eng.sample_rate)})
                    eng.join_trace = None
            print(cname, c["id"], flush=True)
        (out / "sentences.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                             encoding="utf-8")
    summarize(rows)
    return 0


def summarize(rows: list[dict]) -> None:
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["config"], r["case"], r["voice"], r["seed"])].append(r)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for (cfg, *_), rs in by.items():
        lv = np.array([r["level_db"] for r in rs])
        f0 = np.array([12 * np.log2(r["f0_med"]) for r in rs if "f0_med" in r])
        agg[cfg]["level_sd"].append(lv.std())
        agg[cfg]["reg_sd"].append(f0.std() if len(f0) > 1 else np.nan)
        st = [r for r in rs if r["kind"] == "st"]
        for k in ("exc", "qyn", "qecho", "qwh"):
            ks = [r for r in rs if r["kind"] == k]
            if ks and st:
                agg[cfg][f"{k}-st level"].append(np.mean([r["level_db"] for r in ks]) - np.mean([r["level_db"] for r in st]))
                f_k = [12 * np.log2(r["f0_med"]) for r in ks if "f0_med" in r]
                f_s = [12 * np.log2(r["f0_med"]) for r in st if "f0_med" in r]
                if f_k and f_s:
                    agg[cfg][f"{k}-st reg"].append(np.mean(f_k) - np.mean(f_s))
        for k in ("st", "qyn", "qecho", "qwh", "exc"):
            agg[cfg][f"{k} end"] += [r["end_st"] for r in rs if r["kind"] == k and "end_st" in r]
    for cfg, m in agg.items():
        print(cfg, "  ".join(f"{k} {np.nanmean(v):+.2f}" for k, v in m.items()))


if __name__ == "__main__":
    raise SystemExit(main())
