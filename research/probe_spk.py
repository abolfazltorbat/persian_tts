"""Speaker similarity (WeSpeaker) of probe takes to their voice clip, per prompt kind.
Runs in .venv-qwen.   .venv-qwen/bin/python research/probe_spk.py prompt_ext8 [--base ref]"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from speaker import SpeakerEmbedder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--base", default="ref")
ap.add_argument("--model", default=str(Path.home() / ".cache/persian_tts_eval/models/wespeaker_resnet34_LM.onnx"))
a = ap.parse_args()
d = ROOT / "output" / "research" / a.tag
rows = [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
emb = SpeakerEmbedder(a.model)
clips = {}
sims = collections.defaultdict(dict)
for r in rows:
    v = r["voice"]
    if v not in clips:
        p = v if "/" in v else str(ROOT / "voices" / f"{v}.wav")
        clips[v] = emb.embed(p)
    w = d / "wav" / f"{r['wid']}.wav"
    if not w.exists():
        continue
    import soundfile as sf
    x, sr = sf.read(str(w), dtype="float32")
    if len(x) < sr * 0.3:          # too short for a stable embedding
        continue
    sims[(v, r["kind"])][(r["case"], r["unit"], r["take"])] = float(clips[v] @ emb.embed(x, sr))
out = {}
for (v, k), s in sorted(sims.items()):
    base = sims.get((v, a.base), {})
    keys = [q for q in s if q in base]
    diff = np.mean([s[q] - base[q] for q in keys]) if keys and k != a.base else 0.0
    rng = np.random.default_rng(0)
    by = collections.defaultdict(list)
    for q in keys:
        by[q[0]].append(s[q] - base[q])
    cs = list(by)
    bs = [np.mean([x for j in rng.integers(0, len(cs), len(cs)) for x in by[cs[j]]]) for _ in range(2000)] if cs and k != a.base else [0, 0]
    out[f"{v}|{k}"] = {"n": len(s), "spk": round(float(np.mean(list(s.values()))), 4), "diff": round(float(diff), 4),
                       "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    print(f"{v:24s} {k:40s} n={len(s):3d} spk={out[f'{v}|{k}']['spk']:.4f}  Δ {diff:+.4f} CI {out[f'{v}|{k}']['ci95']}")
(d / "spk.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
