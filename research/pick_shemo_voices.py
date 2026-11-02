"""Generalisation (review point 5): ten more voices, from ShEMO neutral speech (87 speakers; no
speaker labels in the mirror). Candidates: neutral clips of 2-5 s with DNSMOS BAK >= 2.8 (cleanest first) (little
background). Speakers are told apart by WeSpeaker embeddings: clips are taken greedily so that no two
chosen ones are more similar than MAX_SIM; five with median F0 < 165 Hz (male), five above.
Runs in .venv-qwen.   -> output/research/voices_shemo/sh01..sh10.wav + picked.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "research"))
D = Path.home() / ".cache/persian_tts_ft/shemo"
OUT = ROOT / "output/research/voices_shemo"
MAX_SIM = 0.45


def main() -> int:
    import onnxruntime as ort
    import soundfile as sf
    from speaker import SpeakerEmbedder
    from prosody import f0_track
    from dnsmos_dir import M, score
    emb = SpeakerEmbedder(str(Path.home() / ".cache/persian_tts_eval/models/wespeaker_resnet34_LM.onnx"))
    sess = ort.InferenceSession(str(M), providers=["CPUExecutionProvider"])
    cands = []
    for line in (D / "clips.tsv").read_text(encoding="utf-8").splitlines():
        cid, emo, dur = line.split("\t")[:3]
        if emo != "N" or not (2.0 <= float(dur) <= 5.0):
            continue
        w = D / "wav" / f"{cid}.wav"
        x, sr = sf.read(str(w), dtype="float32")
        q = score(sess, x, sr)
        if q["bak"] < 2.8:                       # female neutral clips are scarce in ShEMO
            continue
        f0, _ = f0_track(x, sr)
        f0m = float(np.nanmedian(f0)) if np.any(~np.isnan(f0)) else np.nan
        if np.isnan(f0m):
            continue
        cands.append({"id": cid, "dur": float(dur), "f0": f0m, "bak": q["bak"], "sig": q["sig"],
                      "e": emb.embed(str(w))})
    print("candidates", len(cands), flush=True)
    cands.sort(key=lambda c: -(c["sig"] + c["bak"]))           # cleanest first
    picked = {"m": [], "f": []}
    for c in cands:
        g = "m" if c["f0"] < 165 else "f"
        if len(picked[g]) >= 5:
            continue
        if all(float(c["e"] @ p["e"]) < MAX_SIM for p in picked["m"] + picked["f"]):
            picked[g].append(c)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for k, c in enumerate(picked["m"] + picked["f"], 1):
        x, sr = sf.read(str(D / "wav" / f"{c['id']}.wav"), dtype="float32")
        sf.write(OUT / f"sh{k:02d}.wav", x, sr)
        meta.append({"voice": f"sh{k:02d}", "src": c["id"], "dur": c["dur"], "f0": round(c["f0"], 1),
                     "sig": c["sig"], "bak": c["bak"]})
    sims = [float(a["e"] @ b["e"]) for i, a in enumerate(picked["m"] + picked["f"])
            for b in (picked["m"] + picked["f"])[i + 1:]]
    (OUT / "picked.json").write_text(json.dumps({"voices": meta, "max_pair_sim": max(sims) if sims else None},
                                                indent=1))
    print(json.dumps(meta, indent=1), "max pair sim", round(max(sims), 3) if sims else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
