"""Speaker-matched prompt extension (no speech recogniser).

The median-length extension take kept the male voices but lowered the female
voice's speaker similarity (−0.03). Here the model reads the neutral sentence
K times from the clip and we keep the take whose WeSpeaker embedding is closest
to the clip's. Two steps (two interpreters, nothing installed):

    # 1. chatbot venv: K takes per voice -> wav + raw latents
    python research/ext_spk_select.py gen --voices male_hello_clean,female_narration_clean,male_news --k 8
    # 2. .venv-qwen: score against the clip, write output/research/voice_ext_s/<stem>.ext.npy
    .venv-qwen/bin/python research/ext_spk_select.py pick --model ~/.cache/persian_tts_eval/models/wespeaker_resnet34_LM.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "output" / "research" / "ext_spk"
OUT = ROOT / "output" / "research" / "voice_ext_s"
PH = "?emruz havA kami sard ?ast va bAd mivazad"   # = OnnxTts.PROMPT_EXT_PHONEMES (sentence A)


def gen(a) -> int:
    import soundfile as sf
    sys.path.insert(0, str(ROOT / "research"))
    import core
    lab = core.Lab(asr=False)
    eng = lab.eng
    WORK.mkdir(parents=True, exist_ok=True)
    meta = {}
    for v in a.voices.split(","):
        raw = lab.prompt_latents(v)
        for k in range(a.k):
            cache, off = lab.prompt_cache(raw)
            tk = core.generate(lab, cache, off, PH, np.random.default_rng(500 + k))
            audio = core.decode(lab, tk)
            idx = eng._speech_frames(audio, len(tk.z))
            if not len(idx):
                continue
            z = tk.z[max(0, int(idx[0]) - 1): int(idx[-1]) + 2]
            stem = Path(lab.voice_path(v)).stem
            np.save(WORK / f"{stem}__k{k}.npy", lab.to_raw(z))
            fs = eng.frame_samples
            sf.write(WORK / f"{stem}__k{k}.wav", audio[max(0, int(idx[0]) - 1) * fs:(int(idx[-1]) + 2) * fs], lab.sr)
            meta.setdefault(stem, []).append({"k": k, "frames": int(len(z)), "clip": lab.voice_path(v)})
    (WORK / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("generated", {s: len(x) for s, x in meta.items()})
    return 0


def pick(a) -> int:
    sys.path.insert(0, str(ROOT / "eval"))
    from speaker import SpeakerEmbedder
    emb = SpeakerEmbedder(str(Path(a.model).expanduser()))
    meta = json.loads((WORK / "meta.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for stem, takes in meta.items():
        r = emb.embed(takes[0]["clip"])
        sims = []
        for t in takes:
            sims.append((float(r @ emb.embed(str(WORK / f"{stem}__k{t['k']}.wav"))), t["k"], t["frames"]))
        sims.sort(reverse=True)
        best = sims[0]
        np.save(OUT / f"{stem}.ext.npy", np.load(WORK / f"{stem}__k{best[1]}.npy"))
        report[stem] = {"picked_k": best[1], "sim": round(best[0], 4), "all": [(round(s, 4), k, f) for s, k, f in sims]}
        print(stem, report[stem])
    (OUT / "pick.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "pick"])
    ap.add_argument("--voices", default="male_hello_clean,female_narration_clean,male_news")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default="~/.cache/persian_tts_eval/models/wespeaker_resnet34_LM.onnx")
    a = ap.parse_args()
    raise SystemExit(gen(a) if a.cmd == "gen" else pick(a))
