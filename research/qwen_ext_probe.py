"""Review point 5 (a second continuation model): does prompt self-extension carry over to
Qwen3-TTS-0.6B-Base? Its voice cloning is in-context: the prompt is the reference audio AND its
transcript, after which the model continues with the target text — a continuation model too, but a
different architecture (discrete codec tokens, 12 Hz, Qwen LM). Persian is not an officially supported
language for it, so absolute error is high; the question is only plain vs. extended prompt.

  plain : (clip, clip transcript)
  ext   : (clip + the model's own reading of sentence A, clip transcript + A) — the extension is the
          median-length of four takes of A from the plain prompt, cut 80 ms after its speech, as in
          the ONNX engine (no recogniser).

    .venv-qwen/bin/python research/qwen_ext_probe.py --out qwen1 [--n 50] [--seeds 1,2]
-> output/research/<out>/{plain,ext}/<case>__<voice>__s<seed>.wav  (judge with research/short_stats.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "eval"))
EXT_TEXT = "امروز هوا کمی سرد است و باد می‌وزد."
VOICES = ["female_narration_clean", "male_news"]        # clips whose transcript the recogniser gets right


def speech_span(x: np.ndarray, sr: int, tail_s: float = 0.08) -> np.ndarray:
    e = 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, np.ones(480) / 480, mode="same") + 1e-12)
    sp = np.flatnonzero(e > e.max() - 35)
    if not len(sp):
        return x
    return x[max(0, sp[0] - int(0.02 * sr)): min(len(x), sp[-1] + int(tail_s * sr))]


def main() -> int:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from qwen_backend import QwenBackend
    from core import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qwen1")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--voices", default=",".join(VOICES))
    a = ap.parse_args()
    qb = QwenBackend()
    print("load", round(qb.load(), 1), "s on", qb.device, flush=True)
    m = qb.model
    texts = json.loads((ROOT / "output/research/clip_texts.json").read_text(encoding="utf-8"))
    cases = [c for c in load_set(ROOT / "research/sets/short_test.tsv") if c["lang"] == "fa"][: a.n]
    out = ROOT / "output/research" / a.out
    gen_kw = dict(language="Auto", do_sample=True, top_k=50, top_p=0.9, temperature=0.7,
                  repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
                  subtalker_top_p=0.9, subtalker_temperature=0.7)

    def gen(text, prompt, seed):
        torch.manual_seed(seed)
        with torch.inference_mode():
            waves, sr = m.generate_voice_clone(text=text, voice_clone_prompt=prompt,
                                               max_new_tokens=min(1200, max(120, len(text) * 5)), **gen_kw)
        return np.asarray(waves[0], dtype=np.float32).reshape(-1), sr

    meta = {}
    for v in a.voices.split(","):
        x, sr = sf.read(str(ROOT / "voices" / f"{v}.wav"), dtype="float32")
        x = (x.mean(axis=1) if x.ndim > 1 else x)[: 5 * sr]
        ref_text = texts[v]
        with torch.inference_mode():
            plain = m.create_voice_clone_prompt(ref_audio=(x, sr), ref_text=ref_text, x_vector_only_mode=False)
        takes = []
        for s in range(4):
            y, osr = gen(EXT_TEXT, plain, 500 + s)
            y = speech_span(y, osr)
            takes.append((len(y), y, osr))
        takes.sort(key=lambda t: t[0])
        _, ext, osr = takes[len(takes) // 2]
        if osr != sr:
            g = np.gcd(osr, sr)
            ext = resample_poly(ext, sr // g, osr // g).astype(np.float32)
        both = np.concatenate([x, ext])
        with torch.inference_mode():
            extp = m.create_voice_clone_prompt(ref_audio=(both, sr), ref_text=ref_text + " " + EXT_TEXT,
                                               x_vector_only_mode=False)
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        sf.write(out / "prompts" / f"{v}_ext.wav", both, sr)
        meta[v] = {"ref_text": ref_text, "clip_s": len(x) / sr, "ext_s": len(ext) / sr}
        for name, prompt in (("plain", plain), ("ext", extp)):
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            for c in cases:
                for s in [int(t) for t in a.seeds.split(",")]:
                    y, osr = gen(c["text"], prompt, s)
                    sf.write(d / f"{c['id']}__{v}__s{s}.wav", y, osr)
            print(v, name, "done", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
