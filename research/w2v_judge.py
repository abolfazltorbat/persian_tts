"""Second, independent judge: Persian wav2vec2-CTC (m3hrdadfi/wav2vec2-large-xlsr-persian-v3).

Runs in persian_tts/.venv-qwen (torch + transformers already there; nothing
installed). Whisper is the primary judge everywhere; this one exists so that
no conclusion rests on a single recogniser — and so an "ASR-verified" retry
can be selected by one model and scored by another (no circularity).

    .venv-qwen/bin/python research/w2v_judge.py --wavs DIR [--out hyps.jsonl]
    .venv-qwen/bin/python research/w2v_judge.py --list items.jsonl --out hyps.jsonl

Greedy CTC decoding, 16 kHz, Persian script only (English words come out as
Persian letters or nothing — use it for Persian items).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", default="", help="directory of .wav files")
    ap.add_argument("--list", default="", help="jsonl with {id, wav}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    items = []
    if a.wavs:
        items = [{"id": p.stem, "wav": str(p)} for p in sorted(Path(a.wavs).glob("*.wav"))]
    if a.list:
        items += [json.loads(x) for x in Path(a.list).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    with out.open("a", encoding="utf-8") as f:
        for it in items:
            if it["id"] in done:
                continue
            x, sr = sf.read(it["wav"])
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                g = np.gcd(int(sr), 16000)
                x = resample_poly(x, 16000 // g, int(sr) // g)
            x = x.astype(np.float32)
            if len(x) < 1600:
                hyp = ""
            else:
                inp = proc(x, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.inference_mode():
                    logits = model(inp.input_values).logits
                hyp = proc.batch_decode(torch.argmax(logits, dim=-1))[0].strip()
            f.write(json.dumps({"id": it["id"], "hyp_w2v": hyp}, ensure_ascii=False) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
