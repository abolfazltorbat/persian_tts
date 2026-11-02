"""UTMOS22 for every WAV in one or more directories (PythonProject5/.venv, torch).

    .venv/bin/python persian_tts/research/utmos_dir.py DIR... [--model ~/.cache/persian_tts_eval/models/utmos.jit]
Writes DIR/utmos.jsonl ({"wav": name, "utmos": x}) and prints the mean per dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> int:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    model_path = str(Path.home() / ".cache/persian_tts_eval/models/utmos.jit")
    if "--model" in argv:
        i = argv.index("--model")
        model_path, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=dev).eval()
    for d in argv:
        d = Path(d)
        rows = []
        for w in sorted(d.glob("*.wav")):
            x, sr = sf.read(str(w), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                g = np.gcd(sr, 16000)
                x = resample_poly(x, 16000 // g, sr // g)
            ten = torch.from_numpy((np.clip(x, -1, 1) * 32767).astype(np.int16)).unsqueeze(0).to(dev)
            with torch.no_grad():
                rows.append({"wav": w.name, "utmos": round(float(model(ten).item()), 4)})
        (d / "utmos.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(d, len(rows), "mean", round(float(np.mean([r["utmos"] for r in rows])), 4) if rows else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
