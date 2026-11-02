"""Study C — does pushing latents along the pitch direction raise the pitch?

Post-hoc test, no generation: real (or rendered) clips -> Mimi latents ->
add alpha * ramp(t) * u to the last K speech frames (u = unit pitch direction
from latent_f0.py fit, ramp 0 -> 1) -> Mimi decoder -> WAV. Then
research/prosody.py measures end_st per alpha.

    python research/steer_probe.py WAV... --alphas 0,0.5,1,1.5,2,3 --k 5 --out output/research/steer1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME, SR = 1920, 24000


def last_speech_frame(x: np.ndarray, n: int) -> int:
    e = 10 * np.log10(np.mean(np.pad(x, (0, n * FRAME - len(x)))[: n * FRAME].reshape(n, FRAME) ** 2, 1) + 1e-10)
    sp = np.flatnonzero(e > e.max() - 30)
    return int(sp[-1]) if len(sp) else n - 1


def steer(z: np.ndarray, end: int, u: np.ndarray, alpha: float, k: int, shape: str = "ramp") -> np.ndarray:
    z = z.copy()
    lo = max(0, end - k + 1)
    m = end - lo + 1
    r = np.linspace(1.0 / m, 1.0, m) if shape == "ramp" else np.ones(m)
    z[lo:end + 1] += alpha * r[:, None] * u[None, :]
    # the frames after the last speech frame (tail silence) keep the end level
    if shape == "ramp" and end + 1 < len(z):
        z[end + 1:] += 0.0
    return z


def main() -> int:
    import soundfile as sf
    from scipy.signal import resample_poly
    sys.path.insert(0, str(ROOT / "scripts"))
    import onnxruntime  # noqa: F401
    from tts_onnx import OnnxTts

    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--alphas", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--shape", default="ramp")
    ap.add_argument("--dir", default=str(ROOT / "output/research/latf0/f0_dir_norm.npy"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eng = OnnxTts()
    w = np.load(a.dir)[:-1]
    u = (w / np.linalg.norm(w)).astype(np.float32)
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in a.wavs:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            g = np.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        raw = eng.s_enc.run(None, {"audio": x[None, None, :]})[0][0]
        z = ((raw - eng.emb_mean) / eng.emb_std).astype(np.float32)
        end = last_speech_frame(x, len(z))
        for al in [float(v) for v in a.alphas.split(",")]:
            zz = steer(z, end, u, al, a.k, a.shape)
            audio = eng._decode_all([r[None] for r in zz])
            name = f"{Path(p).stem}__a{al:g}"
            sf.write(out / "wav" / f"{name}.wav", audio, SR)
            rows.append({"src": p, "alpha": al, "k": a.k, "end": end, "wav": str(out / "wav" / f"{name}.wav")})
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print("wrote", len(rows), "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
