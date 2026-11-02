"""Study C — is pitch a direction in the Mimi latent space?

If a linear map from the 32-dim latent frame to the frame's F0 (semitones re
the clip median) predicts well, then adding a multiple of that direction to the
last frames of a generated take should raise the pitch there and the Mimi
decoder renders the change itself — a question rise with no DSP.

    # 1. chatbot venv: clips -> raw latents (+ the audio, for F0)
    python research/latent_f0.py dump WAV_DIR... --out output/research/latf0/lat.npz
    # 2. .venv-qwen: F0 per latent frame, ridge fit, CV R² by lag
    .venv-qwen/bin/python research/latent_f0.py fit output/research/latf0/lat.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME = 1920
SR = 24000


def dump(a) -> int:
    import soundfile as sf
    from scipy.signal import resample_poly
    sys.path.insert(0, str(ROOT / "scripts"))
    import onnxruntime  # noqa: F401  (before anything that might pull faster_whisper)
    from tts_onnx import OnnxTts
    eng = OnnxTts()
    lats, auds, names = [], [], []
    wavs = []
    for d in a.dirs:
        p = Path(d)
        wavs += sorted(p.glob("*.wav"))[: a.max_per] if p.is_dir() else [p]
    for w in wavs:
        x, sr = sf.read(str(w), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            g = np.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x = x[: 12 * SR]
        if len(x) < SR:
            continue
        lat = eng.s_enc.run(None, {"audio": x[None, None, :]})[0][0].astype(np.float32)
        lats.append(lat)
        # the encoder rounds the frame count UP: pad so clip offsets stay aligned
        auds.append(np.pad(x, (0, max(0, len(lat) * FRAME - len(x))))[: len(lat) * FRAME])
        names.append(str(w))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), lens=np.array([len(z) for z in lats]),
             lat=np.concatenate(lats), audio=np.concatenate(auds),
             emb_mean=eng.emb_mean, emb_std=eng.emb_std)
    print("dumped", len(names), "clips", sum(len(z) for z in lats), "frames ->", out)
    return 0


def frame_f0(audio: np.ndarray, n: int) -> np.ndarray:
    """Median F0 (semitones re the clip's median) per 80 ms latent frame;
    NaN where less than half the frame is voiced."""
    sys.path.insert(0, str(ROOT / "research"))
    from prosody import f0_track, _octave_clean
    f0, _ = f0_track(audio, SR)
    st = _octave_clean(12 * np.log2(f0 / np.nanmedian(f0)))
    per = int(round(FRAME / SR / 0.010))          # 8 pitch frames per latent frame
    out = np.full(n, np.nan)
    for i in range(n):
        seg = st[i * per:(i + 1) * per]
        if np.sum(~np.isnan(seg)) >= per // 2:
            out[i] = np.nanmedian(seg)
    return out, float(np.nanmedian(f0))


def fit(a) -> int:
    d = np.load(a.npz)
    lens, lat, audio = d["lens"], d["lat"], d["audio"]
    z = (lat - d["emb_mean"].reshape(1, -1)) / d["emb_std"].reshape(1, -1)
    cache = Path(a.npz).with_suffix(".f0.npz")
    if cache.exists():
        c = np.load(cache)
        y, med = c["y"], c["med"]
    else:
        ys, meds, o = [], [], 0
        for i, n in enumerate(lens):
            yy, m = frame_f0(audio[o * FRAME:(o + n) * FRAME], n)
            ys.append(yy)
            meds.append(m)
            o += n
        y, med = np.concatenate(ys), np.array(meds)
        np.savez(cache, y=y, med=med)
    clip = np.repeat(np.arange(len(lens)), lens)
    rng = np.random.default_rng(0)
    folds = rng.integers(0, 5, len(lens))[clip]          # clip-level CV folds
    res = {}
    for space, X0 in (("raw", lat), ("norm", z)):
        for lag in (-2, -1, 0, 1, 2):
            # latent frame t predicts F0 of frame t+lag (within clip)
            Xs, Ys, Fs = [], [], []
            o = 0
            for i, n in enumerate(lens):
                Xi, Yi = X0[o:o + n], y[o:o + n]
                if lag > 0:
                    Xi, Yi = Xi[:-lag], Yi[lag:]
                elif lag < 0:
                    Xi, Yi = Xi[-lag:], Yi[:lag]
                Xs.append(Xi)
                Ys.append(Yi)
                Fs.append(np.full(len(Yi), folds[o]))
                o += n
            X, Y, F = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Fs)
            ok = ~np.isnan(Y)
            X, Y, F = X[ok], Y[ok], F[ok]
            X1 = np.hstack([X, np.ones((len(X), 1))])
            pred = np.zeros_like(Y)
            for k in range(5):
                tr, te = F != k, F == k
                w = np.linalg.solve(X1[tr].T @ X1[tr] + a.ridge * np.eye(X1.shape[1]), X1[tr].T @ Y[tr])
                pred[te] = X1[te] @ w
            r2 = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
            res[f"{space}|lag{lag}"] = round(float(r2), 3)
    # absolute pitch (speaker register): clip-mean latent -> log2 median F0
    print(json.dumps({"frames": int(len(y)), "voiced": int(np.sum(~np.isnan(y))), "clips": int(len(lens)),
                      "cv_r2": res}, indent=1))
    # the direction itself (norm space, lag 0, all data) for steering
    ok = ~np.isnan(y)
    X1 = np.hstack([z[ok], np.ones((ok.sum(), 1))])
    w = np.linalg.solve(X1.T @ X1 + a.ridge * np.eye(X1.shape[1]), X1.T @ y[ok])
    np.save(Path(a.npz).with_name("f0_dir_norm.npy"), w)
    g = w[:-1]
    print("direction |w| (st per unit z):", round(float(np.linalg.norm(g)), 3),
          "-> z step for +1 st along w:", round(float(1 / np.linalg.norm(g)), 3))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--max-per", type=int, default=400)
    p = sub.add_parser("fit")
    p.add_argument("npz")
    p.add_argument("--ridge", type=float, default=1.0)
    a = ap.parse_args()
    raise SystemExit(dump(a) if a.cmd == "dump" else fit(a))
