"""Study C — the latent displacement of a pitch shift (analysis by synthesis).

A regression direction (latent_f0.py) is correlational: pushing along it did
nothing up to alpha 3, then broke. Here the direction is causal from the
codec's own view: pitch-shift real speech by s semitones (pedalboard
PitchShift, formants kept by the shifter), encode original and shifted with the
Mimi encoder, and fit the per-frame displacement

    z_s - z_0  ≈  s * (A z_0 + b)            (speech frames only)

`b` alone is the mean "+1 semitone" direction; `A z_0 + b` lets the step depend
on the frame. Saved to output/research/latf0/pitch_op.npz for steering.

    python research/pitch_dir.py WAV_DIR... --shifts -4,-2,2,4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME, SR = 1920, 24000


def main() -> int:
    import soundfile as sf
    from scipy.signal import resample_poly
    sys.path.insert(0, str(ROOT / "scripts"))
    import onnxruntime  # noqa: F401
    import pedalboard
    from tts_onnx import OnnxTts

    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--shifts", default="-4,-2,2,4")
    ap.add_argument("--max-per", type=int, default=60)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "output/research/latf0/pitch_op.npz"))
    a = ap.parse_args()
    eng = OnnxTts()
    shifts = [float(s) for s in a.shifts.split(",")]
    wavs = []
    for d in a.dirs:
        p = Path(d)
        wavs += sorted(p.glob("*.wav"))[: a.max_per] if p.is_dir() else [p]
    Z0, DZ, S = [], [], []
    for w in wavs:
        x, sr = sf.read(str(w), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            g = np.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        x = x[: 8 * SR]
        enc = lambda y: ((eng.s_enc.run(None, {"audio": y[None, None, :].astype(np.float32)})[0][0]
                          - eng.emb_mean) / eng.emb_std).astype(np.float32)
        z0 = enc(x)
        n = len(z0)
        e = 10 * np.log10(np.mean(np.pad(x, (0, n * FRAME - len(x)))[: n * FRAME].reshape(n, FRAME) ** 2, 1) + 1e-10)
        sp = e > e.max() - 30
        for s in shifts:
            y = pedalboard.Pedalboard([pedalboard.PitchShift(semitones=s)])(x[None], SR)[0]
            zs = enc(y[: len(x)])
            m = min(len(zs), n)
            keep = sp[:m]
            Z0.append(z0[:m][keep])
            DZ.append((zs[:m] - z0[:m])[keep])
            S.append(np.full(keep.sum(), s))
    Z0, DZ, S = np.concatenate(Z0), np.concatenate(DZ), np.concatenate(S)
    # mean direction per semitone: b = sum(s*dz) / sum(s^2)
    b = (S[:, None] * DZ).sum(0) / (S ** 2).sum()
    pred_b = S[:, None] * b[None]
    r2_b = 1 - ((DZ - pred_b) ** 2).sum() / ((DZ - DZ.mean(0)) ** 2).sum()
    # frame-dependent operator: dz/s = [z0, 1] @ M   (ridge)
    X = np.hstack([Z0, np.ones((len(Z0), 1))]) * S[:, None]
    rng = np.random.default_rng(0)
    te = rng.random(len(X)) < 0.2
    M = np.linalg.solve(X[~te].T @ X[~te] + a.ridge * np.eye(X.shape[1]), X[~te].T @ DZ[~te])
    res = DZ[te] - X[te] @ M
    r2_m = 1 - (res ** 2).sum() / ((DZ[te] - DZ[te].mean(0)) ** 2).sum()
    # sign symmetry: +s and -s displacements should be opposite
    up = DZ[S > 0].mean(0)
    dn = DZ[S < 0].mean(0)
    cos = float(up @ dn / (np.linalg.norm(up) * np.linalg.norm(dn)))
    np.savez(a.out, b=b.astype(np.float32), M=M.astype(np.float32))
    print(f"frames {len(Z0)} clips {len(wavs)} | |b| per st {np.linalg.norm(b):.3f} | "
          f"R2 mean-direction {r2_b:.3f} | R2 frame-operator (held-out) {r2_m:.3f} | cos(up, down) {cos:+.3f} "
          f"| mean |dz| at 4 st {np.linalg.norm(DZ[np.abs(S) == 4], axis=1).mean():.3f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
