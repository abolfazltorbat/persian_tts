"""DNSMOS P.835 (Reddy et al., ICASSP 2022; microsoft/DNS-Challenge ONNX, non-personalized)
for every WAV in DIR(s): SIG (speech signal: distortion / clarity), BAK (background
noise), OVRL. A reference-free check that a "clarity" change does not just add
noise or distortion. Clips shorter than 9.01 s are tiled, as the reference code does.

    python research/dnsmos_dir.py DIR...  -> DIR/dnsmos.jsonl
Runs in any interpreter with onnxruntime + scipy (chatbot venv)."""
import json
import sys
from pathlib import Path

import numpy as np

M = Path.home() / ".cache/persian_tts_eval/models/dnsmos/sig_bak_ovr.onnx"
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def score(sess, x: np.ndarray, sr: int) -> dict:
    from scipy.signal import resample_poly
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
    n = int(9.01 * 16000)
    while len(x) < n:
        x = np.concatenate([x, x])
    hops = int(np.floor(len(x) / 16000) - 9.01) + 1
    s, b, o = [], [], []
    for i in range(max(1, hops)):
        seg = x[i * 16000: i * 16000 + n]
        if len(seg) < n:
            continue
        r = sess.run(None, {"input_1": seg.astype(np.float32)[None]})[0][0]
        s.append(P_SIG(r[0])); b.append(P_BAK(r[1])); o.append(P_OVR(r[2]))
    return {"sig": round(float(np.mean(s)), 4), "bak": round(float(np.mean(b)), 4), "ovrl": round(float(np.mean(o)), 4)}


def main(argv) -> int:
    import onnxruntime as ort
    import soundfile as sf
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(M), opts, providers=["CPUExecutionProvider"])
    for d in map(Path, argv):
        rows = []
        for w in sorted(d.glob("*.wav")):
            x, sr = sf.read(str(w), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            rows.append({"wav": w.name, **score(sess, x, sr)})
        (d / "dnsmos.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        m = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("sig", "bak", "ovrl")} if rows else {}
        print(d, len(rows), m, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
