"""Study C — where does the UTMOS cost of the intonation rules come from?

From kept bench renders (base tag and intonation tag), writes control variants:
  flat2   base, PSOLA with a constant +2 st shift (processing, no contour change)
  rt      base -> Mimi encoder -> decoder (codec round-trip cost)
  int_rt  intonation render -> Mimi round-trip (does the codec clean PSOLA?)

    python research/into_controls.py c_base c_int --out output/research/into_ctl
    PythonProject5/.venv/bin/python research/utmos_dir.py output/research/into_ctl/* ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    import soundfile as sf
    import onnxruntime  # noqa: F401
    import intonation as I
    from tts_onnx import OnnxTts

    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("into")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="p,e,w,x", help="case-id prefixes to include")
    ap.add_argument("--qonly", action="store_true", help="from pairs, only the «؟» member")
    a = ap.parse_args()
    eng = OnnxTts()
    pre = tuple(a.cases.split(","))
    out = Path(a.out)
    for v in ("flat2", "rt", "int_rt"):
        (out / v).mkdir(parents=True, exist_ok=True)

    def rt(x):
        raw = eng.s_enc.run(None, {"audio": x[None, None, :].astype(np.float32)})[0][0]
        z = ((raw - eng.emb_mean) / eng.emb_std).astype(np.float32)
        y = eng._decode_all([r[None] for r in z])
        return y[: len(x)]

    n = 0
    for w in sorted((ROOT / "output/eval" / f"flu_{a.base}").glob("*.wav")):
        case = w.stem.split("__")[0]
        if not case.startswith(pre) or (case.startswith("p") and not case.endswith("q")):
            continue
        x, sr = sf.read(str(w), dtype="float32")
        f0, hop = I.yin(x, sr)
        ts = np.arange(len(x)) / hop
        fr = np.arange(len(f0))
        f0s = np.where(np.interp(ts, fr, (f0 > 0).astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
        sf.write(out / "flat2" / w.name, I.psola(x.astype(np.float64), sr, f0s, np.full(len(x), 2 ** (2 / 12))), sr)
        sf.write(out / "rt" / w.name, rt(x), sr)
        wi = ROOT / "output/eval" / f"flu_{a.into}" / w.name
        if wi.exists():
            xi, _ = sf.read(str(wi), dtype="float32")
            sf.write(out / "int_rt" / w.name, rt(xi), sr)
        n += 1
    print("wrote", n, "x 3 variants ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
