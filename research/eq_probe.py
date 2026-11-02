"""Study D4 — voice-referenced spectral correction on kept renders (offline).

    python research/eq_probe.py SRC_DIR --out eq1
Variants per voice (curve source x strength):
  r05 / r10  curve from 60 plain-clip bench renders of the voice (flu_c_base), strength .5 / 1
  e10        curve from the decoded prompt extension only (what a new voice has), strength 1
Writes output/research/<out>/<variant>/*.wav and curves.json.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    import soundfile as sf
    import voice_eq as Q
    from tts_onnx import OnnxTts
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="eq1")
    a = ap.parse_args()
    eng = OnnxTts()
    sr = eng.sample_rate
    out = ROOT / "output/research" / a.out
    curves = {}
    for v in ("male_hello_clean", "female_narration_clean", "male_news"):
        ref, _ = sf.read(ROOT / "voices" / f"{v}.wav", dtype="float32")
        bench = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(str(ROOT / f"output/eval/flu_c_base/*__{v}__*.wav")))[:60]]
        raw = np.load(ROOT / f"output/research/voice_ext_m/{v}.ext.npy")
        ext = eng._decode_all([r.reshape(1, -1) for r in (raw - eng.emb_mean) / eng.emb_std])
        f, cr = Q.curve(ref, bench, sr)
        _, ce = Q.curve(ref, ext, sr)
        curves[v] = {"f": f[::8].round(1).tolist(), "bench": cr[::8].round(2).tolist(), "ext": ce[::8].round(2).tolist()}
        hs = {"r05": Q.fir(f, cr, sr, 0.5), "r10": Q.fir(f, cr, sr, 1.0), "e10": Q.fir(f, ce, sr, 1.0)}
        for name, h in hs.items():
            (out / name).mkdir(parents=True, exist_ok=True)
            for w in sorted(Path(a.src).glob(f"*__{v}__*.wav")):
                x, _ = sf.read(w, dtype="float32")
                sf.write(out / name / w.name, Q.apply(x, sr, h), sr)
        print(v, "bench curve dB @1k,3k,5k:", [round(float(np.interp(fq, f, cr)), 1) for fq in (1000, 3000, 5000)],
              "ext curve:", [round(float(np.interp(fq, f, ce)), 1) for fq in (1000, 3000, 5000)], flush=True)
    (out / "curves.json").write_text(json.dumps(curves))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
