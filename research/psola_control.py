"""Study D — processing cost of the two TD-PSOLA implementations, contour unchanged:
a flat +2 st shift of kept renders with (v1) waveform-peak marks and (v2) ZFF epoch marks;
also v2 at +4 st and a v2 final-syllable rise only.
    python research/psola_control.py SRC_TAG --out output/research/psola_ctl [--cases p,e,w,x]"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    import soundfile as sf
    import intonation as I
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="p,e,w,x")
    a = ap.parse_args()
    out = Path(a.out)
    for v in ("v1_2st", "v2_2st", "v1_4st", "v2_4st"):
        (out / v).mkdir(parents=True, exist_ok=True)
    n = 0
    for w in sorted((ROOT / "output/eval" / f"flu_{a.src}").glob("*.wav")):
        case = w.stem.split("__")[0]
        if not case.startswith(tuple(a.cases.split(","))) or (case.startswith("p") and not case.endswith("q")):
            continue
        x, sr = sf.read(str(w), dtype="float32")
        f0, hop = I.yin(x, sr)
        ts = np.arange(len(x)) / hop
        fr = np.arange(len(f0))
        f0s = np.where(np.interp(ts, fr, (f0 > 0).astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
        for st in (2, 4):
            r = np.full(len(x), 2 ** (st / 12))
            sf.write(out / f"v1_{st}st" / w.name, I.psola(x.astype(np.float64), sr, f0s, r), sr)
            y, _ = I.psola_v2(x, sr, f0s, r)
            sf.write(out / f"v2_{st}st" / w.name, y.astype(np.float32), sr)
        n += 1
    print("wrote", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
