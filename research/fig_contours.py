"""F0 contours for the paper's expressiveness figure (.venv-qwen, pYIN):
a statement / question pair and an exclamation, no rule vs. the final method.
    .venv-qwen/bin/python research/fig_contours.py -> output/research/dfinal/contours_fig.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))


def main() -> int:
    import soundfile as sf
    from prosody import f0_track, energy_db
    D = ROOT / "output/research/dfinal"
    items = {"st": D / "base/p08s__male_hello_clean__s2.wav",
             "q_base": D / "base/p08q__male_hello_clean__s2.wav",
             "q_new": D / "new2/p08q__male_hello_clean__s2.wav",
             "x_base": D / "base/x02__male_hello_clean__s1.wav",
             "x_new": D / "new2/x02__male_hello_clean__s1.wav"}
    out = {}
    for k, w in items.items():
        x, sr = sf.read(str(w), dtype="float32")
        f0, t = f0_track(x, sr)
        e = energy_db(x, sr)
        out[k] = {"t": t.round(3).tolist(), "f0": [None if np.isnan(v) else round(float(v), 1) for v in f0],
                  "e": [round(float(v), 1) for v in e[: len(t)]]}
    (D / "contours_fig.json").write_text(json.dumps(out))
    print("ok", {k: len(v["t"]) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
