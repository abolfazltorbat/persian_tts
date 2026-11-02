"""Chatbot speed: what does a slightly faster pace cost? Kept renders of DIR are
time-stretched (pitch kept) with the chatbot's numpy WSOLA (the server's path; its
CPU cannot run pedalboard) and with pedalboard/Rubber Band (the laptop's path).
    python research/pace_probe.py SRC_DIR --out pace1 --paces 1.06,1.10"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRETCH = ROOT.parent / "chatbot/ai_backend/app/backend/tts/engine/stretch.py"


def main() -> int:
    import soundfile as sf
    spec = importlib.util.spec_from_file_location("stretch", STRETCH)
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="pace1")
    ap.add_argument("--paces", default="1.06,1.10")
    a = ap.parse_args()
    out = ROOT / "output/research" / a.out
    for pace in map(float, a.paces.split(",")):
        for how in ("wsola", "pb"):
            d = out / f"{how}{int(round(pace * 100))}"
            d.mkdir(parents=True, exist_ok=True)
            for w in sorted(Path(a.src).glob("*.wav")):
                x, sr = sf.read(str(w), dtype="float32")
                if how == "wsola":
                    y = st.wsola(x, sr, pace)
                else:
                    from pedalboard import time_stretch
                    y = time_stretch(np.ascontiguousarray(x), sr, stretch_factor=pace).reshape(-1)
                sf.write(d / w.name, y, sr)
            print(d, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
