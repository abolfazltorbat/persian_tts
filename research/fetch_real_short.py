"""Real human recordings of short sentences (GPTInformal-Persian, CC0) for the
ASR noise floor: what CER does the judge give REAL speech of 1-5 words?

Pages through the dataset-server rows API (no parquet reader needed), keeps
clips <= --max-dur seconds, saves them as 24 kHz WAV + a tsv of transcripts.

    python research/fetch_real_short.py --n 150
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

API = "https://datasets-server.huggingface.co/rows"
OUT = Path.home() / ".cache" / "persian_tts_ft" / "data_gptinformal"


def page(offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": "MahtaFetrat/GPTInformal-Persian", "config": "default",
                                "split": "train", "offset": offset, "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=180) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except Exception as exc:                        # transient API errors
            print("retry", offset, exc, flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def main() -> int:
    import soundfile as sf
    from scipy.signal import resample_poly

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-dur", type=float, default=2.6)
    ap.add_argument("--stride", type=int, default=7, help="sample every k-th page of 10 (spread topics)")
    a = ap.parse_args()
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    tsv = OUT / "short_real.tsv"
    have = tsv.read_text(encoding="utf-8").count("\n") if tsv.exists() else 0
    off = 0
    while have < a.n and off < 4724:
        for row in page(off, 10):
            if have >= a.n:
                break
            dur = float(row["duration"])
            text = row["transcript"].strip()
            if dur > a.max_dur or not text:
                continue
            x = np.asarray(row["audio"], dtype=np.float32)
            sr = int(float(row["samplerate"]))
            x = resample_poly(x, 24000 // np.gcd(24000, sr), sr // np.gcd(24000, sr)).astype(np.float32)
            name = Path(row["file name"]).stem
            sf.write(OUT / "wav" / f"{name}.wav", x, 24000)
            with tsv.open("a", encoding="utf-8") as f:
                f.write(f"{name}\t{dur}\t{text}\n")
            have += 1
        print(f"offset {off}: {have}/{a.n}", flush=True)
        off += 10 * a.stride
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
