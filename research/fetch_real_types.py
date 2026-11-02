"""Real recordings by sentence type (GPTInformal-Persian, CC0, one female speaker)
as the intonation reference for Study C: how does a human end a yes/no question,
a wh-question, an exclamation, a statement?

Pages through the dataset-server rows API at spread offsets and keeps clips of
1.0-6.0 s until each type has --per clips (or --max-rows rows were read).
Types come from the transcript's final mark: «؟» -> question (split later into
yes/no vs wh by the presence of a question word), «!» -> exclamation,
«.» -> statement.

    python research/fetch_real_types.py --per 40 --max-rows 600
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
OUT = Path.home() / ".cache" / "persian_tts_ft" / "data_gptinformal_types"
N_ROWS = 4724


def page(offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": "MahtaFetrat/GPTInformal-Persian", "config": "default",
                                "split": "train", "offset": offset, "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=240) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except Exception as exc:                        # transient API errors
            print("retry", offset, exc, flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def kind(text: str) -> str | None:
    t = text.strip().rstrip(" .…")
    if t.endswith(("؟", "?")):
        return "q"
    if t.endswith("!"):
        return "x"
    if text.strip().endswith("."):
        return "s"
    return None


def main() -> int:
    import soundfile as sf
    from scipy.signal import resample_poly

    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=600)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=6.0)
    a = ap.parse_args()
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    tsv = OUT / "clips.tsv"
    seen = set()
    have = {"q": 0, "x": 0, "s": 0}
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            cid, k, _d, _t = line.split("\t", 3)
            seen.add(cid)
            have[k] += 1
    # spread offsets: a golden-ratio walk over the pages covers every subject
    pages = sorted({int((i * 0.6180339887 % 1.0) * (N_ROWS // 10)) * 10 for i in range(N_ROWS // 10)},
                   key=lambda o: (o * 0.6180339887) % 1.0)
    read = 0
    with tsv.open("a", encoding="utf-8") as f:
        for off in pages:
            if read >= a.max_rows or all(v >= a.per for v in have.values()):
                break
            rows = page(off, 10)
            read += len(rows)
            for j, row in enumerate(rows):
                cid = f"r{off + j}"
                text = row["transcript"].strip()
                dur = float(row["duration"])
                k = kind(text)
                if cid in seen or k is None or have[k] >= a.per or not (a.min_dur <= dur <= a.max_dur):
                    continue
                x = np.asarray(row["audio"], dtype=np.float32)
                sr = int(float(row["samplerate"]))
                g = np.gcd(24000, sr)
                x = resample_poly(x, 24000 // g, sr // g).astype(np.float32)
                sf.write(OUT / "wav" / f"{cid}.wav", x, 24000)
                f.write(f"{cid}\t{k}\t{dur:.2f}\t{text}\n")
                f.flush()
                have[k] += 1
            print(f"off {off} read {read} have {have}", flush=True)
    print("done", have, "rows read", read)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
