"""Study D — an EXPRESSIVE real-speech reference: ShEMO (Sharif Emotional Speech
Database; Mohamad Nezami et al. 2019; radio plays, 87 speakers; free for
academic use). Only measured here (F0 / energy statistics), never trained on or
redistributed. The HF mirror minoosh/shEMO_nosplits has (audio, emotion) for all
3000 utterances; minoosh/shEMO_transcripts has (text, emotion) in ANOTHER order,
so texts are attached later by recogniser match (research/shemo_align.py).

    python research/fetch_shemo.py          # neutral, happy, surprised utterances
-> ~/.cache/persian_tts_ft/shemo/{wav/<i>.wav (24 kHz), clips.tsv (id, emotion, dur)},
   transcripts.tsv (all 3000 texts with their emotion)
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

API = "https://datasets-server.huggingface.co/rows"
OUT = Path.home() / ".cache" / "persian_tts_ft" / "shemo"
EMO = ["A", "H", "N", "S", "W", "F"]


def rows(ds: str, split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": ds, "config": "default", "split": split,
                                "offset": offset, "length": length})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=240) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except Exception as exc:
            print("retry", ds, split, offset, exc, flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def main() -> int:
    import soundfile as sf
    from scipy.signal import resample_poly
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    tt = OUT / "transcripts.tsv"
    if not tt.exists():
        lines = []
        for split, n in (("train", 2400), ("test", 300), ("valid", 300)):
            for off in range(0, n, 100):
                for t in rows("minoosh/shEMO_transcripts", split, off, 100):
                    text = " ".join(t["transcription"].replace("\ufeff", "").split())
                    lines.append(f"{EMO[t['emotion']]}\t{text}\n")
        tt.write_text("".join(lines), encoding="utf-8")
        print("transcripts", len(lines), flush=True)
    tsv = OUT / "clips.tsv"
    done = set()
    if tsv.exists():
        done = {l.split("\t")[0] for l in tsv.read_text(encoding="utf-8").splitlines()}
    keep_n = 0
    with tsv.open("a", encoding="utf-8") as f:
        for off in range(0, 3000, 100):
            for j, a in enumerate(rows("minoosh/shEMO_nosplits", "train", off, 100)):
                cid = f"s{off + j}"
                emo = EMO[a["emotion"]]
                # every surprised / happy utterance, every third neutral one
                if cid in done or emo not in ("W", "H", "N") or (emo == "N" and (off + j) % 3):
                    continue
                src = a["audio"][0]["src"]
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen(src, timeout=120) as r:
                            x, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
                        break
                    except Exception as exc:
                        print("wav retry", cid, exc, flush=True)
                        time.sleep(3 * (attempt + 1))
                else:
                    continue
                if x.ndim > 1:
                    x = x.mean(axis=1)
                x = resample_poly(x, 3, 2).astype(np.float32) if sr == 16000 else x
                sf.write(OUT / "wav" / f"{cid}.wav", x, 24000)
                f.write(f"{cid}\t{emo}\t{len(x) / 24000:.2f}\n")
                f.flush()
                keep_n += 1
            print(off, keep_n, flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
