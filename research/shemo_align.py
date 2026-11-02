"""Attach transcripts to the ShEMO clips (research/fetch_shemo.py): the audio and
text mirrors are in different row orders, so each clip is transcribed with
Whisper and matched to the closest of the 3000 texts with the SAME emotion
label (character 3-gram cosine). Kept when the best match is clearly better
than the runner-up. Type from the matched text's final mark (intonation.sentence_plan).

    python research/shemo_align.py   -> ~/.cache/persian_tts_ft/shemo/aligned.tsv
       (id, emotion, kind, score, text, whisper)
"""
import collections
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
D = Path.home() / ".cache/persian_tts_ft/shemo"


def grams(t: str) -> collections.Counter:
    t = re.sub(r"[^\w]+", " ", t.replace("‌", "").replace("‎", "")).strip()
    t = f" {t} "
    return collections.Counter(t[i:i + 3] for i in range(len(t) - 2))


def cos(a, b) -> float:
    num = sum(v * b.get(k, 0) for k, v in a.items())
    return num / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values())) + 1e-9)


def main() -> int:
    import soundfile as sf
    import harness
    import intonation as I
    texts = collections.defaultdict(list)
    for line in (D / "transcripts.tsv").read_text(encoding="utf-8").splitlines():
        emo, t = line.split("\t", 1)
        texts[emo].append((t, grams(t)))
    asr = harness.Asr("auto", "turbo")
    out = []
    for line in (D / "clips.tsv").read_text(encoding="utf-8").splitlines():
        cid, emo, _dur = line.split("\t")[:3]
        x, sr = sf.read(D / "wav" / f"{cid}.wav", dtype="float32")
        hyp = asr.transcribe(x, sr, "fa")
        g = grams(hyp)
        sc = sorted(((cos(g, tg), t) for t, tg in texts[emo]), reverse=True)
        best, t = sc[0]
        second = sc[1][0] if len(sc) > 1 else 0.0
        if best < 0.45 or best - second < 0.10:
            continue
        kind = I.sentence_plan(t)[0]
        out.append(f"{cid}\t{emo}\t{kind}\t{best:.2f}\t{t}\t{hyp}\n")
    (D / "aligned.tsv").write_text("".join(out), encoding="utf-8")
    c = collections.Counter((l.split("\t")[1], l.split("\t")[2]) for l in out)
    print("aligned", len(out), sorted(c.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
