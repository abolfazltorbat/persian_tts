"""Study D — the recogniser as a listener for SENTENCE TYPE.

Whisper writes punctuation, and for a Persian yes/no question the words give it
no clue: «فایل را ذخیره کردید؟» and «فایل را ذخیره کردید.» are the same words.
Whether it ends its transcript with «؟» is then decided by the sound — a
reference-free proxy for "does this sound like a question?". The statement
twin (same words) is the lexical baseline; the difference between the two is
what the intonation carries.

    python research/punct_judge.py DIR_OR_WAV... --out file.jsonl [--set prosody_set.tsv]
    python research/punct_judge.py --real               # real GPTInformal clips by type

Per WAV: the transcript and its final mark ('?' '!' '.' or '-'). With --set the
case id (file stem up to "__") gives the written type. Runs in the chatbot venv
(faster-whisper large-v3-turbo, the same judge as eval/fluency.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
REAL = Path.home() / ".cache" / "persian_tts_ft" / "data_gptinformal_types"


def final_mark(hyp: str) -> str:
    h = hyp.strip().rstrip(" »\"')")
    if h.endswith(("؟", "?")):
        return "?"
    if h.endswith("!"):
        return "!"
    if h.endswith((".", "۔", "…")):
        return "."
    return "-"


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(Counter)
    for r in rows:
        by[r.get("kind", "?")][r["mark"]] += 1
    return {k: {"n": sum(c.values()), "q_rate": round(c["?"] / max(1, sum(c.values())), 3),
                "x_rate": round(c["!"] / max(1, sum(c.values())), 3), **dict(c)} for k, c in sorted(by.items())}


def main() -> int:
    import soundfile as sf
    import harness

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", default="")
    ap.add_argument("--set", default=",".join(str(ROOT / f"research/sets/{n}.tsv")
                                              for n in ("prosody_set", "prosody_set2")))
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    kinds = {}
    wavs = []
    refs = {}
    if a.real:
        import intonation as I
        for line in (REAL / "clips.tsv").read_text(encoding="utf-8").splitlines():
            cid, k, _d, text = line.split("\t", 3)
            kinds[cid] = {"q": I.sentence_kind(text.replace(" ؟", "؟")), "x": "exc", "s": "st"}[k]
            wavs.append(REAL / "wav" / f"{cid}.wav")
        out = Path(a.out or ROOT / "output/research/punct/real.jsonl")
    else:
        from core import load_set
        for sp in a.set.split(","):
            for c in load_set(sp):
                kinds[c["id"]] = c["cat"]
                refs[c["id"]] = c
        for p in map(Path, a.paths):
            wavs += sorted(p.glob("*.wav")) if p.is_dir() else [p]
        out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    asr = harness.Asr(a.device, "turbo")
    rows = []
    for w in wavs:
        x, sr = sf.read(str(w), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        hyp = asr.transcribe(x, sr, "fa")
        case = w.stem.split("__")[0]
        row = {"wav": str(w), "name": w.name, "case": case, "kind": kinds.get(case, "?"), "hyp": hyp,
               "mark": final_mark(hyp)}
        if case in refs:
            from scoring import score
            row["cer"] = score(hyp, refs[case]["ref"], refs[case]["lang"])["cer"]
        rows.append(row)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=1))
    print("wrote", len(rows), "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
