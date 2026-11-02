"""Study C — apply scripts/intonation.py to kept bench renders (offline, same
audio the engine produced) so rule settings can be compared without
re-synthesis. One-sentence cases: the bench WAV is exactly the sentence audio
the engine's join() would process.

    python research/intonation_eval.py c_base --out int_v1 [--set ...] [--knob MAX_SHIFT_ST=9 ...]
    .venv-qwen/bin/python research/prosody.py output/research/int_v1/wav --out output/research/int_v1/prosody.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))


def main() -> int:
    import soundfile as sf
    import intonation as I
    from core import load_set

    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default=str(ROOT / "research/sets/prosody_set.tsv"))
    ap.add_argument("--knob", action="append", default=[], help="NAME=value module constant")
    ap.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    for kv in a.knob:
        k, v = kv.split("=")
        if not hasattr(I, k):
            raise SystemExit(f"intonation has no {k}")
        setattr(I, k, float(v))
    cases = {c["id"]: c for c in load_set(a.set)}
    out = ROOT / "output" / "research" / a.out / "wav"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for w in sorted((ROOT / "output" / "eval" / f"flu_{a.tag}").glob("*.wav")):
        case = w.stem.split("__")[0]
        if case not in cases:
            continue
        kind = I.sentence_kind(cases[case]["text"])
        x, sr = sf.read(str(w), dtype="float32")
        y = I.apply(x, sr, kind, a.strength, cases[case]["text"])
        sf.write(out / w.name, y, sr)
        meta.append({"wav": w.name, "case": case, "kind": kind, "cat": cases[case]["cat"]})
    (out.parent / "meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta),
                                           encoding="utf-8")
    (out.parent / "knobs.json").write_text(json.dumps({"knob": a.knob, "strength": a.strength}), encoding="utf-8")
    print("wrote", len(meta), "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
