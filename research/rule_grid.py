"""Study D — apply rule variants of scripts/intonation.py to kept single-sentence
renders (offline: the bench WAV is exactly what frontend.join would process).

    python research/rule_grid.py SRC_DIR --out grid1 --var A:Q_END_ST=5,Q_LEN=0,Q_SUSTAIN_DB=0 --var v1:VERSION=1 ...
        [--cases p,e] [--split dev]
Each variant -> output/research/<out>/<name>/<same file names>.wav. Knobs are module
constants of intonation.py (VERSION=1 runs the study C rule through apply()).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "eval"))


def main() -> int:
    import importlib
    import soundfile as sf
    from core import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--cases", default="", help="case-id prefixes")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    a = ap.parse_args()
    cases = {}
    for s in ("prosody_set", "prosody_set2"):
        cases.update({c["id"]: c for c in load_set(ROOT / f"research/sets/{s}.tsv")})
    wavs = []
    for w in sorted(Path(a.src).glob("*.wav")):
        cid = w.stem.split("__")[0]
        if cid not in cases or (a.cases and not cid.startswith(tuple(a.cases.split(",")))):
            continue
        num = int("".join(ch for ch in cid if ch.isdigit()) or 0)
        if a.split != "all" and (num % 2 == 0) != (a.split == "test"):
            continue
        wavs.append((w, cases[cid]))
    for spec in a.var:
        name, _, kv = spec.partition(":")
        import intonation as I
        I = importlib.reload(I)
        for item in filter(None, kv.split(",")):
            k, v = item.split("=")
            if not hasattr(I, k):
                raise SystemExit(f"intonation has no {k}")
            setattr(I, k, type(getattr(I, k))(float(v)) if not isinstance(getattr(I, k), bool) else v == "1")
        out = ROOT / "output/research" / a.out / name
        out.mkdir(parents=True, exist_ok=True)
        for w, c in wavs:
            x, sr = sf.read(str(w), dtype="float32")
            if I.VERSION >= 2:
                y = I.process(x, sr, c["text"], 1.0)
            else:
                k = I.sentence_kind(c["text"])
                y = I.apply(x, sr, k, 1.0, c["text"]) if k in I.MODIFIED else x
            sf.write(out / w.name, y, sr)
        print(name, len(wavs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
