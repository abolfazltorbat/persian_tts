"""Study D — assemble the per-case source renders for the final comparison
(symlinks; one file per <case>__<voice>__s<seed>):
  src_base : product config, no audio rule (flu_d_base + flu_d2_base)
  src_sty  : the same sentences from the styled prompts the new method uses —
             statements / alternative questions from src_base (no style), yes/no + wh
             questions from flu_d_sty / flu_d2_sty ("ask"), exclamations + surprised
             questions from flu_d_sty3 / flu_d2_sty3 ("lively"; the female voice from
             flu_d_sty4f / flu_d2_sty4f, half register shift), one-word questions of
             set 1 from flu_d_sty4q
    python research/build_d_dirs.py --out output/research/dfinal
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "eval"))


def main() -> int:
    import intonation as I
    from core import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output/research/dfinal"))
    a = ap.parse_args()
    cases = {}
    for s in ("prosody_set", "prosody_set2"):
        cases.update({c["id"]: c for c in load_set(ROOT / f"research/sets/{s}.tsv")})
    ev = ROOT / "output/eval"
    out = Path(a.out)
    for d in ("src_base", "src_sty"):
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in (out / d).glob("*.wav"):
            f.unlink()
    n = {"base": 0, "sty": 0}
    for tag in ("d_base", "d2_base"):
        for w in sorted((ev / f"flu_{tag}").glob("*.wav")):
            cid = w.stem.split("__")[0]
            if cid not in cases or cases[cid]["cat"] == "dyn":
                continue
            (out / "src_base" / w.name).symlink_to(w)
            n["base"] += 1
            style = I.STYLE_OF_KIND.get(I.sentence_plan(cases[cid]["text"])[0])
            female = "__female" in w.name
            src = w
            if cases[cid]["cat"] == "qshort" and style:
                src = ev / "flu_d_sty4q" / w.name          # current styles, all voices
            elif style == "ask":
                src = ev / ("flu_d_sty" if tag == "d_base" else "flu_d2_sty") / w.name
            elif style == "lively" and female:             # high-voice register scaling
                src = ev / ("flu_d_sty4f" if tag == "d_base" else "flu_d2_sty4f") / w.name
            elif style == "lively":
                src = ev / ("flu_d_sty3" if tag == "d_base" else "flu_d2_sty3") / w.name
            if not src.exists():
                print("missing", src)
                continue
            (out / "src_sty" / w.name).symlink_to(src)
            n["sty"] += 1
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
