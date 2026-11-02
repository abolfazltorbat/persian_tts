"""Re-score fluency-bench renders with the second judge (wav2vec2 transcripts
from research/w2v_judge.py), Persian rows only. Writes
output/eval/flu_<tag>/report_w2v.json with the same row keys, so
research/paired_stats.py can compare tags under either judge (--judge w2v).

    python research/rescore_w2v.py TAG [TAG ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "research"))
import harness  # noqa: E402,F401  (paths, typer shim)
from scoring import score  # noqa: E402

EVAL = ROOT / "output" / "eval"


def refs_for(meta: dict) -> dict:
    s = meta.get("set") or "eval/cases.py"
    if s.endswith(".tsv"):
        from core import load_set
        return {c["id"]: c for c in load_set(ROOT / s if not Path(s).is_absolute() else s)}
    from cases import CASES
    return {c["id"]: c for c in CASES}


def main(tags: list[str]) -> int:
    for tag in tags:
        d = EVAL / f"flu_{tag}"
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        hyps = {}
        hp = d / "w2v_hyps.jsonl"
        for line in hp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                h = json.loads(line)
                hyps[h["id"]] = h["hyp_w2v"]
        refs = refs_for(rep["meta"])
        rows = []
        for r in rep["rows"]:
            if r["lang"] != "fa":
                continue
            key = f"{r['case']}__{r['voice']}__s{r['seed']}"
            if key not in hyps:
                continue
            sc = score(hyps[key], refs[r["case"]]["ref"], "fa")
            rows.append({**r, "hyp": hyps[key], **sc})
        (d / "report_w2v.json").write_text(json.dumps({"meta": {**rep["meta"], "judge": "w2v"}, "rows": rows},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(tag, "w2v rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
