"""Second-judge labels for a take corpus (study A) and for probe runs (study B).

Whisper hallucinates on short audio («موسیقی» on real 2 s speech; ASR floor:
research/asr_floor.py), so every label is recomputed with the wav2vec2 judge
and combined:  cer_cons = min(cer_whisper, cer_w2v)  — a take fails only if
BOTH recognisers say so (a hallucination of one judge cannot fail a take).

    python research/rejudge_corpus.py decode corpus_dev        # latents -> trimmed WAVs (fa rows)
    .venv-qwen/bin/python research/w2v_judge.py --wavs output/research/corpus_dev/wav \\
        --out output/research/corpus_dev/w2v_hyps.jsonl
    python research/rejudge_corpus.py merge corpus_dev         # -> rows_judged.jsonl
    python research/rejudge_corpus.py merge-probe prompt_ext3  # probe rows + wav/ -> rows_judged.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402


def decode(name: str) -> int:
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor

    d = core.OUT / name
    (d / "wav").mkdir(exist_ok=True)
    rows = core.read_jsonl(d / "rows.jsonl")
    lab = core.Lab(asr=False)

    def one(r):
        safe = r["rid"].replace("|", "__")
        out = d / "wav" / f"{safe}.wav"
        if out.exists() or r["lang"] != "fa":
            return 0
        z = np.load(d / "lat" / f"{safe}.npy").astype(np.float32)
        tk = core.Take(chunk=r["chunk"], z=z, eos_step=r["eos_step"], max_gen=r["max_gen"])
        audio = core.decode(lab, tk)
        speech, _ = core.engine_qc(lab, audio, r["chunk"])
        sf.write(out, speech, lab.sr)
        return 1

    with ThreadPoolExecutor(max_workers=4) as ex:
        n = sum(ex.map(one, rows))
    print(f"{name}: wrote {n} wavs")
    return 0


def _w2v(d: Path) -> dict:
    hyps = {}
    p = d / "w2v_hyps.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h = json.loads(line)
            hyps[h["id"]] = h["hyp_w2v"]
    return hyps


def merge(name: str, probe: bool = False) -> int:
    sys.path.insert(0, str(core.ROOT / "scripts"))
    from verify import skeleton_cer

    d = core.OUT / name
    rows = core.read_jsonl(d / "rows.jsonl")
    hyps = _w2v(d)
    out = []
    for r in rows:
        key = r.get("wid") if probe else r["rid"].replace("|", "__")
        if key not in hyps:
            continue
        lang = r.get("lang", "fa")
        if lang != "fa":
            continue
        cw = float(skeleton_cer(r["chunk"], hyps[key], "fa")) if hyps[key].strip() else 1.0
        out.append({**r, "cer_whisper": r["cer"], "hyp_w2v": hyps[key], "cer_w2v": round(cw, 4),
                    "cer_cons": round(min(r["cer"], cw), 4)})
    with (d / "rows_judged.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import statistics as st
    for k in ("cer_whisper", "cer_w2v", "cer_cons"):
        v = [r[k] for r in out]
        print(f"{name} {k}: mean {st.mean(v):.4f}  fail>.3 {sum(x > 0.3 for x in v) / len(v):.4f}  (n={len(v)})")
    return 0


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "decode":
        raise SystemExit(decode(name))
    if cmd == "merge":
        raise SystemExit(merge(name))
    if cmd == "merge-probe":
        raise SystemExit(merge(name, probe=True))
    raise SystemExit(f"unknown command {cmd}")
