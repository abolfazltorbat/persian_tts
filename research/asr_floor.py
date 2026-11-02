"""ASR noise floor on REAL short speech (GPTInformal-Persian clips, CC0):
how much CER does each judge charge to recordings that are correct by
definition? Whisper (the bench judge: beam 5, digits suppressed) vs the
wav2vec2-CTC second judge (hyps from research/w2v_judge.py).

    python research/asr_floor.py
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import core  # noqa: E402
from scoring import score  # noqa: E402
from verify import _edits, text_skeleton  # noqa: E402

D = Path.home() / ".cache" / "persian_tts_ft" / "data_gptinformal"


def sk_cer(ref: str, hyp: str) -> float:
    r, h = text_skeleton(ref, "fa"), text_skeleton(hyp, "fa")
    return min(1.5, _edits(h, r) / max(1, len(r)))


def main() -> int:
    import soundfile as sf
    lab = core.Lab()
    items = {}
    for line in (D / "short_real.tsv").read_text(encoding="utf-8").splitlines():
        cid, dur, text = line.split("\t", 2)
        items[cid] = (float(dur), text)
    w2v = {}
    for line in (D / "w2v_hyps.jsonl").read_text(encoding="utf-8").splitlines():
        h = json.loads(line)
        w2v[h["id"]] = h["hyp_w2v"]
    rows = []
    for cid, (dur, text) in items.items():
        wav = D / "wav" / f"{cid}.wav"
        if not wav.exists() or cid not in w2v:
            continue
        a, sr = sf.read(wav)
        hw = lab.judge.transcribe(a.astype("float32"), sr, "fa")
        ref = core._PUNCT.sub(" ", text).strip()
        rows.append({"id": cid, "dur": dur, "ref": ref, "whisper": hw, "w2v": w2v[cid],
                     "cer_whisper": score(hw, ref, "fa")["cer"], "cer_w2v": score(w2v[cid], ref, "fa")["cer"],
                     "sk_whisper": sk_cer(ref, hw), "sk_w2v": sk_cer(ref, w2v[cid])})
    out = {"n": len(rows), "mean_dur": round(st.mean(r["dur"] for r in rows), 2)}
    for k in ("cer_whisper", "cer_w2v", "sk_whisper", "sk_w2v"):
        v = [r[k] for r in rows]
        out[k] = {"mean": round(st.mean(v), 4), "fail>.3": round(sum(x > 0.3 for x in v) / len(v), 4)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    for r in sorted(rows, key=lambda r: -r["sk_whisper"])[:8]:
        print(f"  {r['ref']} | W: {r['whisper']} | w2v: {r['w2v']}")
    (core.OUT / "asr_floor.json").write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
