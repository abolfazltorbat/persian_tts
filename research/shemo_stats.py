"""Study D — prosody of EXPRESSIVE real speech (ShEMO) by emotion and sentence type,
with the same measurements as study C (research/prosody.py, pYIN). Runs in .venv-qwen.

    .venv-qwen/bin/python research/shemo_stats.py  -> output/research/prosody/shemo.jsonl + table
Register (median F0) is compared within gender (median F0 < 165 Hz = male), since
the mirror carries no speaker ids."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
D = Path.home() / ".cache/persian_tts_ft/shemo"


def main() -> int:
    import soundfile as sf
    from prosody import measure
    kinds = {}
    if (D / "aligned.tsv").exists():
        for l in (D / "aligned.tsv").read_text(encoding="utf-8").splitlines():
            cid, emo, kind, *_ = l.split("\t")
            kinds[cid] = kind
    out = ROOT / "output/research/prosody/shemo.jsonl"
    rows = []
    done = {}
    if out.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in out.read_text().splitlines() if l.strip()}
    for l in (D / "clips.tsv").read_text(encoding="utf-8").splitlines():
        cid, emo, dur = l.split("\t")[:3]
        if cid in done:
            r = done[cid]
        else:
            x, sr = sf.read(D / "wav" / f"{cid}.wav", dtype="float32")
            x16 = x[::1]
            r = {"id": cid, "emo": emo, **measure(x16, sr)}
            # loudness contour inside the utterance: first vs second half (dB)
            e = 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, np.ones(480) / 480, mode="same") + 1e-12)
            sp = np.flatnonzero(e > e.max() - 35)
            if len(sp) > 10:
                h = sp[len(sp) // 2]
                a = x[sp[0]:h]; b = x[h:sp[-1]]
                r["half_db"] = float(10 * np.log10(np.mean(b ** 2) + 1e-12) - 10 * np.log10(np.mean(a ** 2) + 1e-12))
        r["kind"] = kinds.get(cid, "?")
        rows.append(r)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    def stat(sel, k):
        v = [r[k] for r in sel if isinstance(r.get(k), (int, float)) and not np.isnan(r[k])]
        return (np.mean(v), len(v)) if v else (np.nan, 0)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["emo"], "*")].append(r)
        groups[(r["emo"], r["kind"])].append(r)
    print(f"{'emo':4s} {'kind':6s} {'n':>4s} {'end_st':>7s} {'slope':>7s} {'range':>6s} {'sd':>5s} {'e_rng':>6s} {'half':>6s}")
    for (emo, kind), sel in sorted(groups.items()):
        if len(sel) < 3:
            continue
        vals = [stat(sel, k)[0] for k in ("end_st", "end_slope", "range_st", "sd_st", "e_range", "half_db")]
        print(f"{emo:4s} {kind:6s} {len(sel):4d} " + " ".join(f"{v:7.2f}" for v in vals))
    for g, lo, hi in (("female", 165, 600), ("male", 0, 165)):
        med = {e: [r["f0_med"] for r in rows if r["emo"] == e and isinstance(r.get("f0_med"), float)
                   and lo <= r["f0_med"] < hi] for e in ("N", "W", "H")}
        if med["N"]:
            n = np.median(med["N"])
            print(g, " ".join(f"{e}: {12 * np.log2(np.median(v) / n):+.2f} st (n={len(v)})" for e, v in med.items() if v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
