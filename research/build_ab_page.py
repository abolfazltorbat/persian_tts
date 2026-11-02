"""Blind A/B listening page for two fluency-bench tags (renders must be kept:
KEEP_AUDIO=1). For each case one seed is shown; the two systems are placed
left/right at random. The listener picks A / same / B per pair (and may flag
a word error); "Download results" saves a JSON that `--tally` scores.

    python research/build_ab_page.py b3s_base b3s_ext --seed 1 --out output/research/listen_b3s
    python research/build_ab_page.py --tally output/research/listen_b3s/results.json
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "output" / "eval"


def build(a) -> int:
    base = json.loads((EVAL / f"flu_{a.base}" / "report.json").read_text(encoding="utf-8"))
    new = json.loads((EVAL / f"flu_{a.new}" / "report.json").read_text(encoding="utf-8"))
    texts = {}
    s = base["meta"].get("set", "")
    cats = set(a.cats.split(",")) if a.cats else None
    if s.endswith(".tsv"):
        for line in (ROOT / s).read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                cid, _l, _c, t = line.split("\t", 3)
                texts[cid] = t
    else:
        import sys
        sys.path.insert(0, str(ROOT / "eval"))
        from cases import CASES
        texts = {c["id"]: c["text"] for c in CASES}
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.shuffle_seed)
    items = []
    voice = base["rows"][0]["voice"]
    cases = sorted({r["case"] for r in base["rows"] if cats is None or r["cat"] in cats})
    rng.shuffle(cases)
    for i, cid in enumerate(cases):
        root_a = Path(a.wav_root) / a.base if a.wav_root else EVAL / f"flu_{a.base}"
        root_b = Path(a.wav_root) / a.new if a.wav_root else EVAL / f"flu_{a.new}"
        fa = root_a / f"{cid}__{voice}__s{a.seed}.wav"
        fb = root_b / f"{cid}__{voice}__s{a.seed}.wav"
        if not (fa.exists() and fb.exists()):
            continue
        swap = rng.random() < 0.5
        left, right = (fb, fa) if swap else (fa, fb)
        ln, rn = f"audio/{i:03d}_L.wav", f"audio/{i:03d}_R.wav"
        shutil.copyfile(left, out / ln)
        shutil.copyfile(right, out / rn)
        items.append({"i": i, "case": cid, "text": texts.get(cid, cid), "L": ln, "R": rn,
                      "L_is": a.new if swap else a.base, "R_is": a.base if swap else a.new})
    rows = "\n".join(
        f'<tr><td>{it["i"] + 1}</td><td dir="rtl" class="t">{html.escape(it["text"])}</td>'
        f'<td><audio controls preload="none" src="{it["L"]}"></audio></td>'
        f'<td><audio controls preload="none" src="{it["R"]}"></audio></td>'
        f'<td><label><input type="radio" name="q{it["i"]}" value="L">A</label> '
        f'<label><input type="radio" name="q{it["i"]}" value="same">same</label> '
        f'<label><input type="radio" name="q{it["i"]}" value="R">B</label></td>'
        f'<td><label><input type="checkbox" name="eL{it["i"]}">A</label> '
        f'<label><input type="checkbox" name="eR{it["i"]}">B</label></td></tr>'
        for it in items)
    key = json.dumps(items, ensure_ascii=False)
    page = f"""<!doctype html><meta charset="utf-8"><title>A/B listening</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:1100px}}td{{padding:4px 8px;border-bottom:1px solid #ddd}}
.t{{font-size:18px}}audio{{width:220px}}</style>
<h2>Blind A/B listening test ({len(items)} pairs)</h2>
<p>For each sentence, play A and B. Pick the one that reads the sentence more correctly and naturally
("same" if you cannot tell). Tick "word error" for any clip that drops, adds or garbles a word.
Then press <b>Download results</b> and send the file.</p>
<table><tr><th>#</th><th>sentence</th><th>A</th><th>B</th><th>better</th><th>word error</th></tr>
{rows}</table>
<p><button onclick="dl()">Download results</button></p>
<script>
const KEY = {key};
function dl() {{
  const res = KEY.map(it => {{
    const q = document.querySelector('input[name=q' + it.i + ']:checked');
    return {{...it, choice: q ? q.value : null,
      errL: document.querySelector('input[name=eL' + it.i + ']').checked,
      errR: document.querySelector('input[name=eR' + it.i + ']').checked}};
  }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(res, null, 1)], {{type: 'application/json'}}));
  a.download = 'results.json'; a.click();
}}
</script>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} pairs -> {out / 'index.html'}")
    return 0


def tally(path: str) -> int:
    from scipy.stats import binomtest

    res = json.loads(Path(path).read_text(encoding="utf-8"))
    wins = {}
    errs = {}
    for r in res:
        if r["choice"] in ("L", "R"):
            w = r["L_is"] if r["choice"] == "L" else r["R_is"]
            wins[w] = wins.get(w, 0) + 1
        elif r["choice"] == "same":
            wins["same"] = wins.get("same", 0) + 1
        for side in ("L", "R"):
            if r.get("err" + side):
                errs[r[side + "_is"]] = errs.get(r[side + "_is"], 0) + 1
    systems = [k for k in wins if k != "same"]
    print("preferences:", wins, "| word-error flags:", errs)
    if len(systems) == 2:
        a, b = systems
        n = wins[a] + wins[b]
        print(f"sign test {a} vs {b}: p = {binomtest(wins[a], n).pvalue:.4f} (n={n}, ties excluded)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shuffle-seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "output" / "research" / "listen"))
    ap.add_argument("--tally", default="")
    ap.add_argument("--wav-root", default="", help="dir holding <tag>/ WAV folders (e.g. output/research/keep_wav)")
    ap.add_argument("--cats", default="", help="only these case categories")
    a = ap.parse_args()
    return tally(a.tally) if a.tally else build(a)


if __name__ == "__main__":
    raise SystemExit(main())
