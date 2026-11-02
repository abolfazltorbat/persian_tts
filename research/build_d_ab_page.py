"""Study D human evaluation: a blind A/B page, shipped engine vs. study D.

Pairs (same text, voice and seed; left/right at random):
  * whole texts:  output/research/dyn1/prod  vs  output/research/dyn2/final   (8 texts x 3 voices)
  * sentences:    output/research/dfinal/prod vs output/research/dfinal/new2  (held-out questions and
                  exclamations, 3 voices, seed 1; a balanced sample per type)
Two questions per pair, each A / same / B:
  Q1 expressiveness — which one sounds more like what the punctuation asks for (a question like a
     question, an exclamation or surprise like one, a text that is not monotone)?
  Q2 clarity — which one is clearer and more natural?
plus a word-error flag per clip. "Download results" saves results_<listener>.json.

    python research/build_d_ab_page.py --out output/research/listen_studyD_ab
    python research/build_d_ab_page.py --tally results_*.json      # one or more listeners
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research"
VOICES = ["male_hello_clean", "female_narration_clean", "male_news"]
PER_TYPE = {"qyn": 6, "qtag": 3, "qecho": 6, "qwh": 3, "qalt": 3, "exc": 6}


def build(a) -> int:
    sys.path.insert(0, str(ROOT / "research"))
    sys.path.insert(0, str(ROOT / "eval"))
    from core import load_set
    cases = {}
    for s in ("prosody_set", "prosody_set2"):
        cases.update({c["id"]: c for c in load_set(ROOT / f"research/sets/{s}.tsv")})
    rng = random.Random(a.shuffle_seed)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True)
    pairs = []
    for cid, c in cases.items():                                   # whole texts
        if c["cat"] == "dyn":
            for v in VOICES:
                pairs.append(("text", cid, c["text"], v, OUT / "dyn1/prod", OUT / "dyn2/final"))
    by_type = {}
    for cid, c in cases.items():                                   # held-out sentences
        num = int("".join(ch for ch in cid if ch.isdigit()) or 0)
        cat = "qyn" if (c["cat"] == "qyn" and cid.endswith("q")) else c["cat"]
        if num % 2 == 0 and cat in PER_TYPE:
            by_type.setdefault(cat, []).append(cid)
    for cat, ids in by_type.items():
        rng.shuffle(ids)
        for k, cid in enumerate(ids[:PER_TYPE[cat]]):
            v = VOICES[k % len(VOICES)]
            pairs.append((cat, cid, cases[cid]["text"], v, OUT / "dfinal/prod", OUT / "dfinal/new2"))
    rng.shuffle(pairs)
    items = []
    for i, (kind, cid, text, v, da, db) in enumerate(pairs):
        fa, fb = da / f"{cid}__{v}__s1.wav", db / f"{cid}__{v}__s1.wav"
        if not (fa.exists() and fb.exists()):
            continue
        swap = rng.random() < 0.5
        left, right = (fb, fa) if swap else (fa, fb)
        ln, rn = f"audio/{i:03d}_A.wav", f"audio/{i:03d}_B.wav"
        shutil.copyfile(left, out / ln)
        shutil.copyfile(right, out / rn)
        items.append({"i": i, "kind": kind, "case": cid, "voice": v, "text": text, "A": ln, "B": rn,
                      "A_is": "new" if swap else "shipped", "B_is": "shipped" if swap else "new"})

    def radio(name):
        return " ".join(f'<label><input type="radio" name="{name}" value="{v}">{t}</label>'
                        for v, t in (("A", "A"), ("same", "same"), ("B", "B")))

    rows = "\n".join(
        f'<tr><td>{n + 1}</td><td dir="rtl" class="t">{html.escape(it["text"]).replace(chr(10), " ¶ ")}</td>'
        f'<td><audio controls preload="none" src="{it["A"]}"></audio></td>'
        f'<td><audio controls preload="none" src="{it["B"]}"></audio></td>'
        f'<td>{radio("x" + str(it["i"]))}</td><td>{radio("c" + str(it["i"]))}</td>'
        f'<td><label><input type="checkbox" name="eA{it["i"]}">A</label> '
        f'<label><input type="checkbox" name="eB{it["i"]}">B</label></td></tr>'
        for n, it in enumerate(items))
    key = json.dumps(items, ensure_ascii=False)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Blind listening test</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1250px; padding:16px; }}
p {{ color:var(--ink2); max-width:860px; }}
table {{ border-collapse:collapse; width:100%; }} td, th {{ border-bottom:1px solid var(--line); padding:6px 6px; vertical-align:middle; }}
th {{ font-size:13px; color:var(--ink2); text-align:left; }} .t {{ font-size:16px; max-width:380px; }}
audio {{ width:190px; height:32px; }} label {{ white-space:nowrap; margin-right:4px; }}
@media (max-width:760px) {{ table, tr, td, th {{ display:block; }} th {{ display:none; }} audio {{ width:100%; }} }}
</style></head><body>
<h2>Blind listening test ({len(items)} pairs)</h2>
<p>Each row is the same text read by two versions of the voice, A and B, in random order. Listen to both,
then answer two questions. <b>Expressive:</b> which one sounds more like what the punctuation asks for —
a question like a question, an exclamation or surprise like one, a whole text less monotone?
<b>Clear:</b> which one is clearer and more natural? Choose "same" when you cannot tell. Tick "word error"
for a clip that drops, adds or garbles a word. When done, enter your name and press <b>Download results</b>.</p>
<p>Listener: <input id="who" placeholder="name"></p>
<table><tr><th>#</th><th>text</th><th>A</th><th>B</th><th>more expressive</th><th>clearer</th><th>word error</th></tr>
{rows}</table>
<p><button onclick="dl()">Download results</button></p>
<script>
const KEY = {key};
function pick(n) {{ const q = document.querySelector('input[name="' + n + '"]:checked'); return q ? q.value : null; }}
function dl() {{
  const who = (document.getElementById('who').value || 'listener').trim().replace(/[^\\w-]+/g, '_');
  const res = KEY.map(it => ({{...it, listener: who, expr: pick('x' + it.i), clear: pick('c' + it.i),
    errA: document.querySelector('input[name=eA' + it.i + ']').checked,
    errB: document.querySelector('input[name=eB' + it.i + ']').checked}}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(res, null, 1)], {{type: 'application/json'}}));
  a.download = 'results_' + who + '.json'; a.click();
}}
</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "key.json").write_text(key, encoding="utf-8")
    print(f"{len(items)} pairs -> {out / 'index.html'}")
    return 0


def tally(paths: list[str]) -> int:
    from scipy.stats import binomtest
    rows = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))]
    listeners = sorted({r.get("listener", "?") for r in rows})
    print(f"{len(listeners)} listener(s): {', '.join(listeners)}; {len(rows)} judgements per question")
    for q, name in (("expr", "more expressive"), ("clear", "clearer / more natural")):
        for group in ["all", "text"] + list(PER_TYPE):
            sel = [r for r in rows if group == "all" or r["kind"] == group]
            new = sum(1 for r in sel if r.get(q) in ("A", "B") and r[r[q] + "_is"] == "new")
            old = sum(1 for r in sel if r.get(q) in ("A", "B") and r[r[q] + "_is"] == "shipped")
            same = sum(1 for r in sel if r.get(q) == "same")
            if new + old + same == 0:
                continue
            p = binomtest(new, new + old).pvalue if new + old else float("nan")
            print(f"  {name:24s} {group:6s} new {new:3d} | same {same:3d} | shipped {old:3d}   sign test p = {p:.4f}")
    err = {"new": 0, "shipped": 0}
    for r in rows:
        for side in ("A", "B"):
            if r.get("err" + side):
                err[r[side + "_is"]] += 1
    print("  word-error flags:", err)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "listen_studyD_ab"))
    ap.add_argument("--shuffle-seed", type=int, default=11)
    ap.add_argument("--tally", nargs="*", default=None)
    a = ap.parse_args()
    return tally(a.tally) if a.tally else build(a)


if __name__ == "__main__":
    raise SystemExit(main())
