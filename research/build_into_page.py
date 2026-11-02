"""Study C listening page (labelled, not blind): for every question / exclamation
of the prosody set, the statement twin (where there is one), the question as the
engine renders it today (identical to the statement) and with sentence-type
intonation — three voices, seed 1.

    python research/build_into_page.py c_base c_int --out output/research/listen_intonation
"""
from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICES = ["male_hello_clean", "female_narration_clean", "male_news"]
CAT_NAME = {"qyn": "yes/no question", "qshort": "short question", "qwh": "wh-question", "exc": "exclamation"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="s1")
    a = ap.parse_args()
    rows = []
    for line in (ROOT / "research/sets/prosody_set.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            cid, _l, cat, text = line.split("\t", 3)
            rows.append((cid, cat, text))
    text_of = {c: t for c, _k, t in rows}
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    def cp(tag, cid, voice, label):
        src = ROOT / "output/eval" / f"flu_{tag}" / f"{cid}__{voice}__{a.seed}.wav"
        if not src.exists():
            return "<td>—</td>"
        dst = f"audio/{label}_{cid}_{voice}.wav"
        shutil.copyfile(src, out / dst)
        return f'<td><audio controls preload="none" src="{dst}"></audio></td>'

    body = []
    for voice in VOICES:
        body.append(f"<h2>{html.escape(voice)}</h2><table><tr><th>type</th><th>sentence</th>"
                    "<th>statement «.» (unchanged)</th><th>BEFORE: question/exclamation today</th>"
                    "<th>AFTER: with intonation</th></tr>")
        for cid, cat, text in rows:
            if cat == "st":
                continue
            twin = cid[:-1] + "s" if cat == "qyn" else None
            st_cell = cp(a.base, twin, voice, "st") if twin and twin in text_of else "<td>—</td>"
            body.append(f"<tr><td>{CAT_NAME[cat]}</td><td class='fa'>{html.escape(text)}</td>{st_cell}"
                        f"{cp(a.base, cid, voice, 'before')}{cp(a.new, cid, voice, 'after')}</tr>")
        body.append("</table>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Intonation before and after</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }}
h1 {{ font-size:22px; margin:8px 0; }} h2 {{ font-size:18px; margin:28px 0 8px; }}
p {{ color:var(--ink2); max-width:780px; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:middle; }}
th {{ font-size:13px; color:var(--ink2); font-weight:600; }}
td.fa {{ direction:rtl; font-size:17px; white-space:nowrap; }}
audio {{ width:210px; height:32px; }}
@media (max-width:760px) {{ table, tr, td, th {{ display:block; }} th {{ display:none; }} audio {{ width:100%; }} }}
</style></head><body>
<h1>Sentence-type intonation: before and after</h1>
<p>The model never sees punctuation, so today a question is rendered exactly like its statement twin
(compare the first two audio columns of a yes/no question: they are the same file content).
The last column restores the contour from the written mark: a final rise for yes/no questions, no
deep final fall for wh-questions and exclamations. Same words, same seed, same voice.</p>
{''.join(body)}
</body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print("wrote", out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
