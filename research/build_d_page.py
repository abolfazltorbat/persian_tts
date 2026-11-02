"""Study D listening page (labelled, not blind): NOW = the shipped engine (prompt
self-extension + study C intonation rule), NEW = study D (prosody-styled prompts +
the v2 rule + loudness plan). Same words, same seed, same voice.

  Part 1  whole texts (output/research/dyn1/{prod,v2s})   — does the voice change through a text?
  Part 2  single sentences by type (output/research/dfinal/{prod,new}; the statement twin from base)

    python research/build_d_page.py --out output/research/listen_studyD
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
VOICES = ["male_hello_clean", "female_narration_clean", "male_news"]
TYPE_NAME = {"qyn": "yes/no question", "qshort": "short question", "qecho": "surprised question",
             "qwh": "wh-question", "qalt": "alternative question (should fall)", "qtag": "tag question",
             "exc": "exclamation"}


def main() -> int:
    from core import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output/research/listen_studyD"))
    ap.add_argument("--seed", default="s1")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cases = []
    for s in ("prosody_set", "prosody_set2"):
        cases += load_set(ROOT / f"research/sets/{s}.tsv")
    by_id = {c["id"]: c for c in cases}

    def cell(src: Path, label: str) -> str:
        if not src.exists():
            return "<td>—</td>"
        dst = f"audio/{label}_{src.name}"
        shutil.copyfile(src, out / dst)
        return f'<td><audio controls preload="none" src="{dst}"></audio></td>'

    body = ["<h2>1. Whole texts</h2><p>Listen for the questions, the surprise and the loudness changing "
            "from sentence to sentence. Paragraph breaks are marked ¶.</p>"]
    dyn = ROOT / "output/research/dyn2"
    for voice in VOICES:
        body.append(f"<h3>{html.escape(voice)}</h3><table><tr><th>text</th><th>NOW (shipped)</th>"
                    "<th>NEW</th></tr>")
        for c in cases:
            if c["cat"] != "dyn":
                continue
            name = f"{c['id']}__{voice}__{a.seed}.wav"
            txt = html.escape(c["text"]).replace("\n\n", " ¶ ")
            body.append(f"<tr><td class='fa'>{txt}</td>{cell(dyn / 'prod' / name, 'now')}"
                        f"{cell(dyn / 'final' / name, 'new')}</tr>")
        body.append("</table>")
    body.append("<h2>2. Single sentences</h2>")
    fin = ROOT / "output/research/dfinal"
    for voice in VOICES:
        body.append(f"<h3>{html.escape(voice)}</h3><table><tr><th>type</th><th>sentence</th>"
                    "<th>statement «.» twin</th><th>NOW (shipped)</th><th>NEW</th></tr>")
        for c in cases:
            if c["cat"] in ("st", "dyn"):
                continue
            name = f"{c['id']}__{voice}__{a.seed}.wav"
            twin = c["id"][:-1] + "s" if c["cat"] == "qyn" else None
            st = cell(fin / "base" / f"{twin}__{voice}__{a.seed}.wav", "st") if twin in by_id else "<td>—</td>"
            body.append(f"<tr><td>{TYPE_NAME.get(c['cat'], c['cat'])}</td><td class='fa'>{html.escape(c['text'])}"
                        f"</td>{st}{cell(fin / 'prod' / name, 'now')}{cell(fin / 'new2' / name, 'new')}</tr>")
        body.append("</table>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Expressive prosody: now and new</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }}
h1 {{ font-size:22px; margin:8px 0; }} h2 {{ font-size:19px; margin:32px 0 6px; }} h3 {{ font-size:16px; margin:22px 0 6px; }}
p {{ color:var(--ink2); max-width:820px; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:middle; }}
th {{ font-size:13px; color:var(--ink2); font-weight:600; }}
td.fa {{ direction:rtl; font-size:16px; }}
audio {{ width:220px; height:32px; }}
@media (max-width:760px) {{ table, tr, td, th {{ display:block; }} th {{ display:none; }} audio {{ width:100%; }} }}
</style></head><body>
<h1>Expressive prosody: NOW vs NEW</h1>
<p>NOW is the engine as shipped (prompt self-extension, study C question rise). NEW adds study D: exclamations and
surprised questions are generated from a livelier voice prompt (the model itself speaks higher and with a wider
pitch range — the body of those sentences is not pitch-processed); the audio rule gives
yes/no questions a rise on the last syllable, keeps wh-questions from falling like statements, lets alternative
questions fall, keeps the energy up in the second half of excited sentences, and plans loudness over the text
(exclamations louder, a paragraph's first sentence a little louder, its last a little softer).</p>
{''.join(body)}
</body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print("wrote", out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
