"""Multi-listener human evaluation (review points 1-2). One self-contained web page, ~15 minutes per
listener, three parts; every listener gets a different but balanced assignment (Latin square over
listener number L, so each sentence is heard once per listener, in one system's version):

  Part 1  sentence type + naturalness (single clips, text shown WITHOUT its final mark):
          is it a statement, a question, or an exclamation / surprise?  and MOS 1-5.
          yes/no questions, their statement twins, exclamations, surprised questions;
          shipped engine (study C rule) vs. ours (study D).
  Part 2  paired comparison (CMOS, -3..+3, A vs B) for naturalness, and which is more expressive;
          whole texts and held-out questions/exclamations; shipped vs. ours.
  Part 3  intelligibility of short replies (single clips): 1-5 "every word clear" + word-error flag;
          plain clip vs. prompt self-extension (held-out short set, default voice).

    python research/build_human_eval.py --out output/research/human_eval        # page + mp3 audio
    python research/build_human_eval.py --tally results/*.json [--tex paper/figures/human_eval.tex]

Listeners open index.html?L=<1..N> (or type their number), answer, press Download and send the
results_<L>.json. The tally reports means with 95% bootstrap CIs over listeners and items,
identification accuracy with Wilson CIs, sign tests, and inter-rater agreement (Krippendorff's alpha,
ordinal, for the 1-5 ratings; Fleiss' kappa for the type labels).
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research"
VOICES = ["male_hello_clean", "female_narration_clean", "male_news"]


def _sets():
    sys.path.insert(0, str(ROOT / "research"))
    sys.path.insert(0, str(ROOT / "eval"))
    from core import load_set
    cases = {}
    for s in ("prosody_set", "prosody_set2", "short_test"):
        cases.update({c["id"]: c for c in load_set(ROOT / f"research/sets/{s}.tsv")})
    return cases


def _even(cid: str) -> bool:
    return int("".join(ch for ch in cid if ch.isdigit()) or 0) % 2 == 0


def _strip_mark(t: str) -> str:
    return t.rstrip(" .!?؟…»\"").rstrip()


def stimuli() -> dict:
    """The fixed stimulus lists (ids; per-listener assignment happens in the page)."""
    cases = _sets()
    p1 = []
    for cid, c in sorted(cases.items()):
        if not _even(cid):
            continue
        if c["cat"] == "qyn" and cid.endswith("q") and cid[:-1] + "s" in cases:
            # a minimal pair: each listener hears ONE member (statement, or the question in one version)
            p1.append({"case": cid[:-1], "kind": "pair", "q": cid, "s": cid[:-1] + "s",
                       "text": _strip_mark(c["text"])})
        elif c["cat"] in ("exc", "qecho"):
            p1.append({"case": cid, "kind": {"exc": "exclamation", "qecho": "surprise"}[c["cat"]],
                       "text": _strip_mark(c["text"])})
    p2 = [{"case": cid, "kind": "text", "text": c["text"]} for cid, c in sorted(cases.items()) if c["cat"] == "dyn"]
    p2 += [{"case": cid, "kind": c["cat"], "text": c["text"]} for cid, c in sorted(cases.items())
           if _even(cid) and c["cat"] in ("exc", "qecho") and not cid.startswith("x0")][:8]
    shorts = json.loads((OUT.parent / "eval/flu_b3m_base_male_hello_clean/report.json").read_text())["rows"]
    seen = []
    for r in shorts:
        if r["lang"] == "fa" and r["case"] not in seen:
            seen.append(r["case"])
    p3 = [{"case": cid, "kind": "short", "text": cases[cid]["text"]} for cid in seen[:24] if cid in cases]
    return {"p1": p1, "p2": p2, "p3": p3}


def _src(part: str, system: str, case: str, voice: str) -> Path:
    if part == "p1" or (part == "p2" and not case.startswith("d")):
        return OUT / "dfinal" / ("prod" if system == "A0" else "new2") / f"{case}__{voice}__s1.wav"
    if part == "p2":
        return OUT / ("dyn1/prod" if system == "A0" else "dyn2/final") / f"{case}__{voice}__s1.wav"
    tag = "b3m_base_male_hello_clean" if system == "A0" else "b3m_ext_male_hello_clean"
    return OUT.parent / "eval" / f"flu_{tag}" / f"{case}__male_hello_clean__s1.wav"


def build(a) -> int:
    import soundfile as sf
    st = stimuli()
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    files = {}
    for part, items in st.items():
        for k, it in enumerate(items):
            voice = "male_hello_clean" if part == "p3" else VOICES[k % len(VOICES)]
            it["voice"] = voice
            if it["kind"] == "pair":                 # S = the statement (identical in both systems)
                src = _src("p1", "A0", it["s"], voice)
                if src.exists():
                    x, sr = sf.read(str(src), dtype="float32")
                    it["S"] = f"audio/p1_{it['s']}_{voice[:6]}_S.mp3"
                    sf.write(out / it["S"], x, sr, format="MP3")
            for system in ("A0", "A1"):              # A0 = shipped / plain clip, A1 = ours
                src = _src(part, system, it.get("q", it["case"]), voice)
                if not src.exists():
                    continue
                x, sr = sf.read(str(src), dtype="float32")
                name = f"audio/{part}_{it.get('q', it['case'])}_{voice[:6]}_{system}.mp3"
                sf.write(out / name, x, sr, format="MP3")
                files[(part, it["case"], system)] = name
            it["A0"] = files.get((part, it["case"], "A0"))
            it["A1"] = files.get((part, it["case"], "A1"))
    for part in st:
        st[part] = [it for it in st[part] if it.get("A0") and it.get("A1") and (it["kind"] != "pair" or it.get("S"))]
    (out / "stimuli.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    page = PAGE.replace("__STIM__", json.dumps(st, ensure_ascii=False)).replace(
        "__N1__", str(a.n1)).replace("__N2__", str(a.n2)).replace("__N3__", str(a.n3))
    (out / "index.html").write_text(page, encoding="utf-8")
    print({k: len(v) for k, v in st.items()}, "->", out / "index.html")
    return 0


PAGE = r"""<!doctype html><html lang="fa"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Persian speech listening test</title>
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --acc:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; --acc:#3987e5; } }
body { background:var(--bg); color:var(--ink); font:15px/1.6 system-ui, sans-serif; margin:0 auto; max-width:980px; padding:16px; }
h1 { font-size:22px; } h2 { font-size:18px; margin-top:32px; } p, li { color:var(--ink2); }
.item { border-bottom:1px solid var(--line); padding:12px 0; }
.fa { direction:rtl; font-size:18px; color:var(--ink); }
label { margin:0 10px 0 0; white-space:nowrap; } audio { width:100%; max-width:420px; height:34px; }
.scale label { margin-right:6px; } button { font-size:16px; padding:8px 16px; }
.done { color:var(--acc); }
</style></head><body>
<h1>Persian speech listening test / آزمون شنیداری گفتار فارسی</h1>
<p>About 15 minutes, with headphones in a quiet room. Listen to each clip at least once; replay as often as
you like. There are no right voices to prefer — answer by ear. / لطفاً با هدفون و در محیط آرام گوش دهید.</p>
<p>Listener number: <input id="L" size="4"> <button onclick="start()">Start</button> <span id="who"></span></p>
<div id="test" hidden>
<h2>Part 1 — What kind of sentence is it? How natural is it?</h2>
<p>The text is shown <b>without</b> its final punctuation. Decide from the voice whether it is a statement, a
question, or an exclamation / surprise, then rate how natural the voice sounds (1 = very unnatural,
5 = completely natural).</p><div id="p1"></div>
<h2>Part 2 — Compare A and B</h2>
<p>The same text, two versions. Which sounds more natural (−3 = A much better, 0 = equal, +3 = B much better)?
Which conveys the punctuation better — questions like questions, exclamations and surprise like them, a text
that is not monotone?</p><div id="p2"></div>
<h2>Part 3 — Short replies: is every word clear?</h2>
<p>Rate how clearly you understand every word (1 = not at all, 5 = every word clear) and tick if a word is
missing, wrong or garbled.</p><div id="p3"></div>
<p><button onclick="dl()">Download results</button> <span class="done" id="msg"></span></p>
</div>
<script>
const ST = __STIM__;
const N = {p1: __N1__, p2: __N2__, p3: __N3__};
let L = 0, PLAN = null;
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function shuffle(a, r) { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
function radios(name, opts) { return opts.map(([v, t]) => `<label><input type="radio" name="${name}" value="${v}">${t}</label>`).join(''); }
function start() {
  L = parseInt(document.getElementById('L').value || new URLSearchParams(location.search).get('L') || '0', 10);
  if (!L) { alert('Enter your listener number'); return; }
  document.getElementById('who').textContent = 'Listener ' + L;
  const r = rng(1000 + L);
  PLAN = {};
  for (const part of ['p1', 'p2', 'p3']) {
    // Latin square: the version alternates by item and listener; the item subset rotates with L
    const items = ST[part].map((it, k) => {
      if (it.kind === 'pair') {            // thirds: statement / question shipped / question ours
        const v = (k + L) % 3;
        return {...it, k, sys: ['S', 'A0', 'A1'][v], kind: v ? 'question' : 'statement'};
      }
      return {...it, k, sys: ((k + L) % 2) ? 'A1' : 'A0'}; });
    const n = Math.min(N[part], items.length), off = (L * n) % items.length;
    let pick = items.slice(off).concat(items.slice(0, off)).slice(0, n);
    PLAN[part] = shuffle(pick, r).map(it => ({...it, swap: r() < 0.5}));
  }
  document.getElementById('p1').innerHTML = PLAN.p1.map((it, i) => `<div class="item"><div class="fa">${it.text}</div>
    <audio controls preload="none" src="${it[it.sys]}"></audio><div>
    ${radios('t' + i, [['statement', 'statement / خبری'], ['question', 'question / پرسشی'], ['exclamation', 'exclamation-surprise / تعجبی']])}</div>
    <div class="scale">naturalness: ${radios('m' + i, [[1, '1'], [2, '2'], [3, '3'], [4, '4'], [5, '5']])}</div></div>`).join('');
  document.getElementById('p2').innerHTML = PLAN.p2.map((it, i) => {
    const A = it.swap ? it.A1 : it.A0, B = it.swap ? it.A0 : it.A1;
    return `<div class="item"><div class="fa">${it.text.replace(/\n+/g, ' ¶ ')}</div>
    <div>A <audio controls preload="none" src="${A}"></audio></div><div>B <audio controls preload="none" src="${B}"></audio></div>
    <div class="scale">more natural: ${radios('c' + i, [[-3, '−3'], [-2, '−2'], [-1, '−1'], [0, '0'], [1, '+1'], [2, '+2'], [3, '+3']])}</div>
    <div>conveys punctuation better: ${radios('x' + i, [['A', 'A'], ['same', 'same'], ['B', 'B']])}</div></div>`; }).join('');
  document.getElementById('p3').innerHTML = PLAN.p3.map((it, i) => `<div class="item"><div class="fa">${it.text}</div>
    <audio controls preload="none" src="${it[it.sys]}"></audio>
    <div class="scale">every word clear: ${radios('i' + i, [[1, '1'], [2, '2'], [3, '3'], [4, '4'], [5, '5']])}
    <label><input type="checkbox" name="e${i}"> word missing / wrong</label></div></div>`).join('');
  document.getElementById('test').hidden = false;
}
function val(n) { const q = document.querySelector(`input[name="${n}"]:checked`); return q ? q.value : null; }
function dl() {
  const res = {listener: L, when: new Date().toISOString(),
    p1: PLAN.p1.map((it, i) => ({case: it.case, kind: it.kind, voice: it.voice, sys: it.sys, label: val('t' + i), mos: +val('m' + i) || null})),
    p2: PLAN.p2.map((it, i) => { const c = val('c' + i), x = val('x' + i);
      // stored as "ours minus shipped": positive = ours better
      const cm = c === null ? null : (it.swap ? -(+c) : +c);
      const xp = x === null ? null : (x === 'same' ? 'same' : (((x === 'B') !== it.swap) ? 'ours' : 'shipped'));
      return {case: it.case, kind: it.kind, voice: it.voice, cmos: cm, expr: xp}; }),
    p3: PLAN.p3.map((it, i) => ({case: it.case, sys: it.sys, clear: +val('i' + i) || null,
      err: document.querySelector(`input[name="e${i}"]`).checked}))};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(res, null, 1)], {type: 'application/json'}));
  a.download = 'results_' + L + '.json'; a.click();
  document.getElementById('msg').textContent = 'Saved results_' + L + '.json — please send it back.';
}
</script></body></html>"""


# ── analysis ───────────────────────────────────────────────────────────────

def _boot_ci(vals_by_listener: dict, n: int = 4000) -> tuple[float, float, float]:
    keys = list(vals_by_listener)
    allv = [v for k in keys for v in vals_by_listener[k]]
    if not allv:
        return (float("nan"),) * 3
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        ks = rng.choice(len(keys), len(keys))
        v = [x for j in ks for x in vals_by_listener[keys[j]]]
        bs.append(np.mean(v))
    return float(np.mean(allv)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def kripp_alpha_ordinal(units: dict) -> float:
    """Krippendorff's alpha (ordinal) for units -> list of ratings (>= 2 ratings per unit used)."""
    vals = sorted({v for r in units.values() for v in r})
    if len(vals) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for r in units.values():
        m = len(r)
        if m < 2:
            continue
        for i, a in enumerate(r):
            for j, b in enumerate(r):
                if i != j:
                    o[idx[a], idx[b]] += 1 / (m - 1)
    nc = o.sum(axis=1)
    n = nc.sum()
    if n <= 1:
        return float("nan")
    def d2(c, k):
        lo, hi = min(c, k), max(c, k)
        s = nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2
        return s * s
    D = np.array([[d2(c, k) for k in range(len(vals))] for c in range(len(vals))])
    do = (o * D).sum() / n
    de = (np.outer(nc, nc) * D).sum() / (n * (n - 1))
    return float(1 - do / de) if de > 0 else float("nan")


def fleiss_kappa(units: dict, cats: list[str]) -> float:
    rows = [r for r in units.values() if len(r) >= 2]
    if not rows:
        return float("nan")
    P, pj = [], np.zeros(len(cats))
    for r in rows:
        n = len(r)
        cnt = np.array([r.count(c) for c in cats], dtype=float)
        pj += cnt
        P.append(((cnt * cnt).sum() - n) / (n * (n - 1)))
    pj /= pj.sum()
    Pe = (pj * pj).sum()
    return float((np.mean(P) - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def tally(paths: list[str], tex: str = "") -> int:
    from scipy.stats import binomtest
    res = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    nl = len(res)
    print(f"{nl} listener(s)")
    out = {}
    # part 1: identification + MOS
    for sysname, sysk in (("shipped", "A0"), ("ours", "A1")):
        for kind, ok in (("question", {"question"}), ("statement", {"statement"}),
                         ("exclamation", {"exclamation"}), ("surprise", {"question", "exclamation"})):
            rows = [r for x in res for r in x["p1"] if r["kind"] == kind and r["label"]
                    and (r["sys"] == sysk or kind == "statement")]      # statements: same audio in both
            k = sum(r["label"] in ok for r in rows)
            p, lo, hi = _wilson(k, len(rows))
            out[f"id_{kind}_{sysname}"] = (p, lo, hi, len(rows))
        by = {}
        for x in res:
            by[x["listener"]] = [r["mos"] for r in x["p1"] if r["sys"] == sysk and r["mos"]]
        out[f"mos_{sysname}"] = _boot_ci(by)
        by = {x["listener"]: [r["clear"] for r in x["p3"] if r["sys"] == sysk and r["clear"]] for x in res}
        out[f"clear_{sysname}"] = _boot_ci(by)
        e = [r["err"] for x in res for r in x["p3"] if r["sys"] == sysk]
        out[f"err_{sysname}"] = (np.mean(e) if e else float("nan"), len(e))
    # part 2: CMOS + expressiveness preference
    by = {x["listener"]: [r["cmos"] for r in x["p2"] if r["cmos"] is not None] for x in res}
    out["cmos"] = _boot_ci(by)
    for kind in ("text", "sentence"):
        by = {x["listener"]: [r["cmos"] for r in x["p2"] if r["cmos"] is not None
                              and ((r["kind"] == "text") == (kind == "text"))] for x in res}
        out[f"cmos_{kind}"] = _boot_ci(by)
    ex = [r["expr"] for x in res for r in x["p2"] if r["expr"]]
    w, l_, s = ex.count("ours"), ex.count("shipped"), ex.count("same")
    out["expr_pref"] = (w, s, l_, binomtest(w, w + l_).pvalue if w + l_ else float("nan"))
    # agreement
    units, labels = {}, {}
    for x in res:
        for r in x["p1"]:
            if r["mos"]:
                units.setdefault((r["case"], r["sys"]), []).append(r["mos"])
            if r["label"]:
                labels.setdefault((r["case"], r["sys"]), []).append(r["label"])
    out["alpha_mos"] = kripp_alpha_ordinal(units)
    out["kappa_type"] = fleiss_kappa(labels, ["statement", "question", "exclamation"])
    for k, v in out.items():
        print(f"  {k:28s} {v}")
    if tex:
        f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"  # noqa: E731
        pc = lambda t: f"{100 * t[0]:.0f}\\% [{100 * t[1]:.0f}, {100 * t[2]:.0f}]"  # noqa: E731
        lines = [f"% generated by research/build_human_eval.py --tally ({nl} listeners)",
                 f"\\gdef\\HEN{{{nl}}}%",
                 f"yes/no clip heard as a question & {pc(out['id_question_shipped'])} & {pc(out['id_question_ours'])} \\\\",
                 f"statement heard as a statement & {pc(out['id_statement_shipped'])} & {pc(out['id_statement_ours'])} \\\\",
                 f"exclamation heard as one & {pc(out['id_exclamation_shipped'])} & {pc(out['id_exclamation_ours'])} \\\\",
                 f"surprised q.\\ heard as q./excl. & {pc(out['id_surprise_shipped'])} & {pc(out['id_surprise_ours'])} \\\\",
                 f"naturalness MOS (1--5) & {f(out['mos_shipped'])} & {f(out['mos_ours'])} \\\\",
                 f"CMOS ours vs.\\ shipped, all & \\multicolumn{{2}}{{c}}{{{f(out['cmos'])}}} \\\\",
                 f"expressiveness: ours / same / shipped & \\multicolumn{{2}}{{c}}{{{out['expr_pref'][0]} / {out['expr_pref'][1]} / {out['expr_pref'][2]} ($p={out['expr_pref'][3]:.3f}$)}} \\\\",
                 f"short replies: clarity (1--5) & {f(out['clear_shipped'])} & {f(out['clear_ours'])} \\\\",
                 f"agreement: $\\alpha_\\text{{MOS}}$ / $\\kappa_\\text{{type}}$ & \\multicolumn{{2}}{{c}}{{{out['alpha_mos']:.2f} / {out['kappa_type']:.2f}}} \\\\"]
        Path(tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote", tex)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "human_eval"))
    ap.add_argument("--n1", type=int, default=24)
    ap.add_argument("--n2", type=int, default=12)
    ap.add_argument("--n3", type=int, default=12)
    ap.add_argument("--tally", nargs="*", default=None)
    ap.add_argument("--tex", default="")
    a = ap.parse_args()
    return tally(a.tally, a.tex) if a.tally else build(a)


if __name__ == "__main__":
    raise SystemExit(main())
