"""Study D paper figure (PythonProject5/.venv: matplotlib) -> paper/figures/fig_expr.pdf

  (a) prompt styling: register shift of the voice prompt vs register shift of the take
      (output/research/style1|style3/measures.jsonl, per voice)
  (b) the recogniser as a listener: «؟» rate on prosodic yes/no questions — TTS without a
      rule, with study C's rule, with study D; real read speech (GPTInformal) and real
      expressive speech (ShEMO); statements as the floor
  (c) energy kept up to the end: second-half minus first-half energy, TTS vs real speech
  (d) one text, sentence by sentence: level and register, shipped engine vs study D

Inputs: output/research/dfinal/compare_test.json (research/dir_compare.py --json),
output/research/prosody/shemo.jsonl, output/research/punct/real.jsonl + the ShEMO
aligned.tsv, output/research/dyn1/sentences.jsonl.
Colours: palette slots 1-3 (blue = shipped engine / before, orange = study D, aqua = real
speech), validated all-pairs; grey + hatching = no rule. Texts in ink.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research"
FIG = ROOT / "paper" / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BEFORE, OURS, REAL, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
plt.rcParams.update({"font.family": "serif", "font.size": 7.2, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
VOICE_MARK = {"male_hello_clean": ("o", "male 1 (default)"), "female_narration_clean": ("s", "female"),
              "male_news": ("^", "male 2")}
STYLES = {"rt": (0.0, 1.0), "p2": (2.0, 1.0), "lv": (3.0, 1.4), "p4": (4.0, 1.0), "lv3": (4.0, 1.6),
          "lv2": (5.0, 1.5), "lo": (-2.0, 0.8)}


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def style_transfer() -> dict:
    """variant -> voice -> (mean output register shift st, mean output range st)"""
    rows = _jsonl(OUT / "style1/measures.jsonl") + _jsonl(OUT / "style3/measures.jsonl")
    ref = {(r["case"], r["voice"], r["seed"]): r for r in rows if r["variant"] == "ext"}
    acc = collections.defaultdict(list)
    for r in rows:
        k = (r["case"], r["voice"], r["seed"])
        if r["variant"] in STYLES and k in ref and "f0_med" in r and "f0_med" in ref[k]:
            acc[(r["variant"], r["voice"])].append((12 * np.log2(r["f0_med"] / ref[k]["f0_med"]),
                                                   r["range_st"] - ref[k]["range_st"]))
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def prosodic_yn(text: str) -> bool:
    return not re.search(r"آیا|مگه|مگر|یا نه|نه ?؟", text)


def real_qrates() -> dict:
    gi = {l.split("\t")[0]: l.split("\t")[3] for l in
          (Path.home() / ".cache/persian_tts_ft/data_gptinformal_types/clips.tsv").read_text(encoding="utf-8").splitlines()}
    out = {}
    rr = _jsonl(OUT / "punct/real.jsonl")
    yn = [r for r in rr if r["kind"] == "qyn" and prosodic_yn(gi[r["case"]])]
    st = [r for r in rr if r["kind"] == "st"]
    out["read q"] = (np.mean([r["mark"] == "?" for r in yn]), len(yn))
    out["read st"] = (np.mean([r["mark"] == "?" for r in st]), len(st))
    al = [l.split("\t") for l in (Path.home() / ".cache/persian_tts_ft/shemo/aligned.tsv").read_text(encoding="utf-8").splitlines()]
    q = [h.strip().endswith(("؟", "?")) for _c, _e, k, _s, t, h in al if k == "qyn" and prosodic_yn(t)]
    s = [h.strip().endswith(("؟", "?")) for _c, _e, k, _s, t, h in al if k == "st"]
    out["expr q"] = (np.mean(q), len(q))
    out["expr st"] = (np.mean(s), len(s))
    return out


def main() -> int:
    cmp_ = json.loads((OUT / "dfinal/compare_test.json").read_text(encoding="utf-8"))
    by_name = {Path(k).name: v for k, v in cmp_.items()}
    get = lambda d, cat, m: by_name[d].get(cat, {}).get(m, np.nan)  # noqa: E731
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.95), dpi=300, gridspec_kw={"width_ratios": [1, 1.15, 1.05, 1.3]})

    # (a) prompt register -> take register
    ax = axs[0]
    tr = style_transfer()
    for v, (mk, lab) in VOICE_MARK.items():
        pts = [(STYLES[s][0] + {"o": -0.12, "s": 0.0, "^": 0.12}[mk], tr[(s, v)][0]) for s in STYLES if (s, v) in tr]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=11, color=OURS if mk != "s" else SURF,
                   edgecolors=OURS, linewidths=0.8, label=lab, zorder=3)
    ax.plot([-2.5, 5.5], [-2.5, 5.5], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(4.1, 4.9, "1:1", fontsize=6, color=INK2)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.set_xlabel("prompt register shift (st)")
    ax.set_ylabel("take register shift (st)")
    ax.set_xlim(-2.6, 5.6)
    ax.set_ylim(-2.6, 5.6)
    _grid(ax, "both")
    ax.legend(fontsize=5.6, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    ax.set_title("(a) prompt → take", fontsize=7.2, loc="left", color=INK)

    # (b) «؟» heard on prosodic yes/no questions
    ax = axs[1]
    rq = real_qrates()
    bars = [("none", get("base", "qyn", "q"), MUTED, "////"),
            ("C", get("prod", "qyn", "q"), BEFORE, None),
            ("D", get("new2", "qyn", "q"), OURS, None),
            ("read", rq["read q"][0], REAL, "////"),
            ("expr.", rq["expr q"][0], REAL, None)]
    for i, (lab, v, c, h) in enumerate(bars):
        ax.bar(i, 100 * v, width=0.62, color=c, hatch=h, edgecolor=SURF, linewidth=0.6, zorder=3)
        ax.text(i, 100 * v + 1.2, f"{100 * v:.0f}", ha="center", fontsize=6, color=INK)
    st_floor = 100 * get("base", "st", "q")
    ax.axhline(st_floor, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=4)
    ax.text(-0.45, 47, "dashed: TTS statements", fontsize=5.4, color=INK2, ha="left", va="top")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6)
    ax.text(1.0, -9.5, "TTS rule", ha="center", fontsize=6, color=INK2)
    ax.text(3.5, -9.5, "real speech", ha="center", fontsize=6, color=INK2)
    ax.set_ylabel("Whisper ends with «?» (%)")
    ax.set_ylim(0, 50)
    _grid(ax)
    ax.set_title("(b) heard as a question", fontsize=7.2, loc="left", color=INK)
    ax.set_xlim(-0.6, 4.6)

    # (c) energy decay, second half - first half
    ax = axs[2]
    sh = _jsonl(OUT / "prosody/shemo.jsonl")
    shemo = {e: np.nanmean([r["half_db"] for r in sh if r["emo"] == e and "half_db" in r]) for e in ("N", "W", "H")}
    groups = [("stmt", [("no rule", get("base", "st", "half_db"), MUTED, "////")]),
              ("excl.", [("no rule", get("base", "exc", "half_db"), MUTED, "////"),
                         ("study C", get("prod", "exc", "half_db"), BEFORE, None),
                         ("study D", get("new2", "exc", "half_db"), OURS, None)]),
              ("real", [("neutral", shemo["N"], REAL, "////"), ("surprise", shemo["W"], REAL, None),
                        ("happy", shemo["H"], REAL, "...")])]
    x = 0
    ticks, tlabs = [], []
    for g, items in groups:
        xs = []
        for lab, v, c, h in items:
            ax.bar(x, v, width=0.8, color=c, hatch=h, edgecolor=SURF, linewidth=0.6, zorder=3)
            ax.text(x, v - 0.35, f"{v:.1f}", ha="center", va="top", fontsize=5.4, color=INK)
            ax.text(x, 0.2, lab, ha="center", va="bottom", fontsize=4.6, color=INK2, rotation=90)
            xs.append(x)
            x += 1
        ticks.append(np.mean(xs))
        tlabs.append(g)
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=6)
    ax.set_ylim(-8.5, 3.4)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("2nd − 1st half energy (dB)")
    _grid(ax)
    ax.set_title("(c) energy to the end", fontsize=7.2, loc="left", color=INK)

    # (d) one text sentence by sentence
    ax = axs[3]
    sents = _jsonl(OUT / "dyn2/sentences.jsonl")
    case, voice, seed = "d03", "male_hello_clean", 1
    for cfg, c, mk, lab in (("prod", BEFORE, "o", "shipped (study C)"), ("final", OURS, "s", "study D")):
        rs = sorted([r for r in sents if r["config"] == cfg and r["case"] == case and r["voice"] == voice
                     and r["seed"] == seed], key=lambda r: r["i"])
        lv = np.array([r["level_db"] for r in rs])
        ax.plot(np.arange(len(rs)), lv - lv.mean(), marker=mk, ms=3.4, lw=0.9, color=c, label=lab, zorder=3)
    kinds = [r["kind"] for r in sorted([r for r in sents if r["config"] == "final" and r["case"] == case
                                        and r["voice"] == voice and r["seed"] == seed], key=lambda r: r["i"])]
    short = {"st": "stmt", "qyn": "y/n?", "qecho": "echo?!", "qwh": "wh?", "exc": "excl!", "qalt": "alt?"}
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([short.get(k, k) for k in kinds], fontsize=5.4, rotation=35)
    ax.set_ylabel("sentence level (dB, re text mean)")
    _grid(ax)
    ax.legend(fontsize=5.6, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    ax.set_ylim(-2.6, 3.2)
    ax.set_title("(d) one text, by sentence", fontsize=7.2, loc="left", color=INK)

    fig.tight_layout(pad=0.3, w_pad=0.9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_expr.pdf")
    fig.savefig(FIG / "fig_expr.png", dpi=200)
    print("wrote", FIG / "fig_expr.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
