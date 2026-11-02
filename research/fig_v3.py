"""Figures of the rewritten paper (PythonProject5/.venv: matplotlib).

    .venv/bin/python persian_tts/research/fig_v3.py robust   -> paper/figures/fig_robust.pdf
    .venv/bin/python persian_tts/research/fig_v3.py expr     -> paper/figures/fig_expr.pdf

Each figure states one idea and shows the evidence for it next to a sketch of the
mechanism. Colours: palette slots 1-3 (blue = the engine before our changes /
the shipped rule, orange = ours, aqua = real speech), validated all-pairs; grey +
hatching = "no rule / baseline". Texts in ink, never in a series colour.
Inputs: output/research/{dfinal,dyn1,dyn2,style1,style3,punct,prosody} (see research/PROGRESS.md).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/research"
FIG = ROOT / "paper" / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BEFORE, OURS, REAL, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
CLIP, TARGET, BAD = "#d9d8d3", "#cfe1f7", "#f6d4d2"
plt.rcParams.update({"font.family": "serif", "font.size": 7.0, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "axes.titlesize": 7.2, "axes.titlelocation": "left"})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _block(ax, x, y, w, h, fc, text, ec=INK2, abrupt=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.5, zorder=3))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=5.7, color=tc, zorder=4)
    if abrupt:   # a cut-off ending: a zig-zag on the right edge
        zy = np.linspace(y - h / 2, y + h / 2, 7)
        zx = x + w + np.array([0, 0.07, 0, 0.07, 0, 0.07, 0])
        ax.plot(zx, zy, color="#c0392b", lw=0.8, zorder=5)


# ── Fig. 2: how the prompt ends ────────────────────────────────────────────
def fig_robust() -> None:
    fig = plt.figure(figsize=(7.16, 2.05), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    # rows: label, blocks [(width, colour, text, abrupt)], failed takes of 360 (dev, male+female)
    OWN = "#fbd9c7"
    rows = [("training", [(2.15, CLIP, "utterance up to a cut", False), (2.0, TARGET, "rest of it (text given)", False)], None),
            ("plain clip", [(0.8, CLIP, "clip", True), (2.0, TARGET, "take", False)], 25),
            ("clip twice", [(0.8, CLIP, "clip", True), (0.8, CLIP, "clip", True), (2.0, TARGET, "take", False)], 35),
            ("ours", [(0.8, CLIP, "clip", False), (1.45, OWN, "own sentence.", False),
                      (2.0, TARGET, "take", False)], 10),
            ("cut short", [(0.8, CLIP, "clip", False), (0.95, OWN, "own sent", True),
                           (2.0, TARGET, "take", False)], 120),
            ("clip last", [(1.45, OWN, "own sentence.", False), (0.8, CLIP, "clip", True),
                           (2.0, BAD, "reads the sentence", False)], 348)]
    ys = np.arange(len(rows))[::-1] * 1.0
    bar0 = 5.0
    for y, (lab, blocks, fails) in zip(ys, rows):
        ax.text(-0.1, y, lab, ha="right", va="center", fontsize=6.2,
                color=INK, fontweight="bold" if lab == "ours" else "normal")
        x = 0.0
        for w, fc, txt, ab in blocks:
            _block(ax, x, y, w, 0.62, fc, txt, abrupt=ab)
            x += w + (0.1 if ab else 0.03)
        if fails is not None:
            r = 100 * fails / 360
            L = 2.2 * min(r, 100) / 100
            ax.add_patch(FancyBboxPatch((bar0, y - 0.22), max(L, 0.02), 0.44, boxstyle="square,pad=0",
                                        fc=OURS if lab == "ours" else MUTED, ec="none", zorder=3))
            ax.text(bar0 + L + 0.06, y, f"{r:.1f}%", va="center", fontsize=6.2, color=INK)
    ax.plot([bar0, bar0], [ys[-1] - 0.45, ys[1] + 0.45], color=INK2, lw=0.6)
    ax.text(bar0 + 1.1, ys[0], "failed short takes\n(dev, 360 per row)", ha="center", va="center", fontsize=6.0,
            color=INK2)
    ax.set_xlim(-1.25, 7.9)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.axis("off")
    ax.set_title("(a) the prompt must end at a finished sentence", loc="left", color=INK)
    # (b) held-out: new clips by how they end + the default voice
    bx = fig.add_subplot(gs[0, 1])
    groups = [("ends a\nsentence", 6, 4, 150), ("ends at\na comma", 20, 11, 150), ("ends mid-\nclause", 70, 4, 150),
              ("default\nvoice", 19, 8, 300)]
    w = 0.36
    for i, (lab, b, e, n) in enumerate(groups):
        for j, (v, c, name, h) in enumerate(((b, MUTED, "clip", "////"), (e, OURS, "clip + own sentence", None))):
            x = i + (j - 0.5) * (w + 0.04)
            r = 100 * v / n
            bx.bar(x, r, width=w, color=c, hatch=h, edgecolor=SURF, linewidth=0.6, zorder=3,
                   label=name if i == 0 else None)
            bx.text(x, r + 0.9, f"{r:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK)
    bx.set_xticks(range(len(groups)))
    bx.set_xticklabels([g[0] for g in groups], fontsize=5.8, linespacing=0.9)
    bx.axvline(2.5, color=GRID, lw=0.8)
    bx.text(-0.4, 52, "new real clips, cut 0.02 s after speech", ha="left", fontsize=5.8, color=INK2)
    bx.set_ylabel("failed renders (%)")
    bx.set_ylim(0, 55)
    _grid(bx)
    bx.legend(loc="upper left", fontsize=5.8, handlelength=1.4, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.88))
    bx.set_title("(b) held-out: worse ending, larger gain", color=INK)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.17, top=0.9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_robust.pdf")
    fig.savefig(FIG / "fig_robust.png")
    print("wrote", FIG / "fig_robust.pdf")


# ── Fig. 3: expressiveness ─────────────────────────────────────────────────
STYLES = {"rt": (0.0, 1.0), "p2": (2.0, 1.0), "lv": (3.0, 1.4), "p4": (4.0, 1.0), "lv3": (4.0, 1.6),
          "lv2": (5.0, 1.5), "lo": (-2.0, 0.8)}
VOICE_MARK = {"male_hello_clean": ("o", "male 1 (default)"), "female_narration_clean": ("s", "female"),
              "male_news": ("^", "male 2")}


def _contour(ax, d, key, col, lab, lw=1.2, ls="-"):
    t = np.array(d[key]["t"])
    f = np.array([np.nan if v is None else v for v in d[key]["f0"]], dtype=float)
    ax.plot(t, f, color=col, lw=lw, ls=ls, label=lab, zorder=3)


def fig_expr() -> None:
    cmp_ = json.loads((OUT / "dfinal/compare_test.json").read_text(encoding="utf-8"))
    by = {Path(k).name: v for k, v in cmp_.items()}
    get = lambda d, cat, m: by[d].get(cat, {}).get(m, np.nan)  # noqa: E731
    con = json.loads((OUT / "dfinal/contours_fig.json").read_text())
    fig = plt.figure(figsize=(7.16, 3.35), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.34)

    # (a) a question is a statement for this model
    ax = fig.add_subplot(gs[0, 0])
    _contour(ax, con, "st", MUTED, "statement = question, no rule", lw=2.2)
    _contour(ax, con, "q_new", OURS, "question, ours")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("F0 (Hz)")
    _grid(ax)
    ax.legend(loc="lower left", fontsize=5.6, handlelength=1.5, borderaxespad=0.1)
    ax.set_ylim(75, 200)
    ax.set_title("(a) a question rises at the end", color=INK)

    # (b) exclamation: register and range from the prompt
    bx = fig.add_subplot(gs[0, 1])
    _contour(bx, con, "x_base", MUTED, "exclamation, no rule", lw=2.2)
    _contour(bx, con, "x_new", OURS, "exclamation, ours")
    bx.set_xlabel("time (s)")
    bx.set_ylim(70, 260)
    _grid(bx)
    bx.legend(loc="upper right", fontsize=5.6, handlelength=1.5, borderaxespad=0.1)
    bx.set_title("(b) an exclamation is livelier", color=INK)

    # (c) loudness through one text
    cx = fig.add_subplot(gs[0, 2])
    sents = _jsonl(OUT / "dyn2/sentences.jsonl")
    case, voice, seed = "d03", "male_hello_clean", 1
    for cfg, c, mk, lab in (("prod", BEFORE, "o", "shipped engine"), ("final", OURS, "s", "ours")):
        rs = sorted([r for r in sents if r["config"] == cfg and r["case"] == case and r["voice"] == voice
                     and r["seed"] == seed], key=lambda r: r["i"])
        lv = np.array([r["level_db"] for r in rs])
        cx.plot(np.arange(len(rs)), lv - lv.mean(), marker=mk, ms=3.2, lw=0.9, color=c, label=lab, zorder=3)
    kinds = [r["kind"] for r in sorted([r for r in sents if r["config"] == "final" and r["case"] == case
                                        and r["voice"] == voice and r["seed"] == seed], key=lambda r: r["i"])]
    short = {"st": "stmt", "qyn": "y/n?", "qecho": "echo?!", "qwh": "wh?", "exc": "excl!", "qalt": "alt?"}
    cx.set_xticks(range(len(kinds)))
    cx.set_xticklabels([short.get(k, k) for k in kinds], fontsize=5.4, rotation=30)
    cx.set_ylabel("level re text mean (dB)")
    cx.set_ylim(-2.7, 3.2)
    _grid(cx)
    cx.legend(loc="upper left", fontsize=5.6, handlelength=1.5, borderaxespad=0.1)
    cx.set_title("(c) loudness through one text", color=INK)

    # (d) the prompt carries register
    dx = fig.add_subplot(gs[1, 0])
    rows = _jsonl(OUT / "style1/measures.jsonl") + _jsonl(OUT / "style3/measures.jsonl")
    ref = {(r["case"], r["voice"], r["seed"]): r for r in rows if r["variant"] == "ext"}
    for v, (mk, lab) in VOICE_MARK.items():
        pts = []
        for s, (reg, _k) in STYLES.items():
            dd = [12 * np.log2(r["f0_med"] / ref[(r["case"], v, r["seed"])]["f0_med"]) for r in rows
                  if r["variant"] == s and r["voice"] == v and "f0_med" in r
                  and "f0_med" in ref.get((r["case"], v, r["seed"]), {})]
            if dd:
                pts.append((reg + {"o": -0.12, "s": 0.0, "^": 0.12}[mk], np.mean(dd)))
        dx.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=12, color=OURS if mk != "s" else SURF,
                   edgecolors=OURS, linewidths=0.8, label=lab, zorder=3)
    dx.plot([-2.5, 5.5], [-2.5, 5.5], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    dx.text(3.9, 4.8, "1:1", fontsize=5.8, color=INK2)
    dx.set_xlabel("register shift of the prompt (st)")
    dx.set_ylabel("register shift of the take (st)")
    dx.set_xlim(-2.6, 5.6)
    dx.set_ylim(-2.6, 5.6)
    _grid(dx, "both")
    dx.legend(loc="upper left", fontsize=5.6, handlelength=1.2, borderaxespad=0.1)
    dx.set_title("(d) the take follows the prompt", color=INK)

    # (e) heard as a question
    ex = fig.add_subplot(gs[1, 1])
    gi = {l.split("\t")[0]: l.split("\t")[3] for l in
          (Path.home() / ".cache/persian_tts_ft/data_gptinformal_types/clips.tsv").read_text(encoding="utf-8").splitlines()}
    pros = lambda t: not re.search(r"آیا|مگه|مگر|یا نه|نه ?؟", t)  # noqa: E731
    rr = _jsonl(OUT / "punct/real.jsonl")
    read_q = np.mean([r["mark"] == "?" for r in rr if r["kind"] == "qyn" and pros(gi[r["case"]])])
    al = [l.split("\t") for l in (Path.home() / ".cache/persian_tts_ft/shemo/aligned.tsv").read_text(encoding="utf-8").splitlines()]
    expr_q = np.mean([h.strip().endswith(("؟", "?")) for _c, _e, k, _s, t, h in al if k == "qyn" and pros(t)])
    bars = [("no\nrule", get("base", "qyn", "q"), MUTED, "////"), ("shipped\nrule", get("prod", "qyn", "q"), BEFORE, None),
            ("ours", get("new2", "qyn", "q"), OURS, None), ("real,\nread", read_q, REAL, "////"),
            ("real,\nexpressive", expr_q, REAL, None)]
    for i, (lab, v, c, h) in enumerate(bars):
        ex.bar(i, 100 * v, width=0.64, color=c, hatch=h, edgecolor=SURF, linewidth=0.6, zorder=3)
        ex.text(i, 100 * v + 1.0, f"{100 * v:.0f}", ha="center", fontsize=5.8, color=INK)
    ex.axhline(100 * get("base", "st", "q"), color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=4)
    ex.text(-0.4, 46, "dashed: our statements", fontsize=5.3, color=INK2, ha="left", va="top")
    ex.set_xticks(range(len(bars)))
    ex.set_xticklabels([b[0] for b in bars], fontsize=5.4, linespacing=0.9)
    ex.set_ylabel("recogniser writes «?» (%)")
    ex.set_ylim(0, 48)
    _grid(ex)
    ex.set_title("(e) yes/no heard as a question", color=INK)

    # (f) energy kept to the end
    fx = fig.add_subplot(gs[1, 2])
    sh = _jsonl(OUT / "prosody/shemo.jsonl")
    shemo = {e: np.nanmean([r["half_db"] for r in sh if r["emo"] == e and "half_db" in r]) for e in ("N", "W", "H")}
    items = [("no rule", get("base", "exc", "half_db"), MUTED, "////"),
             ("shipped", get("prod", "exc", "half_db"), BEFORE, None),
             ("ours", get("new2", "exc", "half_db"), OURS, None),
             ("neutral", shemo["N"], REAL, "////"), ("surprised", shemo["W"], REAL, None), ("happy", shemo["H"], REAL, "....")]
    for i, (lab, v, c, h) in enumerate(items):
        x = i + (0.4 if i >= 3 else 0)
        fx.bar(x, v, width=0.7, color=c, hatch=h, edgecolor=SURF, linewidth=0.6, zorder=3)
        fx.text(x, v - 0.3, f"{v:.1f}", ha="center", va="top", fontsize=5.6, color=INK)
    fx.set_xticks([i + (0.4 if i >= 3 else 0) for i in range(len(items))])
    fx.set_xticklabels([it[0] for it in items], fontsize=5.4, rotation=30)
    fx.text(1.0, 1.2, "TTS exclamations", ha="center", fontsize=5.8, color=INK2)
    fx.text(4.4, 1.2, "real speech", ha="center", fontsize=5.8, color=INK2)
    fx.axhline(0, color=INK2, lw=0.5)
    fx.set_ylim(-8.8, 2.4)
    fx.set_ylabel("2nd − 1st half energy (dB)")
    _grid(fx)
    fx.set_title("(f) energy kept to the end", color=INK)

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.12, top=0.94)
    fig.savefig(FIG / "fig_expr.pdf")
    fig.savefig(FIG / "fig_expr.png")
    print("wrote", FIG / "fig_expr.pdf")


if __name__ == "__main__":
    {"robust": fig_robust, "expr": fig_expr}[sys.argv[1]]()
