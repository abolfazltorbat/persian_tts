"""Paper figures for studies B and C (PythonProject5/.venv: matplotlib).

    .venv/bin/python persian_tts/research/fig_paper.py ext     -> paper/figures/fig_ext.pdf
    .venv/bin/python persian_tts/research/fig_paper.py into    -> paper/figures/fig_into.pdf
                                                                 (needs output/research/prosody/contours.json
                                                                  and output/research/prosody/table_test.json)

Colours: reference palette slots 1-3 (blue = before/baseline, orange = ours,
aqua = real speech), validated all-pairs (dataviz skill, palette.md); texts in
ink, never the series colour; every series also has a legend entry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BEFORE, OURS, REAL, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig_ext() -> None:
    """(a) failed renders before/after on held-out sets; (b) prompt layouts on dev."""
    groups = [("short set\n3 voices", 24, 12, 900), ("short set\ndefault\nvoice", 19, 8, 300),
              ("new clip\nends at\nsentence", 6, 4, 150), ("new clip\nends at\ncomma", 20, 11, 150),
              ("new clip\nends mid-\nclause", 70, 4, 150)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.15), dpi=300, gridspec_kw={"width_ratios": [1.25, 1]})
    w = 0.36
    for i, (lab, b, e, n) in enumerate(groups):
        for j, (v, c, name) in enumerate(((b, BEFORE, "clip only"), (e, OURS, "clip + self-extension"))):
            x = i + (j - 0.5) * (w + 0.04)
            r = 100 * v / n
            ax.bar(x, r, width=w, color=c, zorder=3, edgecolor=SURF, linewidth=0.6,
                   label=name if i == 0 else None, hatch="////" if j == 0 else None)
            ax.text(x, r + 0.8, f"{r:.1f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.0, linespacing=0.95)
    ax.set_ylabel("failed renders (%)")
    ax.set_ylim(0, 52)
    _grid(ax)
    ax.legend(loc="upper left", fontsize=6.6, handlelength=1.4)
    ax.set_title("(a) held-out sets, consensus judge", fontsize=7.5, loc="left", color=INK)
    # (b) prompt layouts, short dev units, male + female pooled (failed takes of 360)
    lay = [("clip (baseline)", 25), ("clip twice", 35), ("clip + own sentence (ours)", 10),
           ("clip + own sentence cut short", 120), ("own sentence + clip (clip last)", 348)]
    ys = np.arange(len(lay))[::-1]
    for y, (lab, v) in zip(ys, lay):
        r = 100 * v / 360
        c = OURS if "ours" in lab else MUTED
        bx.barh(y, r, height=0.58, color=c, zorder=3, edgecolor=SURF, linewidth=0.6)
        bx.text(r + 1.5, y, f"{r:.1f}%", va="center", fontsize=6.4, color=INK)
    bx.set_yticks(ys)
    bx.set_yticklabels([l for l, _ in lay], fontsize=6.6)
    bx.set_xlim(0, 112)
    bx.set_xlabel("failed takes (%)")
    _grid(bx, "x")
    bx.set_title("(b) prompt ending (dev, 360 takes each)", fontsize=7.5, loc="right", color=INK)
    fig.tight_layout(w_pad=1.2)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_ext.pdf")
    fig.savefig(FIG / "fig_ext.png")
    print("wrote", FIG / "fig_ext.pdf")


def fig_into() -> None:
    """Top: (a) one minimal pair before/after, (b) final pitch per sentence type.
    Bottom: (c-e) mean contours per type — real speech / before / after."""
    c = json.loads((ROOT / "output/research/prosody/contours.json").read_text())
    tab = json.loads((ROOT / "output/research/prosody/table_test.json").read_text())
    fig = plt.figure(figsize=(7.1, 3.75), dpi=300)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.62, wspace=0.95)
    # (a) minimal pair «جلسه فردا برگزار می‌شود.» / «...؟», one voice, one seed
    ax = fig.add_subplot(gs[0, 0:3])
    for key, col, lw, lab in (("base|s", BEFORE, 1.6, "before: statement and question (identical)"),
                              ("ours|q", OURS, 1.3, "after: question")):
        p = c["pair"].get(key)
        if not p:
            continue
        t = np.array(p["t"])
        f = np.array([np.nan if v is None else v for v in p["f0"]], dtype=float)
        ax.plot(t, f, color=col, lw=lw, label=lab, zorder=3 if key.startswith("ours") else 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("F0 (Hz)")
    _grid(ax)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.6, borderaxespad=0.2)
    ax.set_title("(a) one sentence pair, same words and seed", fontsize=7.5, loc="left", color=INK)
    # (b) final pitch per sentence type
    ex = fig.add_subplot(gs[0, 3:6])
    cats = [("st", "statement"), ("qyn", "yes/no question"), ("qwh", "wh-question"), ("exc", "exclamation")]
    w = 0.26
    for j, (name, col, lab) in enumerate((("real", REAL, "real speech"), ("base", BEFORE, "before"),
                                          ("ours", OURS, "after"))):
        for i, (cat, _l) in enumerate(cats):
            v = (tab["real"].get(cat, {}) if name == "real" else tab["sys"].get(name, {}).get(cat, {})).get("end_st")
            if v is None:
                continue
            x = i + (j - 1) * (w + 0.02)
            ex.bar(x, v, width=w, color=col, zorder=3, edgecolor=SURF, linewidth=0.5,
                   label=lab if i == 0 else None, hatch="////" if name == "base" else None)
            ex.text(x, v + (0.12 if v >= 0 else -0.12), f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=5.4, color=INK)
    ex.axhline(0, color=INK2, lw=0.6, zorder=4)
    ex.set_xticks(range(len(cats)))
    ex.set_xticklabels([l for _, l in cats], fontsize=6.4)
    ex.set_ylabel("final F0 re median (st)")
    ex.set_ylim(-5.4, 2.6)
    _grid(ex)
    ex.legend(loc="upper right", fontsize=6.2, ncol=3, handlelength=1.2, borderaxespad=0.1, columnspacing=1.0)
    ex.set_title("(b) how sentences end (held-out renders)", fontsize=7.5, loc="left", color=INK)
    # (c-e) contours
    xs = np.linspace(0, 1, c["n_points"])
    titles = {"qyn": "(c) yes/no questions", "qwh": "(d) wh-questions", "exc": "(e) exclamations"}
    for k, cat in enumerate(("qyn", "qwh", "exc")):
        bx = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        for name, col, lab in (("real", REAL, "real speech"), ("base", BEFORE, "before"), ("ours", OURS, "after")):
            key = f"{name}|{cat}"
            if key not in c["mean"]:
                continue
            m = np.array(c["mean"][key], dtype=float)
            lo = np.array(c["lo"][key], dtype=float)
            hi = np.array(c["hi"][key], dtype=float)
            if name != "real":
                bx.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
            bx.plot(xs, m, color=col, lw=1.3, zorder=3, ls="--" if name == "real" else "-",
                    label=f"{lab} (n={c['n'][key]})")
        bx.axhline(0, color=GRID, lw=0.6, zorder=1)
        bx.set_xlabel("normalised time")
        if k == 0:
            bx.set_ylabel("F0 re median (st)")
        bx.set_ylim(-6.5, 5)
        _grid(bx)
        bx.legend(loc="lower left", fontsize=5.8, handlelength=1.4, borderaxespad=0.2)
        bx.set_title(titles[cat], fontsize=7.5, loc="left", color=INK)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.1, top=0.94)
    fig.savefig(FIG / "fig_into.pdf")
    fig.savefig(FIG / "fig_into.png")
    print("wrote", FIG / "fig_into.pdf")


if __name__ == "__main__":
    {"ext": fig_ext, "into": fig_into}[sys.argv[1]]()
