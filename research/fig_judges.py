"""Fig. 1: single-take failure rate by unit length, Whisper alone vs the
two-recogniser consensus (dev corpus, Persian takes). -> paper/figures/fig_judges.pdf/.png"""
import json
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / "output/research/corpus_dev/rows_judged.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
bins = ["1–3", "4–6", "7–10", "11+"]


def b(t):
    return "1–3" if t <= 3 else "4–6" if t <= 6 else "7–10" if t <= 10 else "11+"


g = collections.defaultdict(list)
for r in rows:
    g[b(r["tokens"])].append(r)
rate = {lab: [100 * sum(x[lab] > 0.3 for x in g[k]) / len(g[k]) for k in bins] for lab in ("cer_whisper", "cer_cons")}
n = [len(g[k]) for k in bins]

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C = {"cer_whisper": "#eb6834", "cer_cons": "#2a78d6"}
NAME = {"cer_whisper": "Whisper alone", "cer_cons": "consensus (Whisper ∧ wav2vec2)"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(3.45, 2.05), dpi=300)
w = 0.36
for i, lab in enumerate(("cer_whisper", "cer_cons")):
    xs = [j + (i - 0.5) * (w + 0.03) for j in range(len(bins))]
    bars = ax.bar(xs, rate[lab], width=w, color=C[lab], label=NAME[lab], zorder=3,
                  hatch="////" if lab == "cer_whisper" else None, edgecolor=SURF, linewidth=0.6)
    for x, v in zip(xs, rate[lab]):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_xticks(range(len(bins)))
ax.set_xticklabels([f"{k}\n(n={m})" for k, m in zip(bins, n)])
ax.set_xlabel("unit length (phoneme tokens)")
ax.set_ylabel("takes failing (%)")
ax.set_ylim(0, max(rate["cer_whisper"]) * 1.18)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=7)
fig.tight_layout(pad=0.3)
out = ROOT / "paper" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_judges.pdf")
fig.savefig(out / "fig_judges.png")
print("ok", rate, n)
