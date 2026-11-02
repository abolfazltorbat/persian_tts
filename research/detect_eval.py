"""Study A, step 2: how well do ASR-free signals catch bad takes, and what
does choosing among takes with them buy?

Two questions, both answered against the Whisper judge (consonant-skeleton
CER; a take "fails" above --fail, default 0.30):

A1  per-take detection: ROC-AUC of every signal, the engine's current QC rule
    as a detector, and a logistic combination (trained on dev, frozen, scored
    on test).
A2  selection among N takes of a unit (what a retry loop does):
      first    take k0                      (no retry)
      engine   today's rule: first QC-passing take of k0..k2, else the best QC key
      detector first take the A1 detector accepts, else its lowest-risk take
      medoid   the take with the smallest mean DTW distance to the others
               (latent consensus: correct readings agree, mistakes are
               idiosyncratic)
      oracle   the lowest-CER take (upper bound)
    and the cost of each in generated takes.

    python research/detect_eval.py --dev corpus_dev --test corpus_test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parents[1] / "output" / "research"


LABEL = "whisper"   # whisper | w2v | cons (both judges must fail a take; Persian rows only)


def load(name: str) -> list[dict]:
    d = OUT / name
    src = "rows.jsonl" if LABEL == "whisper" else "rows_judged.jsonl"
    rows = [json.loads(x) for x in (d / src).read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        r["_lat"] = d / "lat" / (r["rid"].replace("|", "__") + ".npy")
        if LABEL != "whisper":
            r["cer"] = r["cer_" + LABEL]
            r["hyp"] = r["hyp_w2v"] if LABEL == "w2v" else r["hyp"]
    return rows


# ── features ──────────────────────────────────────────────────────────────
def feats(r: dict) -> dict:
    nv = max(1, r["vowels"])
    exp = max(1e-3, r["expected_s"])
    return {
        "loop": r["loop"],
        "leak": r["leak"],
        "syl_dev": abs(math.log((r["syl"] + 0.5) / (nv + 0.5))),
        "rate_dev": abs(math.log(max(1e-3, r["s_per_vowel"]) / 0.17)),
        "dur_ratio_dev": abs(math.log(max(1e-3, r["solid_s"]) / (exp * 0.55))),
        "solid_frac_low": 1.0 - r["solid_frac"],
        "dead_s": r["dead_s"],
        "cap_hit": float(r["cap_hit"]),
        "eos_per_vowel_dev": abs(math.log(((r["eos_step"] if r["eos_step"] is not None else r["max_gen"]) + 1) / nv / 2.2)),
        "short": 1.0 / max(1, r["tokens"]),
    }


FEATS = list(feats({"vowels": 1, "expected_s": 1, "loop": 0, "leak": 0, "syl": 1, "s_per_vowel": 0.17,
                    "solid_s": 0.55, "solid_frac": 1, "dead_s": 0, "cap_hit": False, "eos_step": 1,
                    "max_gen": 1, "tokens": 1}).keys())


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(scores)
    rank = {}
    i = 0
    while i < len(order):          # average ranks for ties
        j = i
        while j + 1 < len(order) and order[j + 1] == order[i]:
            j += 1
        rank[order[i]] = (i + j) / 2 + 1
        i = j + 1
    rs = sum(rank[s] for s in pos)
    return (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class Logit:
    """Plain L2 logistic regression (numpy), standardized inputs."""

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 3000, lr: float = 0.1):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1])
        b = math.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
        n = len(y)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = Z.T @ (p - y) / n + l2 * w / n
            w -= lr * g
            b -= lr * float((p - y).mean())
        self.w, self.b = w, b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))


def matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[feats(r)[k] for k in FEATS] for r in rows], dtype=np.float64)


def prf(pred: list[bool], lab: list[int]) -> dict:
    tp = sum(p and y for p, y in zip(pred, lab))
    fp = sum(p and not y for p, y in zip(pred, lab))
    fn = sum((not p) and y for p, y in zip(pred, lab))
    tn = len(lab) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "reject_rate": round((tp + fp) / len(lab), 3), "false_reject": round(fp / max(1, fp + tn), 3)}


# ── consensus (medoid) selection ─────────────────────────────────────────
def speech_z(path: Path) -> np.ndarray:
    """Normalized latents of a take, leading/trailing quiet removed by a
    latent-energy proxy (distance from the take's quietest frame)."""
    z = np.load(path).astype(np.float32)
    if len(z) < 4:
        return z
    q = z[np.argmin(np.linalg.norm(z, axis=1))]      # a silence-like frame
    d = np.linalg.norm(z - q, axis=1)
    on = np.where(d > 0.5 * np.median(d) + 1e-6)[0]
    return z[on[0]: on[-1] + 1] if len(on) else z


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    A = a / np.maximum(1e-6, np.linalg.norm(a, axis=1, keepdims=True))
    B = b / np.maximum(1e-6, np.linalg.norm(b, axis=1, keepdims=True))
    C = 1.0 - A @ B.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = C[i - 1]
        for j in range(1, m + 1):
            D[i, j] = ci[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def select_report(rows: list[dict], det: Logit | None, thr: float, fail: float, ns=(2, 3, 4)) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case"], r["unit"], r["voice"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["take"])
    risk = {}
    if det is not None:
        P = det.predict(matrix(rows))
        risk = {r["rid"]: float(p) for r, p in zip(rows, P)}
    cache: dict = {}

    def z(r):
        if r["rid"] not in cache:
            cache[r["rid"]] = speech_z(r["_lat"])
        return cache[r["rid"]]

    res = defaultdict(list)
    cost = defaultdict(list)
    for key, g in groups.items():
        if len(g) < max(ns):
            continue
        res["first"].append(g[0]["cer"])
        cost["first"].append(1)
        # engine: first QC-pass among 3, else best QC key (loud, dur, solid_s, dead)
        pick, used = None, 0
        for r in g[:3]:
            used += 1
            if r["qc_ok"]:
                pick = r
                break
        if pick is None:
            pick = max(g[:3], key=lambda r: (r["loud_ok"], r["dur_ok"], r["solid_s"],
                                               0.0 if r["dead_s"] <= 0.35 else -r["dead_s"]))
        res["engine"].append(pick["cer"])
        cost["engine"].append(used)
        if det is not None:
            pick, used = None, 0
            for r in g[:3]:
                used += 1
                if risk[r["rid"]] < thr:
                    pick = r
                    break
            if pick is None:
                pick = min(g[:3], key=lambda r: risk[r["rid"]])
            res["detector"].append(pick["cer"])
            cost["detector"].append(used)
        for n in ns:
            sub = g[:n]
            D = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    D[i, j] = D[j, i] = dtw(z(sub[i]), z(sub[j]))
            if n == 2:
                # two takes: no majority; keep the one the detector (or QC) trusts more
                if det is not None:
                    m = min(range(2), key=lambda i: risk[sub[i]["rid"]])
                else:
                    m = 0 if sub[0]["qc_ok"] or not sub[1]["qc_ok"] else 1
            else:
                m = int(np.argmin(D.sum(1)))
            res[f"medoid{n}"].append(sub[m]["cer"])
            cost[f"medoid{n}"].append(n)
            if det is not None and n >= 3:
                # medoid among the takes the detector accepts (all if none)
                ok = [i for i in range(n) if risk[sub[i]["rid"]] < thr] or list(range(n))
                mm = min(ok, key=lambda i: D[i, ok].sum())
                res[f"det+medoid{n}"].append(sub[mm]["cer"])
                cost[f"det+medoid{n}"].append(n)
        res["oracle6"].append(min(r["cer"] for r in g))
        cost["oracle6"].append(len(g))
    out = {}
    for k, v in res.items():
        out[k] = {"units": len(v), "cer": round(st.mean(v), 4), "fail": round(sum(x > fail for x in v) / len(v), 4),
                  "takes": round(st.mean(cost[k]), 2)}
    return out


def edit_ops(ref: str, hyp: str) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the skeleton alignment."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def taxonomy(rows: list[dict], fail: float) -> dict:
    """Failing takes by dominant error type and unit length (skeleton ops)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import harness  # noqa: F401  (paths + the typer shim normalize_fa needs)
    from verify import phoneme_skeleton, text_skeleton

    def lbin(t):
        return "<=6" if t <= 6 else "7-10" if t <= 10 else "11+"
    out: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["cer"] <= fail:
            continue
        if not r["hyp"].strip():
            kind = "silent"
        else:
            s, d, ins = edit_ops(phoneme_skeleton(r["chunk"]), text_skeleton(r["hyp"], r["lang"]))
            tot = max(1, s + d + ins)
            kind = "deletion" if d / tot >= 0.5 else "insertion" if ins / tot >= 0.5 else "substitution"
        out[lbin(r["tokens"])][kind] += 1
        out["all"][kind] += 1
    return {k: dict(v) for k, v in out.items()}


def within_unit_auc(rows: list[dict], fail: float) -> dict:
    """Take-level discrimination with the unit held fixed: over every pair
    (bad take, good take) of the SAME unit and voice, how often does a signal
    rank the bad take as worse? 0.5 = the signal only knows which units are
    hard, not which take failed. (Pairwise AUC pooled over units.)"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case"], r["unit"], r["voice"])].append(r)
    wins = defaultdict(float)
    pairs = 0
    for g in groups.values():
        bad = [r for r in g if r["cer"] > fail]
        good = [r for r in g if r["cer"] <= fail]
        for b in bad:
            fb = feats(b)
            for o in good:
                fo = feats(o)
                pairs += 1
                for k in FEATS:
                    wins[k] += 1.0 if fb[k] > fo[k] else 0.5 if fb[k] == fo[k] else 0.0
                wins["engine_qc"] += 1.0 if (not b["qc_ok"]) > (not o["qc_ok"]) else 0.5 if b["qc_ok"] == o["qc_ok"] else 0.0
    out = {k: round(v / pairs, 3) for k, v in wins.items()} if pairs else {}
    out["pairs"] = pairs
    return out


def roc_at(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    """Recall of a score at a given false-reject rate."""
    neg = np.sort(scores[y == 0])
    if not len(neg) or not (y == 1).any():
        return float("nan")
    thr = neg[min(len(neg) - 1, int((1 - fpr) * len(neg)))]
    return float((scores[y == 1] > thr).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="corpus_dev")
    ap.add_argument("--test", default="")
    ap.add_argument("--fail", type=float, default=0.30)
    ap.add_argument("--max-fpr", type=float, default=0.15, help="detector threshold: false-reject budget on dev")
    ap.add_argument("--label", default="whisper", choices=["whisper", "w2v", "cons"])
    a = ap.parse_args()
    global LABEL
    LABEL = a.label
    report: dict = {}
    dev = load(a.dev)
    ydev = np.array([r["cer"] > a.fail for r in dev], float)
    report["dev"] = {"takes": len(dev), "fail_rate": round(float(ydev.mean()), 4)}
    Xd = matrix(dev)
    report["dev"]["auc"] = {k: round(auc(list(Xd[:, i]), list(ydev.astype(int))), 3) for i, k in enumerate(FEATS)}
    report["dev"]["engine_qc"] = prf([not r["qc_ok"] for r in dev], list(ydev.astype(int)))
    det = Logit().fit(Xd, ydev)
    pd = det.predict(Xd)
    report["dev"]["auc"]["detector(in-sample)"] = round(auc(list(pd), list(ydev.astype(int))), 3)
    # grouped CV by case on dev: honest dev AUC
    cases = sorted({r["case"] for r in dev})
    folds = 5
    pcv = np.zeros(len(dev))
    for f in range(folds):
        hold = {c for i, c in enumerate(cases) if i % folds == f}
        tr = np.array([r["case"] not in hold for r in dev])
        m = Logit().fit(Xd[tr], ydev[tr])
        pcv[~tr] = m.predict(Xd[~tr])
    report["dev"]["auc"]["detector(cv)"] = round(auc(list(pcv), list(ydev.astype(int))), 3)
    # threshold: false-reject budget on dev (CV scores)
    neg = np.sort(pcv[ydev == 0])
    thr = float(neg[int((1 - a.max_fpr) * (len(neg) - 1))]) if len(neg) else 0.5
    report["detector"] = {"features": FEATS, "weights": [round(float(w), 3) for w in det.w],
                          "bias": round(float(det.b), 3), "threshold": round(thr, 4)}
    report["dev"]["detector_at_thr"] = prf(list(pcv >= thr), list(ydev.astype(int)))
    eq = report["dev"]["engine_qc"]["false_reject"]
    report["dev"]["detector_recall_at_engine_fpr"] = round(roc_at(pcv, ydev, eq), 3)
    report["dev"]["taxonomy"] = taxonomy(dev, a.fail)
    report["dev"]["auc_within_unit"] = within_unit_auc(dev, a.fail)
    report["dev"]["select"] = select_report(dev, det, thr, a.fail)
    if a.test:
        test = load(a.test)
        yt = np.array([r["cer"] > a.fail for r in test], float)
        Xt = matrix(test)
        pt = det.predict(Xt)
        report["test"] = {"takes": len(test), "fail_rate": round(float(yt.mean()), 4),
                          "auc": {k: round(auc(list(Xt[:, i]), list(yt.astype(int))), 3) for i, k in enumerate(FEATS)},
                          "engine_qc": prf([not r["qc_ok"] for r in test], list(yt.astype(int))),
                          "detector_at_thr": prf(list(pt >= thr), list(yt.astype(int))),
                          "select": select_report(test, det, thr, a.fail)}
        report["test"]["auc"]["detector"] = round(auc(list(pt), list(yt.astype(int))), 3)
        report["test"]["detector_recall_at_engine_fpr"] = round(
            roc_at(pt, yt, report["test"]["engine_qc"]["false_reject"]), 3)
        report["test"]["taxonomy"] = taxonomy(test, a.fail)
        report["test"]["auc_within_unit"] = within_unit_auc(test, a.fail)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    (OUT / f"detect_eval_{a.dev}{('_' + a.test) if a.test else ''}{'' if a.label == 'whisper' else '_' + a.label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
