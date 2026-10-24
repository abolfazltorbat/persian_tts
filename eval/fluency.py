"""Fluency bench: does a whole answer sound SPOKEN, or read chunk by chunk?

    # synthesize + transcribe (any python with onnxruntime + faster-whisper)
    python eval/fluency.py --tag base --voices male_hello_clean
    python eval/fluency.py --tag t04 --knobs '{"temp": 0.4}'
    # naturalness (a python with torch; scores the saved WAVs of each tag)
    python eval/fluency.py --utmos base t04 --utmos-model /path/utmos.jit
    python eval/fluency.py --compare base t04

The model reads at most ~18 phoneme tokens per generation, so a sentence is
several independent generations stitched together, and a listener hears every
join. eval/harness.py asks "were the words right?"; this asks, per join:

  join_sil_ms   silence a listener hears at a mid-clause chunk join (the
                chunk's own tail/head quiet + the inserted gap). Speech has
                none there; a reader stopping to turn the page has ~200 ms.
  reset_st      |pitch jump| across a join, in semitones — a generation that
                starts fresh restarts the intonation instead of continuing it.
  fall_mid_st   pitch fall over the last 250 ms of a chunk that is NOT
                sentence-final, against the chunk's median. A large fall is a
                sentence ending in the middle of a sentence.
  jump_db       |level step| across a join after loudness matching.
  rate_cv       speaking-rate spread across a render's chunks (vowels/s).
  runts         chunks under MIN_CHUNK_TOKENS — read alone, a 2-token chunk
                is an isolated word with its own intonation.
  utmos         UTMOS22 naturalness (needs torch; --utmos). It rewards clean
                audio: the noisy HUMAN reference clips score BELOW the
                synthesis (1.5–2.9 vs 3–4), so it ranks our renders against
                each other and says nothing absolute about "human".
  spk           speaker similarity of the render to the voice's reference clip
                (eval/speaker.py; --speaker). Same speaker, two recordings:
                0.81–0.88; different speakers: 0.02–0.10; today's renders
                against their own reference: ~0.2–0.3.

Every row is (case × voice × seed); a variant is judged on the mean. Knobs
(`--knobs` JSON) set engine attributes ("temp") or tts_onnx module constants
("tts_onnx._CHUNK_GAP") for one run, so a sweep never edits the code.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "eval"

# ── pitch (numpy YIN; no extra dependency) ─────────────────────────────────


def f0_track(x, sr: int, fmin: float = 65.0, fmax: float = 420.0,
             hop: float = 0.01, win: float = 0.04, thresh: float = 0.15):
    """F0 in Hz every `hop` seconds (NaN where unvoiced). YIN on 8 kHz."""
    import numpy as np
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float64)
    if sr != 8000:
        g = np.gcd(int(sr), 8000)
        x = resample_poly(x, 8000 // g, sr // g)
        sr = 8000
    W, H = int(win * sr), int(hop * sr)
    tmin, tmax = int(sr / fmax), int(sr / fmin)
    n = W + tmax
    if len(x) < n + H:
        return np.full(0, np.nan)
    starts = np.arange(0, len(x) - n, H)
    fr = np.stack([x[s:s + n] for s in starts])
    nfft = 1 << int(np.ceil(np.log2(n + W)))
    A = np.fft.rfft(fr, nfft)
    B = np.fft.rfft(fr[:, :W], nfft)
    acf = np.fft.irfft(A * np.conj(B), nfft)[:, :tmax + 1]       # sum x_j x_{j+tau}
    sq = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    e0 = sq[:, W:W + 1]                                           # energy of x[0:W]
    taus = np.arange(tmax + 1)
    e_tau = sq[:, taus + W] - sq[:, taus]                         # energy of x[tau:tau+W]
    d = e0 + e_tau - 2 * acf
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cm, 1e-12)
    loud = np.sqrt(e0[:, 0] / W)
    f0 = np.full(len(fr), np.nan)
    floor = 0.08 * np.percentile(loud, 95) if len(loud) else 0
    for i in range(len(fr)):
        if loud[i] < floor:
            continue
        c = cmnd[i, tmin:tmax + 1]
        below = np.where(c < thresh)[0]
        if not len(below):
            continue
        k = below[0]
        while k + 1 < len(c) and c[k + 1] < c[k]:
            k += 1
        tau = k + tmin
        if 1 <= tau < tmax:                                       # parabolic refine
            a, b, cc = cmnd[i, tau - 1], cmnd[i, tau], cmnd[i, tau + 1]
            den = a - 2 * b + cc
            tau = tau + (0.5 * (a - cc) / den if abs(den) > 1e-12 else 0)
        f0[i] = sr / tau
    return f0


def _st(a: float, b: float) -> float:
    import math
    return 12 * math.log2(b / a)


def _edge_f0(f0, hop: float, seconds: float, end: bool):
    """Median F0 over the first/last `seconds` of VOICED frames."""
    import numpy as np

    v = np.where(~np.isnan(f0))[0]
    if len(v) < 5:
        return None
    k = max(3, int(seconds / hop))
    sel = v[-k:] if end else v[:k]
    return float(np.median(f0[sel]))


def _quiet_edges(p, sr: int) -> tuple[float, float]:
    """Seconds of near-silence at the head and the tail of a chunk."""
    import numpy as np

    h = max(1, int(0.01 * sr))
    n = len(p) // h
    if n < 3:
        return 0.0, 0.0
    fr = np.sqrt((p[: n * h].reshape(n, h) ** 2).mean(axis=1))
    speech = fr[fr > 0.1 * fr.max()]
    thr = 0.12 * (float(np.median(speech)) if len(speech) else fr.max())
    loud = np.where(fr > thr)[0]
    if not len(loud):
        return 0.0, 0.0
    return loud[0] * 0.01, (n - 1 - loud[-1]) * 0.01


_VOWELS = set("aeiouAE")


def join_metrics(sentences: list[list[tuple]], sr: int, chunk_gap: float) -> dict:
    """sentences: per sentence, [(chunk_audio, gap_before, phonemes)]."""
    import numpy as np

    hop = 0.01
    sil, resets, falls_mid, falls_end, jumps, rates = [], [], [], [], [], []
    for segs in sentences:
        levels = [float(np.sqrt((a ** 2).mean())) for a, _, _ in segs if len(a)]
        target = float(np.median(levels)) if levels else 0.0
        info = []
        for a, gap, ph in segs:
            rms = float(np.sqrt((a ** 2).mean())) if len(a) else 0.0
            gain = float(np.clip(target / rms, 0.4, 2.5)) if rms > 1e-6 and target > 1e-6 else 1.0
            f0 = f0_track(a, sr)
            head, tail = _quiet_edges(a, sr)
            dur = len(a) / sr - head - tail
            syl = sum(ch in _VOWELS for ch in ph)
            if dur > 0.3 and syl >= 6:
                rates.append(syl / dur)
            info.append({"a": a, "gain": gain, "f0": f0, "gap": gap, "head": head, "tail": tail})
        for i, c in enumerate(info):
            med = float(np.nanmedian(c["f0"])) if np.any(~np.isnan(c["f0"])) else None
            end = _edge_f0(c["f0"], hop, 0.25, end=True)
            if med and end:
                (falls_end if i == len(info) - 1 else falls_mid).append(-_st(med, end))
            if i == 0:
                continue
            p = info[i - 1]
            if abs(c["gap"] - chunk_gap) < 1e-6:          # a mid-clause chunk join
                sil.append(1000 * (p["tail"] + c["gap"] + c["head"]))
            pe, cs = _edge_f0(p["f0"], hop, 0.25, True), _edge_f0(c["f0"], hop, 0.25, False)
            if pe and cs:
                resets.append(abs(_st(pe, cs)))
            w = int(0.3 * sr)
            pa, ca = p["a"][-w:] * p["gain"], c["a"][:w] * c["gain"]
            if len(pa) and len(ca):
                lp = float(np.sqrt((pa ** 2).mean())) + 1e-9
                lc = float(np.sqrt((ca ** 2).mean())) + 1e-9
                jumps.append(abs(20 * np.log10(lc / lp)))

    def m(v):
        return round(statistics.mean(v), 3) if v else None

    return {"join_sil_ms": m(sil), "n_chunk_joins": len(sil), "reset_st": m(resets),
            "fall_mid_st": m(falls_mid), "fall_end_st": m(falls_end), "jump_db": m(jumps),
            "rate": m(rates),
            "rate_cv": (round(statistics.pstdev(rates) / statistics.mean(rates), 3)
                        if len(rates) >= 2 else None)}


# ── run ─────────────────────────────────────────────────────────────────────


def apply_knobs(eng, knobs: dict) -> None:
    import tts_onnx

    for k, v in knobs.items():
        if k.startswith("tts_onnx."):
            name = k.split(".", 1)[1]
            if not hasattr(tts_onnx, name):
                raise SystemExit(f"tts_onnx has no {name!r}")
            setattr(tts_onnx, name, v)
        else:
            if not hasattr(eng, k):
                raise SystemExit(f"engine has no attribute {k!r}")
            setattr(eng, k, v)


def run(a) -> int:
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(ROOT / "eval"))
    import harness  # path setup, typer shim, Engine/Asr

    harness._preload_cuda()
    import frontend
    import tts_onnx
    from cases import CASES
    from scoring import score

    out_dir = OUT / f"flu_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    box = harness.Engine()
    eng = box.eng
    eng.qc_stats = {}
    knobs = json.loads(a.knobs) if a.knobs else {}
    apply_knobs(eng, knobs)
    asr = harness.Asr(a.asr_device, a.asr_model)

    jobs_log: list[list] = []
    stitched: list[list] = []
    orig_jobs, orig_stitch = eng._jobs_for, eng._stitch

    def jobs_spy(ph):
        j = orig_jobs(ph)
        if j:
            jobs_log.append(j)
        return j

    def stitch_spy(segments):
        stitched.append([(np.array(x, dtype=np.float32), g) for x, g in segments])
        return orig_stitch(segments)

    eng._jobs_for, eng._stitch = jobs_spy, stitch_spy

    cats = set(a.cats.split(",")) if a.cats else None
    cases = [c for c in CASES if (cats is None or c["cat"] in cats)
             and (not a.cases or c["id"] in a.cases.split(","))]
    voices = [v.strip() for v in a.voices.split(",") if v.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    frontend.render(eng, "سلام.", box.voice(voices[0]), seed=0)        # warm

    rows = []
    for c in cases:
        for v in voices:
            for s in seeds:
                jobs_log.clear()
                stitched.clear()
                t0 = time.perf_counter()
                audio, phon = frontend.render(eng, c["text"], box.voice(v), seed=s, pack=a.pack)
                synth = time.perf_counter() - t0
                dur = len(audio) / eng.sample_rate
                hyp = asr.transcribe(audio, eng.sample_rate, c["lang"]) if len(audio) else ""
                sc = score(hyp, c["ref"], c["lang"])
                sents = []
                for segs, jl in zip(stitched, jobs_log):
                    sents.append([(x, g, ch) for (x, g), (ch, _) in zip(segs, jl)])
                jm = join_metrics(sents, eng.sample_rate, tts_onnx._CHUNK_GAP)
                ntok = [len(eng.sp.encode(ch, out_type=int)) for jl in jobs_log for ch, _ in jl]
                row = {"case": c["id"], "lang": c["lang"], "cat": c["cat"], "voice": Path(v).stem,
                       "seed": s, "synth_s": round(synth, 3), "audio_s": round(dur, 3),
                       "hyp": hyp, **sc, **jm, "chunks": len(ntok),
                       "sentences": len(jobs_log),
                       "runts": sum(n < tts_onnx.MIN_CHUNK_TOKENS for n in ntok),
                       "max_tokens": max(ntok) if ntok else 0}
                rows.append(row)
                if not a.no_audio and len(audio):
                    sf.write(out_dir / f"{c['id']}__{Path(v).stem}__s{s}.wav", audio, eng.sample_rate)
                print(f"{c['id']:<15} {Path(v).stem[:12]:<12} s{s} cer={sc['cer']:.3f} "
                      f"chunks={row['chunks']:2d} runts={row['runts']} sil={jm['join_sil_ms']} "
                      f"reset={jm['reset_st']} fall={jm['fall_mid_st']}", flush=True)
    meta = {"tag": a.tag, "knobs": knobs, "pack": a.pack, "voices": voices, "seeds": seeds,
            "when": time.strftime("%Y-%m-%d %H:%M"),
            "providers": eng.s_flow.get_providers(), "qc_stats": eng.qc_stats,
            "temp": eng.temp, "asr_model": a.asr_model}
    write(out_dir, meta, rows)
    print(table([a.tag]))
    return 0


# ── report ──────────────────────────────────────────────────────────────────

METRICS = ["cer", "fail", "utmos", "spk", "join_sil_ms", "reset_st", "fall_mid_st",
           "fall_end_st", "jump_db", "rate", "rate_cv", "runts", "chunks", "rtf"]


def summarize(rows: list[dict]) -> dict:
    def mean(key, sel):
        v = [r[key] for r in sel if r.get(key) is not None]
        return round(statistics.mean(v), 3) if v else None

    def agg(sel):
        if not sel:
            return {}
        return {"n": len(sel), "cer": mean("cer", sel),
                "fail": round(sum(r["cer"] > 0.25 for r in sel) / len(sel), 3),
                **{k: mean(k, sel) for k in METRICS if k not in ("cer", "fail", "rtf")},
                "rtf": round(sum(r["synth_s"] for r in sel) / max(1e-9, sum(r["audio_s"] for r in sel)), 3)}
    out = {"all": agg(rows)}
    for key in ("cat", "voice"):
        for v in sorted({r[key] for r in rows}):
            out[f"{key}:{v}"] = agg([r for r in rows if r[key] == v])
    return out


def write(out_dir: Path, meta: dict, rows: list[dict]) -> None:
    (out_dir / "report.json").write_text(json.dumps(
        {"meta": meta, "summary": summarize(rows), "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def table(tags: list[str], slices: tuple[str, ...] = ("all", "cat:paragraph")) -> str:
    reps = {t: json.loads((OUT / f"flu_{t}" / "report.json").read_text(encoding="utf-8")) for t in tags}
    lines = []
    for sl in slices:
        lines += ["", f"### {sl}", "", "| tag | n | " + " | ".join(METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        for t in tags:
            s = reps[t]["summary"].get(sl) or {}
            if not s:
                continue
            lines.append(f"| {t} | {s['n']} | " + " | ".join(
                "—" if s.get(k) is None else f"{s[k]:.3f}".rstrip("0").rstrip(".") for k in METRICS) + " |")
    return "\n".join(lines)


def utmos(tags: list[str], model_path: str) -> int:
    """Score each tag's saved WAVs with UTMOS22 (a TorchScript traced on CUDA)."""
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=dev).eval()
    for t in tags:
        d = OUT / f"flu_{t}"
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        for r in rep["rows"]:
            wav = d / f"{r['case']}__{r['voice']}__s{r['seed']}.wav"
            if not wav.is_file():
                continue
            x, sr = sf.read(wav, dtype="float32")
            x = resample_poly(x, 2, 3) if sr == 24000 else x
            ten = torch.from_numpy((np.clip(x, -1, 1) * 32767).astype(np.int16)).unsqueeze(0).to(dev)
            with torch.no_grad():
                r["utmos"] = round(float(model(ten).item()), 3)
        write(d, rep["meta"], rep["rows"])
        print(t, rep["meta"].get("knobs"), "utmos all",
              summarize(rep["rows"])["all"].get("utmos"))
    return 0


def speaker(tags: list[str], model_path: str, spk_ref: str = "") -> int:
    """Speaker similarity of each saved render to its voice's reference clip
    (or to `spk_ref`, e.g. the ORIGINAL clip when the tag used a new one)."""
    sys.path.insert(0, str(ROOT / "eval"))
    from speaker import SpeakerEmbedder

    emb = SpeakerEmbedder(model_path)
    refs: dict[str, object] = {}
    for t in tags:
        d = OUT / f"flu_{t}"
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        for r in rep["rows"]:
            wav = d / f"{r['case']}__{r['voice']}__s{r['seed']}.wav"
            if not wav.is_file():
                continue
            ref = spk_ref or str(ROOT / "voices" / f"{r['voice']}.wav")
            if ref not in refs:
                refs[ref] = emb.embed(ref)
            r["spk"] = round(float(refs[ref] @ emb.embed(wav)), 3)
        write(d, rep["meta"], rep["rows"])
        print(t, "spk all", summarize(rep["rows"])["all"].get("spk"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--voices", default="male_hello_clean")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--cats", default="", help="comma list of case categories")
    ap.add_argument("--cases", default="", help="comma list of case ids")
    ap.add_argument("--knobs", default="", help="JSON: engine attrs / tts_onnx.CONSTS")
    ap.add_argument("--asr-device", default="auto")
    ap.add_argument("--asr-model", default="turbo", choices=["turbo", "small"])
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--pack", action="store_true", help="read comma phrases in one breath")
    ap.add_argument("--utmos", nargs="+", metavar="TAG")
    ap.add_argument("--utmos-model", default="")
    ap.add_argument("--speaker", nargs="+", metavar="TAG")
    ap.add_argument("--speaker-model", default="")
    ap.add_argument("--spk-ref", default="", help="compare against this clip instead")
    ap.add_argument("--compare", nargs="+", metavar="TAG")
    ap.add_argument("--slices", default="all,cat:paragraph")
    a = ap.parse_args()
    if a.utmos:
        return utmos(a.utmos, a.utmos_model)
    if a.speaker:
        return speaker(a.speaker, a.speaker_model, a.spk_ref)
    if a.compare:
        print(table(a.compare, tuple(a.slices.split(","))))
        return 0
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
