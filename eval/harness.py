"""Speech accuracy + speed harness: synthesize → Whisper → score.

    # from persian_tts/ (any python with onnxruntime + faster-whisper)
    python eval/harness.py --tag baseline --pipeline chatbot
    python eval/harness.py --tag v2 --pipeline v2 --seeds 1,2,3
    python eval/harness.py --tag quick --cases fa_clock,mix_gpt --seeds 1
    python eval/harness.py --compare baseline v2

Every row is (case × voice × seed). The engine is stochastic, so a pipeline is
judged on the MEAN over seeds — one lucky seed proves nothing. Audio lands in
output/eval/<tag>/ for listening; numbers in report.json / report.md.

Pipelines
  chatbot  what Dana does today: chunking.to_speech_text → the engine's
           sentence/phrase plan (no prepare_persian), read from the chatbot
           checkout so the baseline is the deployed behaviour, not a copy.
  speak    persian_tts/speak.py (prepare_persian + the same plan).
  v2       scripts/frontend.py — the new text front-end + engine settings.

ASR: faster-whisper large-v3-turbo (the model Dana's STT ships), beam 5,
temperature 0, language forced per case (fa for Persian and mixed, en for
English), no prompt — an initial prompt would leak the reference into the
transcript.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHATBOT = ROOT.parent / "chatbot" / "ai_backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

# normalize_fa (upstream file) imports typer for its CLI tail only; a venv
# without it can still use the library half.
try:
    import typer  # noqa: F401
except ImportError:
    class _Any:
        def __call__(self, *a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]
            return _Any()

        def __getattr__(self, name):
            return _Any()

    _t = types.ModuleType("typer")
    _t.__getattr__ = lambda name: _Any()          # type: ignore[attr-defined]
    sys.modules["typer"] = _t

from cases import CASES  # noqa: E402
from scoring import score  # noqa: E402

WHISPER_DIRS = {
    "turbo": [CHATBOT.parent / "storage" / "models" / "whisper" / "large-v3-turbo"],
    # an independent recogniser, to check that the ASR self-check is not just
    # teaching the engine to please the one model that also scores it
    "small": [Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-small"],
}


def _model_dir(name: str) -> Path | None:
    for p in WHISPER_DIRS[name]:
        if (p / "model.bin").is_file():
            return p
        snaps = sorted((p / "snapshots").glob("*/model.bin")) if (p / "snapshots").is_dir() else []
        if snaps:
            return snaps[-1].parent
    return None


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preload_cuda() -> None:
    dev = CHATBOT / "app" / "backend" / "stt" / "device.py"
    if dev.is_file():
        try:
            _load_file_module("_dana_stt_device", dev).preload_cuda()
        except Exception as exc:                    # pragma: no cover
            print("cuda preload skipped:", exc)


# ── pipelines ───────────────────────────────────────────────────────────────
class Engine:
    """One loaded engine shared by every pipeline (the graphs are ~500 MB)."""

    def __init__(self):
        from speak import PersianTTS

        self.tts = PersianTTS()
        self.eng = self.tts.engine
        self.sr = self.tts.sample_rate
        print("engine providers:", self.eng.s_flow.get_providers())

    def voice(self, v: str) -> str:
        return v if "/" in v else str(ROOT / "voices" / f"{v}.wav")

    def render_plan(self, text: str, voice: str, seed: int, *, prepare=None):
        """The chatbot/speak render: sentences → phrase plans → audio."""
        import numpy as np
        from tts_onnx import SENTENCE_GAP, plan_phrases, split_sentences

        if prepare:
            text = prepare(text)
        self.eng.rng = np.random.default_rng(seed)
        sentences = []
        for sent in split_sentences(text):
            plan = plan_phrases(sent, self.eng._g2p, self.eng.sp)
            if plan:
                sentences.append(plan)
        phon = " | ".join(" ".join(p for p, _ in s) for s in sentences)
        if not sentences:
            return np.zeros(0, np.float32), phon
        silence = np.zeros(int(SENTENCE_GAP * self.sr), np.float32)
        parts = []
        for plan in sentences:
            parts.append(self.eng.synthesize(plan, self.voice(voice)))
            parts.append(silence)
        return np.concatenate(parts[:-1]), phon


def make_pipeline(name: str, engine: Engine):
    if name == "chatbot":
        chunking = _load_file_module(
            "_dana_chunking", CHATBOT / "app" / "backend" / "tts" / "chunking.py")

        def run(text, voice, seed, lang):
            spoken = chunking.to_speech_text(text, lang="en" if lang == "en" else "fa")
            return engine.render_plan(spoken, voice, seed)
        return run
    if name == "speak":
        from persian_text import prepare_persian

        def run(text, voice, seed, lang):
            return engine.render_plan(text, voice, seed, prepare=prepare_persian)
        return run
    if name == "v2":
        import frontend

        def run(text, voice, seed, lang):
            return frontend.render(engine.eng, text, engine.voice(voice), seed=seed)
        return run
    raise SystemExit(f"unknown pipeline {name!r}")


# ── ASR ─────────────────────────────────────────────────────────────────────
class Asr:
    def __init__(self, device: str = "auto", name: str = "turbo"):
        from faster_whisper import WhisperModel

        path = _model_dir(name)
        if path is None:
            raise SystemExit(f"whisper {name} not found; looked in "
                             + ", ".join(map(str, WHISPER_DIRS[name])))
        _preload_cuda()
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
            except Exception:
                device = "cpu"
        self.model = WhisperModel(str(path), device=device,
                                  compute_type="float16" if device == "cuda" else "int8")
        print("asr:", name, path.name, device)

    def transcribe(self, audio, sr: int, lang: str) -> str:
        import numpy as np
        from scipy.signal import resample_poly

        a = np.asarray(audio, dtype=np.float32)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
        for attempt in range(3):
            try:
                segs, _ = self.model.transcribe(
                    a, language="en" if lang == "en" else "fa", beam_size=5,
                    temperature=0.0, condition_on_previous_text=False, vad_filter=False,
                    without_timestamps=True)
                return " ".join(s.text.strip() for s in segs).strip()
            except RuntimeError as exc:        # a shared, busy GPU: try again
                print("asr retry:", str(exc)[:80], flush=True)
                time.sleep(1.0 + attempt)
        raise RuntimeError("transcription failed 3 times")


# ── run ─────────────────────────────────────────────────────────────────────
def select_cases(spec: str) -> list[dict]:
    if not spec:
        return CASES
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.startswith("cat:"):
            out += [c for c in CASES if c["cat"] == tok[4:]]
        elif tok.startswith("lang:"):
            out += [c for c in CASES if c["lang"] == tok[5:]]
        else:
            out += [c for c in CASES if c["id"] == tok]
    return out


def summarize(rows: list[dict]) -> dict:
    def agg(sel):
        if not sel:
            return {}
        return {
            "n": len(sel),
            "cer": round(statistics.mean(r["cer"] for r in sel), 4),
            "wer": round(statistics.mean(r["wer"] for r in sel), 4),
            "fail_rate": round(sum(r["cer"] > 0.25 for r in sel) / len(sel), 4),
            "rtf": round(sum(r["synth_s"] for r in sel) / max(1e-9, sum(r["audio_s"] for r in sel)), 3),
        }
    out = {"all": agg(rows)}
    for key in ("lang", "cat", "voice"):
        for v in sorted({r[key] for r in rows}):
            out[f"{key}:{v}"] = agg([r for r in rows if r[key] == v])
    return out


def write_report(out_dir: Path, meta: dict, rows: list[dict]) -> dict:
    summ = summarize(rows)
    (out_dir / "report.json").write_text(json.dumps(
        {"meta": meta, "summary": summ, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    lines = [f"# Speech eval — {meta['tag']} ({meta['pipeline']})", "",
             f"{meta['when']} · voices {', '.join(meta['voices'])} · seeds {meta['seeds']} · "
             f"providers {meta['providers']}", "",
             "| slice | n | CER | WER | fail (CER>0.25) | RTF |", "|---|---|---|---|---|---|"]
    for k, v in summ.items():
        if v:
            lines.append(f"| {k} | {v['n']} | {v['cer']:.3f} | {v['wer']:.3f} | "
                         f"{v['fail_rate']:.0%} | {v['rtf']:.2f} |")
    # per case, mean over seeds
    lines += ["", "## Per case (mean over seeds)", "",
              "| case | CER | worst-seed transcript |", "|---|---|---|"]
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    for cid, rs in sorted(by_case.items(), key=lambda kv: -statistics.mean(r["cer"] for r in kv[1])):
        worst = max(rs, key=lambda r: r["cer"])
        lines.append(f"| {cid} | {statistics.mean(r['cer'] for r in rs):.3f} | "
                     f"{worst['hyp'][:110]} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summ


def compare(tags: list[str]) -> None:
    base = ROOT / "output" / "eval"
    reps = {t: json.loads((base / t / "report.json").read_text(encoding="utf-8")) for t in tags}
    keys = list(reps[tags[0]]["summary"].keys())
    print("| slice | " + " | ".join(f"{t} CER" for t in tags) + " |")
    print("|---|" + "---|" * len(tags))
    for k in keys:
        vals = [reps[t]["summary"].get(k, {}).get("cer") for t in tags]
        print(f"| {k} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
    # per-case
    print()
    cases = sorted({r["case"] for t in tags for r in reps[t]["rows"]})
    print("| case | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for c in cases:
        vals = []
        for t in tags:
            rs = [r["cer"] for r in reps[t]["rows"] if r["case"] == c]
            vals.append(f"{statistics.mean(rs):.3f}" if rs else "—")
        print(f"| {c} | " + " | ".join(vals) + " |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--pipeline", default="chatbot", choices=["chatbot", "speak", "v2"])
    ap.add_argument("--voices", default="female_narration")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--cases", default="", help="ids, cat:<cat> or lang:<fa|en|mix>")
    ap.add_argument("--asr-device", default="auto")
    ap.add_argument("--asr-model", default="turbo", choices=["turbo", "small"],
                    help="recogniser that SCORES the audio")
    ap.add_argument("--verify", default="", choices=["", "turbo", "small"],
                    help="ASR self-check during synthesis (v2 only), with this model")
    ap.add_argument("--verify-threshold", type=float, default=None)
    ap.add_argument("--workers", type=int, default=0, help="engine parallel units")
    ap.add_argument("--no-audio", action="store_true", help="do not keep WAVs")
    ap.add_argument("--compare", nargs="+", metavar="TAG")
    ap.add_argument("--retranscribe", nargs="+", metavar="TAG",
                    help="re-transcribe a run's saved WAVs with --asr-model and score")
    ap.add_argument("--rescore", nargs="+", metavar="TAG",
                    help="re-score saved transcripts with the current scorer")
    a = ap.parse_args()

    if a.retranscribe:
        # Score the SAVED audio of a run with another recogniser — the check
        # that the ASR self-check is not just pleasing the model that scores.
        import soundfile as sf
        from cases import by_id

        _preload_cuda()
        asr = Asr(a.asr_device, a.asr_model)
        for tag in a.retranscribe:
            d = ROOT / "output" / "eval" / tag
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
            out = ROOT / "output" / "eval" / f"{tag}__{a.asr_model}"
            out.mkdir(parents=True, exist_ok=True)
            rows = []
            for r in rep["rows"]:
                wav = d / f"{r['case']}__{r['voice']}__s{r['seed']}.wav"
                if not wav.is_file():
                    continue
                audio, sr = sf.read(wav)
                hyp = asr.transcribe(audio, sr, r["lang"])
                rows.append({**r, "hyp": hyp, **score(hyp, by_id(r["case"])["ref"], r["lang"])})
            summ = write_report(out, {**rep["meta"], "asr_model": a.asr_model,
                                      "tag": f"{tag}__{a.asr_model}"}, rows)
            print(tag, a.asr_model, summ["all"])
        return 0
    if a.rescore:
        from cases import by_id

        for tag in a.rescore:
            d = ROOT / "output" / "eval" / tag
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
            for r in rep["rows"]:
                r.update(score(r["hyp"], by_id(r["case"])["ref"], r["lang"]))
            summ = write_report(d, rep["meta"], rep["rows"])
            print(tag, summ["all"])
        return 0
    if a.compare:
        compare(a.compare)
        return 0

    import soundfile as sf

    out_dir = ROOT / "output" / "eval" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    _preload_cuda()          # before the engine: its CUDA EP dlopens cuBLAS by name
    engine = Engine()
    engine.eng.qc_stats = {}
    if a.workers:
        engine.eng.workers = a.workers
    run = make_pipeline(a.pipeline, engine)
    asr = Asr(a.asr_device, a.asr_model)
    if a.verify:
        from verify import AsrVerifier

        vmodel = asr.model if a.verify == a.asr_model else Asr(a.asr_device, a.verify).model
        engine.eng.verifier = AsrVerifier(vmodel, threshold=a.verify_threshold)
    voices = [v.strip() for v in a.voices.split(",") if v.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    cases = select_cases(a.cases)

    # warm both models off the clock
    run("سلام.", voices[0], 0, "fa")

    rows = []
    t_all = time.perf_counter()
    for c in cases:
        for v in voices:
            for s in seeds:
                t0 = time.perf_counter()
                audio, phon = run(c["text"], v, s, c["lang"])
                synth = time.perf_counter() - t0
                dur = len(audio) / engine.sr
                hyp = asr.transcribe(audio, engine.sr, c["lang"]) if len(audio) else ""
                sc = score(hyp, c["ref"], c["lang"])
                row = {"case": c["id"], "lang": c["lang"], "cat": c["cat"], "voice": Path(v).stem,
                       "seed": s, "synth_s": round(synth, 3), "audio_s": round(dur, 3),
                       "hyp": hyp, "phonemes": phon, **sc}
                rows.append(row)
                if not a.no_audio and len(audio):
                    sf.write(out_dir / f"{c['id']}__{Path(v).stem}__s{s}.wav", audio, engine.sr)
                print(f"{c['id']:<16} {Path(v).stem[:10]:<10} s{s} cer={sc['cer']:.3f} "
                      f"rtf={synth / max(dur, 1e-6):.2f}  {hyp[:70]}", flush=True)

    meta = {"tag": a.tag, "pipeline": a.pipeline, "voices": voices, "seeds": seeds,
            "when": time.strftime("%Y-%m-%d %H:%M"), "cases": len(cases),
            "providers": engine.eng.s_flow.get_providers(),
            "wall_s": round(time.perf_counter() - t_all, 1),
            "device_env": os.environ.get("PARSIGO_DEVICE", "cpu"),
            "asr_model": a.asr_model, "verify": a.verify,
            "verify_threshold": getattr(engine.eng.verifier, "threshold", None),
            "qc_stats": engine.eng.qc_stats,
            "verify_calls": getattr(engine.eng.verifier, "calls", 0),
            "verify_seconds": round(getattr(engine.eng.verifier, "seconds", 0.0), 1)}
    summ = write_report(out_dir, meta, rows)
    print()
    for k, v in summ.items():
        if v and (k == "all" or k.startswith("lang:") or k.startswith("cat:")):
            print(f"{k:<16} n={v['n']:<4} CER {v['cer']:.3f}  WER {v['wer']:.3f}  "
                  f"fail {v['fail_rate']:.0%}  RTF {v['rtf']:.2f}")
    if engine.eng.qc_stats is not None:
        print("qc:", engine.eng.qc_stats, "verify calls", meta["verify_calls"],
              "verify s", meta["verify_seconds"])
    print(f"\nreport: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
