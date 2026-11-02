"""Study B probe: does a longer voice prompt make takes more robust?

Prompt variants for one voice (all <= 5 s, the trained prompt window):
  ref       the reference clip as production uses it
  ref+ext   clip, then the model's own reading of a neutral sentence in that
            voice (self-extension, made once per voice)
  ext+ref   the same two pieces, clip last (closest to the take)
  ext       the self-extension alone
Each unit of the chosen sentences is generated --takes times per variant (no
retry) and judged with Whisper (consonant-skeleton CER).

    python research/prompt_probe.py --set dev --cats short --voices male_hello_clean --takes 6
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from detect_corpus import cases_for  # noqa: E402

# neutral, phonetically varied, no numbers or names (phonemised by the engine's G2P)
EXT_TEXT = "امروز هوا کمی سرد است و باد می‌وزد."   # = EXT_TEXTS["A"]
MAX_PROMPT = int(5 * 12.5)


def speech_span(lab: core.Lab, z: np.ndarray, audio: np.ndarray, pad: int = 1) -> np.ndarray:
    db = core.frame_db(audio, len(z))
    idx = np.where(db > db.max() - 30.0)[0]
    if not len(idx):
        return z
    return z[max(0, idx[0] - pad): idx[-1] + 1 + pad]


EXT_TEXTS = {
    "A": "امروز هوا کمی سرد است و باد می‌وزد.",                 # neutral statement
    "B": "سلام، خوشحالم که دوباره با شما صحبت می‌کنم.",          # conversational greeting
    "C": "بله، حتماً. همین الان انجامش می‌دهم.",                  # short replies
    "D": "کتاب‌های زیادی در این کتابخانه نگهداری می‌شود.",        # formal statement
    # added 10-31 08:20: is "a neutral declarative statement" the property that matters?
    "E": "دیروز باران آمد و خیابان‌ها خیس شدند.",                 # neutral, past, two clauses (like A)
    "F": "قطار ساعت هشت صبح از ایستگاه حرکت می‌کند.",              # neutral, present, one clause
    "G": "پدرم هر روز صبح در پارک قدم می‌زند.",                    # neutral, habitual
}


def speech_span_pad(lab: core.Lab, z: np.ndarray, audio: np.ndarray, lead: int = 1, tail: int = 1) -> np.ndarray:
    db = core.frame_db(audio, len(z))
    idx = np.where(db > db.max() - 30.0)[0]
    if not len(idx):
        return z
    return z[max(0, idx[0] - lead): idx[-1] + 1 + tail]


_EXT_MEMO: dict = {}


def self_extension(lab: core.Lab, voice: str, text: str = EXT_TEXT, tries: int = 4,
                   select: str = "asr", tail: int = 1) -> tuple[np.ndarray, dict]:
    """The model's own reading of `text` from the clip, as RAW latents (speech
    span + `tail` frames of the model's own trailing quiet). One-off per voice:
    of `tries` fixed-seed takes keep the cleanest by the ASR judge
    (select="asr", an offline voice-setup step) or, ASR-free, the take whose
    speech length is the median (select="median")."""
    key = (voice, text, tries, select, tail)
    if key in _EXT_MEMO:
        return _EXT_MEMO[key]
    raw = lab.prompt_latents(voice)
    ph = " ".join(u[0] for u in lab.units(text))
    takes = []
    for s in range(tries):
        cache, off = lab.prompt_cache(raw)
        tk = core.generate(lab, cache, off, ph, np.random.default_rng(500 + s))
        audio = core.decode(lab, tk)
        speech, qc = core.engine_qc(lab, audio, ph)
        hyp, cer = core.judge(lab, speech, ph, "fa") if select == "asr" else ("", float("nan"))
        takes.append((s, cer, hyp, speech_span_pad(lab, tk.z, audio, tail=tail), qc["solid_s"]))
    if select == "asr":
        s, cer, hyp, z, _ = min(takes, key=lambda t: (t[1], t[0]))
    else:
        order = sorted(takes, key=lambda t: t[4])
        s, cer, hyp, z, _ = order[len(order) // 2]
    out = (lab.to_raw(z), {"seed": 500 + s, "cer": None if cer != cer else round(cer, 3), "hyp": hyp,
                           "frames": int(len(z)), "select": select, "tail": tail})
    _EXT_MEMO[key] = out
    return out


def ext_from_spec(lab: core.Lab, voice: str, spec: str) -> np.ndarray:
    """spec = TEXTKEYS[:pN][:m], e.g. "A", "A:p6", "AB", "B:m"."""
    parts = spec.split(":")
    keys, tail, select, cut = parts[0], 1, "asr", False
    for p in parts[1:]:
        if p.startswith("p"):
            tail = int(p[1:])
        elif p == "m":
            select = "median"
        elif p == "c":
            cut = True
    zs = [self_extension(lab, voice, EXT_TEXTS[k], select=select, tail=tail)[0] for k in keys]
    z = np.concatenate(zs)
    if cut:
        # mechanism control: drop the extension's last third, so the prompt
        # ends MID-utterance (same voice, same kind of latents, no ending)
        z = z[: max(1, int(len(z) * 2 / 3))]
    return z


ROOM_RMS = 0.0011   # the engine's room-tone level (OnnxTts._ROOM_TONE_RMS)


def room_tone(n: int, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    x = np.convolve(r.standard_normal(n + 64), np.ones(8) / 8, mode="same")[32:32 + n]
    return (x * ROOM_RMS / max(1e-9, float(np.sqrt((x ** 2).mean())))).astype(np.float32)


def clip_audio(lab: core.Lab, voice: str) -> np.ndarray:
    """The reference clip as voice_cache prepares it (24 kHz, 5 s cap, hot-onset trim)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    a, sr = sf.read(lab.voice_path(voice))
    a = (a.mean(axis=1) if a.ndim > 1 else a).astype(np.float32)
    if sr != lab.sr:
        g = np.gcd(int(sr), lab.sr)
        a = resample_poly(a, lab.sr // g, sr // g).astype(np.float32)
    a = a[: 5 * lab.sr]
    a, _ = core.tts_onnx.trim_hot_onset(a, lab.sr)
    return a


def build_audio(lab: core.Lab, voice: str, ext_raw: np.ndarray, kind: str) -> np.ndarray:
    """Audio-domain prompts: clip [+ extension audio] [+ pause of room tone],
    encoded by Mimi as ONE signal; the LAST 5 s are kept so the prompt's
    ending (what the take continues from) is never cut. kind e.g.
    'a:ref', 'a:ref+p4', 'a:ref+ext', 'a:ref+ext+p4' (p4 = 0.4 s, p8 = 0.8 s)."""
    parts = [clip_audio(lab, voice)]
    for tok in kind.split(":", 1)[1].split("+")[1:]:
        if tok == "ext":
            z = lab.to_norm(ext_raw)
            parts.append(lab.eng._decode_all([r[None] for r in z]).astype(np.float32))
        elif tok.startswith("p"):
            parts.append(room_tone(int(int(tok[1:]) / 10 * lab.sr)))
    audio = np.concatenate(parts)[-5 * lab.sr:]
    return lab.eng.s_enc.run(None, {"audio": audio[None, None, :]})[0][0].astype(np.float32)


def build(lab: core.Lab, voice: str, ext_raw: np.ndarray, kind: str) -> np.ndarray:
    if kind.startswith("a:"):
        return build_audio(lab, voice, ext_raw, kind)
    ref = lab.prompt_latents(voice)
    if kind.startswith("ref+file:"):
        # an extension file prepared elsewhere (prep_voice_ext / prepare_extension / ext_spk_select)
        d = Path(kind.split(":", 1)[1])
        ext = np.load(d / f"{Path(lab.voice_path(voice)).stem}.ext.npy").astype(np.float32)
        room = MAX_PROMPT - len(ref)
        return np.concatenate([ref, ext[-room:] if room > 0 else ext[:0]])
    if kind == "ref+ref":
        # mechanism control: more real speech of the voice, same (pause-less) ending
        return np.concatenate([ref, ref])[-MAX_PROMPT:]
    if kind.startswith("ref+ext:"):
        # clip first, then the extension(s); the 5 s window keeps the clip and
        # the END of the extension (what the take continues from).
        # ":min<s>" guarantees <s> seconds of extension by cutting the clip's start.
        spec = kind.split(":", 1)[1]
        min_s = 0.0
        if ":min" in spec:
            spec, _, m = spec.partition(":min")
            min_s = float(m)
        ext = ext_from_spec(lab, voice, spec)
        keep = min(len(ext), int(round(min_s * 12.5)))
        if len(ref) + keep > MAX_PROMPT:
            ref = ref[-(MAX_PROMPT - keep):]
        room = MAX_PROMPT - len(ref)
        return np.concatenate([ref, ext[-room:] if room > 0 else ext[:0]])
    if kind == "ref":
        return ref
    if kind == "ext":
        return ext_raw[-MAX_PROMPT:]
    room = MAX_PROMPT - len(ref)
    ext = ext_raw[-room:] if room > 0 else ext_raw[:0]
    return np.concatenate([ref, ext]) if kind == "ref+ext" else np.concatenate([ext, ref])


def carrier_prime(lab: core.Lab, voice: str, spec: str):
    word, _, gap = spec.partition("@")
    gap = int(gap or 2)
    ph = " ".join(u[0] for u in lab.units(word))
    raw = lab.prompt_latents(voice)
    takes = []
    for s in range(4):
        cache, off = lab.prompt_cache(raw)
        tk = core.generate(lab, cache, off, ph, np.random.default_rng(1000 + s))
        audio = core.decode(lab, tk)
        db = core.frame_db(audio, len(tk.z))
        idx = np.where(db > db.max() - 30.0)[0]
        if len(idx):
            takes.append((int(idx[-1]), tk.z))
    takes.sort(key=lambda t: t[0])
    last, z = takes[len(takes) // 2]
    return lab.eng.sp.encode(ph, out_type=int), z[: last + 1 + gap].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--cats", default="short")
    ap.add_argument("--voices", default="male_hello_clean")
    ap.add_argument("--kinds", default="ref,ref+ext,ext+ref,ext")
    ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="probe")
    a = ap.parse_args()

    lab = core.Lab()
    cases = cases_for(a.set)
    if a.cats:
        cases = [c for c in cases if c["cat"] in set(a.cats.split(","))]
    out = core.OUT / f"prompt_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    import soundfile as sf
    wav_dir = out / "wav"
    wav_dir.mkdir(exist_ok=True)
    units = [(c, ui, ch, lang) for c in cases for ui, (ch, _g, lang, _s) in enumerate(lab.units(c["text"]))]
    for voice in a.voices.split(","):
        ext_raw, info = self_extension(lab, voice)
        print(f"{voice}: self-extension {info}", flush=True)
        for kind in a.kinds.split(","):
            # "carrier:<word>[@<gap>]" = clip prompt + continuation priming: the
            # carrier's own latents (median of 4 fixed-seed takes, cut after its
            # speech + gap frames) teacher-forced before the unit's text
            prime = None
            if kind.startswith("carrier:"):
                prime = carrier_prime(lab, voice, kind.split(":", 1)[1])
                prompt = lab.prompt_latents(voice)
            else:
                prompt = build(lab, voice, ext_raw, kind)

            def work(job):
                c, ui, ch, lang, k = job
                cache, off = lab.prompt_cache(prompt)
                rid = f"{c['id']}|u{ui}|{voice}|k{k}"   # same seed for every kind: paired
                tk = core.generate(lab, cache, off, ch, np.random.default_rng(core.stable_seed(rid)),
                                   prime=prime)
                audio = core.decode(lab, tk, prime_z=prime[1] if prime else None)
                speech, qc = core.engine_qc(lab, audio, ch)
                return job, speech, qc, tk

            jobs = [(c, ui, ch, lang, k) for (c, ui, ch, lang) in units for k in range(a.takes)]
            t0 = time.time()
            cers = []
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                for job, speech, qc, tk in ex.map(work, jobs):
                    c, ui, ch, lang, k = job
                    hyp, cer = core.judge(lab, speech, ch, lang)
                    cers.append(cer)
                    # keep the take for the second judge (research/w2v_judge.py)
                    wid = f"{voice.split('/')[-1].removesuffix('.wav')}__{kind.replace(':', '_').replace('+', '-').replace('@', 'g').replace('/', '_')}__{c['id']}__u{ui}__k{k}"
                    sf.write(wav_dir / f"{wid}.wav", speech, lab.sr)
                    core.append_jsonl(rows_path, {"wid": wid, "voice": voice, "kind": kind, "case": c["id"],
                                                  "cat": c["cat"], "unit": ui, "take": k, "chunk": ch,
                                                  "tokens": qc["tokens"], "prompt_frames": int(len(prompt)),
                                                  "cer": round(cer, 4), "hyp": hyp, "qc_ok": qc["qc_ok"],
                                                  "eos_step": tk.eos_step})
            print(f"{voice:24s} {kind:8s} prompt={len(prompt)/12.5:.2f}s n={len(cers)} "
                  f"cer={st.mean(cers):.3f} fail>.3={sum(x > 0.3 for x in cers)/len(cers):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
