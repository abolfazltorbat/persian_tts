"""Shared plumbing for the A/B studies (ASR-free take verification, continuation priming).

Nothing here changes the engine: generation is re-implemented step for step
from OnnxTts._generate_chunk / voice_cache so it can (a) expose what the
engine throws away -- the per-step EOS flag, the latents, the prompt latents --
and (b) take a different conditioning (prompt latents, teacher-forced
carrier latents) without touching production code.

Latent spaces: the flow LM reads and writes NORMALIZED latents
z = (x - emb_mean) / emb_std; the Mimi encoder emits RAW latents x, and the
voice prompt is conditioned on RAW latents (x @ speaker_proj.T). So a
generated take becomes a usable voice prompt by x = z * emb_std + emb_mean.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
OUT = ROOT / "output" / "research"
sys.path.insert(0, str(ROOT / "eval"))
import harness  # noqa: E402  (path setup, typer shim)

import frontend  # noqa: E402
import tts_onnx  # noqa: E402
from verify import AsrVerifier, skeleton_cer  # noqa: E402

FRAME = 1920            # samples per latent at 24 kHz / 12.5 Hz
VOWELS = set("aAeiou")


# ── sentence sets ──────────────────────────────────────────────────────────
_PUNCT = re.compile(r"[.,!?؟،؛:;«»\"()\-]")


def load_set(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cid, lang, cat, text = line.split("\t", 3)
        text = text.replace("\\n", "\n")          # sets write line breaks as \n
        ref = re.sub(r"\s+", " ", _PUNCT.sub(" ", text)).strip()
        rows.append({"id": cid, "lang": lang, "cat": cat, "text": text, "ref": ref})
    return rows


def bench_cases() -> list[dict]:
    from cases import CASES
    return [dict(c) for c in CASES]


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


# ── lab: engine + ASR ──────────────────────────────────────────────────────
class Lab:
    def __init__(self, asr: bool = True, asr_model: str = "turbo"):
        harness._preload_cuda()
        self.box = harness.Engine()
        self.eng = self.box.eng
        self.sr = self.eng.sample_rate
        self.asr = harness.Asr("auto", asr_model) if asr else None
        # the judge: beam 5, digits suppressed (a spoken number is compared as words)
        self.judge = AsrVerifier(self.asr.model, beam_size=5) if asr else None
        self._prompts: dict = {}

    def voice_path(self, v: str) -> str:
        return self.box.voice(v)

    # sentences -> generation units, exactly as production plans them
    def units(self, text: str) -> list[tuple[str, float, str, int]]:
        """-> [(chunk_phonemes, gap_before, lang, sentence_index)]"""
        sents, plans = frontend.plan(self.eng, text)
        out = []
        for si, (s, p) in enumerate(zip(sents, plans)):
            for ch, gap in self.eng._jobs_for(p, s.lang):
                out.append((ch, gap, s.lang, si))
        return out

    # ── prompts ──
    def prompt_latents(self, voice: str) -> np.ndarray:
        """RAW Mimi latents of a reference clip, prepared exactly as
        OnnxTts.voice_cache prepares them (resample, 5 s cap, hot-onset trim)."""
        if voice in self._prompts:
            return self._prompts[voice]
        import soundfile as sf
        from scipy.signal import resample_poly

        eng = self.eng
        audio, sr = sf.read(self.voice_path(voice))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != eng.sample_rate:
            g = np.gcd(int(sr), eng.sample_rate)
            audio = resample_poly(audio, eng.sample_rate // g, sr // g).astype(np.float32)
        audio = audio[: 5 * eng.sample_rate]
        audio, _ = tts_onnx.trim_hot_onset(audio, eng.sample_rate)
        lat = eng.s_enc.run(None, {"audio": audio[None, None, :]})[0][0]   # [T, ldim] raw
        self._prompts[voice] = lat.astype(np.float32)
        return self._prompts[voice]

    def prompt_cache(self, raw: np.ndarray):
        """RAW prompt latents -> (flow KV cache, offset), as voice_cache does."""
        eng = self.eng
        cond = (raw @ eng.spk_proj.T).astype(np.float32)
        text_emb = np.concatenate([eng.bos_voice, cond])[None]
        cache = np.zeros((eng.L, 2, eng.CAP, eng.H, eng.D), np.float32)
        _, _, cache, off = eng._flow_step(np.zeros((1, 0, eng.ldim), np.float32), text_emb, 0,
                                          np.zeros((1, eng.ldim), np.float32), cache)
        return cache, off

    def to_raw(self, z: np.ndarray) -> np.ndarray:
        return (z * self.eng.emb_std + self.eng.emb_mean).astype(np.float32)

    def to_norm(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.eng.emb_mean) / self.eng.emb_std).astype(np.float32)


# ── instrumented generation ────────────────────────────────────────────────
@dataclass
class Take:
    chunk: str
    z: np.ndarray                      # [T, ldim] normalized latents of the TARGET
    eos_step: int | None               # generation step on which EOS first fired
    max_gen: int
    n_prime: int = 0                   # teacher-forced carrier frames (not in z)
    audio: np.ndarray | None = None    # decoded TARGET audio (raw, untrimmed)
    extra: dict = field(default_factory=dict)


def generate(lab: Lab, cache, off, chunk: str, rng: np.random.Generator,
             prime: tuple[list[int], np.ndarray] | None = None) -> Take:
    """One take of `chunk` from a prompt state (cache is consumed).

    `prime` = (carrier token ids, carrier NORMALIZED latents [K, ldim]):
    the text prefix becomes carrier+chunk and the carrier's latents are
    teacher-forced right after BOS, so generation starts mid-utterance with
    the chunk's first word (continuation priming). Mirrors
    OnnxTts._generate_chunk otherwise (same noise scale, EOS tail, length cap).
    """
    eng = lab.eng
    tokens = eng.sp.encode(chunk, out_type=int)
    ptoks, plat = (prime if prime is not None else ([], np.zeros((0, eng.ldim), np.float32)))
    text_emb = eng.lut[np.asarray(list(ptoks) + list(tokens))][None].astype(np.float32)
    scale = eng.temp ** 0.5
    seq = np.concatenate([np.full((1, eng.ldim), np.nan, np.float32), plat.astype(np.float32)])[None]
    noise = (rng.standard_normal((1, eng.ldim)) * scale).astype(np.float32)
    lat, eos0, cache, off = eng._flow_step(seq, text_emb, off, noise, cache)
    lats = [lat]
    words = len(chunk.split())
    frames_after_eos = (3 if words <= 4 else 1) + 2
    max_gen = int(np.ceil((len(tokens) / eng.tps_est + eng.gen_pad) * eng.frame_rate))
    max_gen = min(max_gen, eng.max_latents - 1 - len(plat))
    eos_step = None
    for step in range(max_gen):
        noise = (rng.standard_normal((1, eng.ldim)) * scale).astype(np.float32)
        lat, eos, cache, off = eng._flow_step(lats[-1].reshape(1, 1, eng.ldim),
                                              np.zeros((1, 0, eng.dim), np.float32), off, noise, cache)
        if bool(eos.reshape(-1)[0]) and eos_step is None:
            eos_step = step
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break
        lats.append(lat)
    z = np.concatenate([x.reshape(1, eng.ldim) for x in lats]).astype(np.float32)
    return Take(chunk=chunk, z=z, eos_step=eos_step, max_gen=max_gen, n_prime=len(plat))


def decode(lab: Lab, take: Take, prime_z: np.ndarray | None = None) -> np.ndarray:
    """Target audio. With a carrier, carrier+target are decoded together (the
    causal decoder needs the context) and the carrier's samples are cut."""
    eng = lab.eng
    if prime_z is not None and len(prime_z):
        full = np.concatenate([prime_z, take.z])
        audio = eng._decode_all([r[None] for r in full])
        audio = audio[len(prime_z) * FRAME:]
    else:
        audio = eng._decode_all([r[None] for r in take.z])
    take.audio = audio.astype(np.float32)
    return take.audio


# ── the engine's own signal QC, replicated (one attempt) ────────────────────
def engine_qc(lab: Lab, audio: np.ndarray, chunk: str) -> tuple[np.ndarray, dict]:
    """-> (trimmed speech as the engine would return it, QC signals).
    Same arithmetic as OnnxTts._generate_with_retry."""
    eng = lab.eng
    tokens = len(eng.sp.encode(chunk, out_type=int))
    expected = tokens / eng.tps_est
    s0, e0 = eng._speech_bounds(audio, tail_keep=eng.tail_keep)
    speech = eng._compress_pauses(audio[s0:e0], max_pause=0.30, keep=0.12)
    dur = len(speech) / eng.sample_rate
    rms = float(np.sqrt((speech ** 2).mean())) if len(speech) else 0.0
    frac = eng._solid_fraction(speech)
    solid = frac * dur
    regions = eng._dead_air_regions(speech)
    dead = max((j - i) for i, j in regions) / eng.sample_rate if regions else 0.0
    loud_ok = rms >= 0.02
    dur_ok = 0.30 * expected <= solid <= 1.6 * expected
    ok = bool(loud_ok and dur_ok and frac >= 0.55 and dead <= 0.35)
    return speech, {"tokens": tokens, "expected_s": expected, "dur_s": dur, "rms": rms,
                    "solid_frac": frac, "solid_s": solid, "dead_s": dead,
                    "loud_ok": bool(loud_ok), "dur_ok": bool(dur_ok), "qc_ok": ok}


# ── ASR-free signals (study A) ─────────────────────────────────────────────
def frame_db(audio: np.ndarray, n: int) -> np.ndarray:
    """Per-latent-frame level in dB (frame k = samples k*FRAME ... )."""
    out = np.full(n, -120.0, np.float32)
    for k in range(n):
        seg = audio[k * FRAME:(k + 1) * FRAME]
        if len(seg):
            out[k] = 20 * np.log10(max(1e-6, float(np.sqrt((seg.astype(np.float64) ** 2).mean()))))
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True) if len(x) > 1 else x
    return x / np.maximum(1e-6, np.linalg.norm(x, axis=1, keepdims=True))


def diag_stripe(S: np.ndarray, valid: np.ndarray, W: int, lag_min: int, self_sim: bool) -> tuple[float, int]:
    """Strongest diagonal stripe of a similarity matrix: max over lags of the
    W-frame running mean along the diagonal, counting only valid pairs.
    -> (score, lag). For a self-similarity matrix lags < lag_min (a frame and
    its neighbours) are skipped."""
    T1, T2 = S.shape
    best, best_lag = 0.0, 0
    lags = range(lag_min, T2) if self_sim else range(-(T1 - 1), T2)
    for d in lags:
        i = np.arange(max(0, -d), min(T1, T2 - d))
        if len(i) < W:
            continue
        v = S[i, i + d]
        m = valid[i, i + d]
        vv = np.where(m, v, 0.0)
        c = np.concatenate(([0.0], np.cumsum(vv)))
        cm = np.concatenate(([0], np.cumsum(m.astype(np.int32))))
        num = c[W:] - c[:-W]
        cnt = cm[W:] - cm[:-W]
        ok = cnt >= max(3, int(0.75 * W))
        if not ok.any():
            continue
        r = np.where(ok, num / np.maximum(1, cnt), -1.0)
        k = float(r.max())
        if k > best:
            best, best_lag = k, int(d)
    return best, best_lag


def loop_score(z: np.ndarray, speech: np.ndarray, W: int = 8, lag_min: int = 6) -> tuple[float, int]:
    """Self-similarity stripe of the take's latents over SPEECH frames: a
    take that says a stretch twice ("the weather in london is cloudy the
    weather in london is …") has two near-identical latent runs -> a bright
    off-diagonal stripe at the repeat lag. W=8 frames = 0.64 s, lag >= 0.48 s."""
    n = len(z)
    if n < W + lag_min:
        return 0.0, 0
    X = _unit(z)
    S = X @ X.T
    m = speech[:n]
    valid = m[:, None] & m[None, :]
    return diag_stripe(S, valid, W, lag_min, self_sim=True)


def leak_score(z: np.ndarray, speech: np.ndarray, prompt_z: np.ndarray, W: int = 6) -> tuple[float, int]:
    """Cross-similarity stripe between the take and its voice prompt: the
    reference-replay failure (the prompt clip's own sentence surfacing in the
    take) is a run of take frames that tracks a run of prompt frames."""
    if len(prompt_z) < W or len(z) < W:
        return 0.0, 0
    mu = np.concatenate([z, prompt_z]).mean(axis=0, keepdims=True)
    A = z - mu
    B = prompt_z - mu
    A = A / np.maximum(1e-6, np.linalg.norm(A, axis=1, keepdims=True))
    B = B / np.maximum(1e-6, np.linalg.norm(B, axis=1, keepdims=True))
    S = A @ B.T
    valid = speech[:len(z), None] & np.ones((1, len(prompt_z)), bool)
    return diag_stripe(S, valid, W, 0, self_sim=False)


def syllable_peaks(audio: np.ndarray, sr: int = 24000) -> int:
    """Syllable nuclei: peaks of the 300-2500 Hz energy envelope (10 ms hop)
    with >= 3 dB prominence, >= 90 ms apart, within 25 dB of the loudest."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    if len(audio) < int(0.05 * sr):
        return 0
    sos = butter(4, [300, 2500], btype="band", fs=sr, output="sos")
    y = sosfiltfilt(sos, audio.astype(np.float64))
    hop, win = int(0.01 * sr), int(0.03 * sr)
    p = y ** 2
    c = np.concatenate(([0.0], np.cumsum(p)))
    starts = np.arange(0, max(1, len(p) - win), hop)
    env = (c[starts + win] - c[starts]) / win
    db = 10 * np.log10(np.maximum(env, 1e-12))
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(db, prominence=3.0, distance=9)
    if not len(peaks):
        return 0
    return int((db[peaks] > db.max() - 25).sum())


def vowel_count(chunk: str) -> int:
    return sum(ch in VOWELS for ch in chunk.replace("1", ""))


def signals(lab: Lab, take: Take, speech_audio: np.ndarray, prompt_raw: np.ndarray) -> dict:
    """Every ASR-free signal of one take."""
    z = take.z
    db = frame_db(take.audio, len(z))
    peak = float(db.max()) if len(db) else -120.0
    speech = db > peak - 30.0
    loop, loop_lag = loop_score(z, speech)
    leak, leak_lag = leak_score(z, speech, lab.to_norm(prompt_raw))
    nv = vowel_count(take.chunk)
    syl = syllable_peaks(speech_audio, lab.sr)
    solid_s = float(speech.sum()) / lab.eng.frame_rate
    return {
        "frames": int(len(z)), "eos_fired": take.eos_step is not None,
        "eos_step": take.eos_step, "max_gen": take.max_gen,
        "cap_hit": take.eos_step is None,
        "loop": round(loop, 4), "loop_lag": loop_lag,
        "leak": round(leak, 4), "leak_lag": leak_lag,
        "vowels": nv, "syl": syl, "syl_ratio": round(syl / max(1, nv), 4),
        "s_per_vowel": round(len(speech_audio) / lab.sr / max(1, nv), 4),
        "speech_frames_s": round(solid_s, 3),
    }


def judge(lab: Lab, audio: np.ndarray, chunk: str, lang: str) -> tuple[str, float]:
    """Ground truth for a take: Whisper (beam 5, digits suppressed) on the
    trimmed speech, consonant-skeleton CER against the chunk's phonemes."""
    if len(audio) < int(0.05 * lab.sr):
        return "", 1.0
    hyp = lab.judge.transcribe(audio, lab.sr, lang)
    return hyp, float(skeleton_cer(chunk, hyp, lang))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
