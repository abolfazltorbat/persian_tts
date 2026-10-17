# Local Persian TTS server: Pocket ONNX plus optional Qwen3-TTS comparison.
#
#   python scripts/server.py     -> http://127.0.0.1:8770
#
# Endpoints:
#   GET  /                  the single-page UI (web/index.html)
#   GET  /api/voices        available reference voices (builtin + uploaded)
#   POST /api/tts           {text, voice, model?} -> audio metadata
#   POST /api/jobs          asynchronous single-model or comparison run
#   GET  /api/jobs/{id}     progress and per-model results/errors
#   GET  /api/audio/{id}    generated WAV
#   POST /api/voice/upload  upload a custom voice (auto-trimmed to 5 s)
import io
import json
import os
import re
import sys
import threading
import time
import uuid
import copy
import hashlib
import logging
from typing import Literal
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE))
from persian_text import prepare_persian, PERSIAN_CASES
from qwen_backend import QwenBackend, QwenUnavailable

WEB = BASE / "web" / "index.html"
UPLOAD_DIR = BASE / "uploads" / "voices"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_TEXT = 800          # keep demo requests bounded
MAX_PHONEMES = 2400     # editable manual-phoneme input cap
MAX_VOICES = 64

BUILTIN_VOICE_META = {
    "male_hello.wav": ("آقا · صمیمی", "صدای مرد، لحن آرام و دوستانه"),
    "male_hello_clean.wav": ("آقا · صمیمی پاک‌سازی‌شده", "نسخه بدون جهش صدای ابتدای کلیپ"),
    "female_narration_clean.wav": ("بانو · روایت (پاک‌سازی‌شده)",
                                   "همان صدا، بدون جملهٔ خودِ نمونه که در پاسخ‌ها تکرار می‌شد"),
    "female_narration.wav": ("بانو · روایت", "صدای زن، روایت‌گر"),
    "male_news.wav": ("آقا · خبری", "صدای مرد، لحن خبرگزاری"),
}

app = FastAPI(title="پارسی‌گو — Persian TTS demo")
app.mount("/fonts", StaticFiles(directory=BASE / "web" / "fonts"), name="fonts")
app.mount("/assets", StaticFiles(directory=BASE / "web"), name="assets")

_engine = None
_engine_lock = threading.RLock()
_qwen = QwenBackend()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_running = False
_audio_store: dict[str, dict] = {}
_store_lock = threading.Lock()

# punctuation-aware phrase splitting and the pause lengths live with the
# engine (single source of truth for where pauses fall and how long they
# last): PAUSE_S is the sentence-final pause — a real reader stops at a
# period, not just breathes (comma is 0.16 in the engine)
from tts_onnx import PKG, SENTENCE_GAP as PAUSE_S, plan_phrases, pack_phrases  # noqa: E402

# the model's audio format, from the same manifest key the engine reads — a
# future model with a different rate flows through the resample/pause math
# here instead of silently writing wrong-rate WAVs. Deliberately read from
# the static manifest, NOT from get_engine(), so a voice upload never
# triggers a full engine load.
SR: int = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))["constants"]["sample_rate"]


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from tts_onnx import OnnxTts
            from g2p_onnx import OnnxG2P
            from speak import _MixedG2P
            engine = OnnxTts()
            engine._g2p = _MixedG2P(OnnxG2P(engine.dir))
            _engine = engine
    return _engine


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?؟])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def list_voices() -> list[dict]:
    out = []
    for f in sorted((BASE / "voices").glob("*.wav")):
        name, desc = BUILTIN_VOICE_META.get(f.name, (f.stem, ""))
        out.append({"id": f.name, "name": name, "desc": desc, "builtin": True})
    for f in sorted(UPLOAD_DIR.glob("*.wav")):
        out.append({
            "id": f"upload:{f.name}", "name": f.name[:22],
            "desc": "صدای بارگذاری‌شده", "builtin": False,
        })
    return out[:MAX_VOICES]


def voice_path(voice_id: str) -> Path:
    if voice_id.startswith("upload:"):
        p = UPLOAD_DIR / Path(voice_id[7:]).name
    else:
        p = BASE / "voices" / Path(voice_id).name
    if not p.is_file():
        raise HTTPException(404, "voice not found")
    return p


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    voice: str
    pace: float = Field(default=1.0, ge=0.6, le=1.5, allow_inf_nan=False)
    mode: Literal["split", "pack"] = "split"
    model: Literal["pocket", "qwen"] = "pocket"
    prepare: bool = True
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    ref_text: str = Field(default="", max_length=800)


class JobRequest(TTSRequest):
    model: Literal["pocket", "qwen", "compare"] = "compare"
    case_id: str = Field(default="custom", max_length=40)


class PhonemizeRequest(BaseModel):
    text: str
    mode: str = "split"
    prepare: bool = True


class PhonemeTTSRequest(BaseModel):
    phonemes: str
    voice: str
    pace: float = 1.0


@app.get("/")
def index():
    return FileResponse(WEB)


@app.get("/api/voices")
def voices():
    return {"voices": list_voices()}


@app.get("/api/config")
def config():
    """UI bootstrap values — the text cap lives here, not hardcoded in the HTML."""
    return {"max_text": MAX_TEXT, "max_phonemes": MAX_PHONEMES,
            "models": [{"id": "pocket", "name": "Pocket TTS Farsi v2", "ready": True,
                        "persian_supported": True}, _qwen.status()], "cases": PERSIAN_CASES}


@app.post("/api/phonemize")
def phonemize(req: PhonemizeRequest):
    """Convert Persian text to editable phonemes without generating audio."""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "متن خالی است")
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"متن طولانی است (حداکثر {MAX_TEXT} نویسه)")
    if req.mode not in ("split", "pack"):
        raise HTTPException(400, "حالت گفتار باید split یا pack باشد")
    engine = get_engine()
    phonemes_all = []
    if req.prepare:
        # the same front-end synthesis uses (render_pocket), so the editable
        # phonemes are exactly what would have been spoken
        import frontend
        with _engine_lock:
            try:
                _, plans = frontend.plan(engine, text, pack=req.mode == "pack")
            except ValueError as e:
                raise HTTPException(400, "متن فارسی معتبری پیدا نشد") from e
        for plan in plans:
            phonemes_all.extend(p.replace("1", "") for p, _ in plan)
    else:
        with _engine_lock:
            for sent in split_sentences(text):
                try:
                    plan = plan_phrases(sent, engine._g2p, engine.sp)
                except ValueError as e:
                    raise HTTPException(400, "متن فارسی معتبری پیدا نشد") from e
                if req.mode == "pack":
                    plan = pack_phrases(plan)
                phonemes_all.extend(p.replace("1", "") for p, _ in plan)

    phonemes = " ".join(phonemes_all).strip()
    if not phonemes:
        raise HTTPException(400, "فونمی تولید نشد")
    return {"phonemes": phonemes}


@app.post("/api/tts-phonemes")
def tts_phonemes(req: PhonemeTTSRequest):
    """Generate speech directly from a user-edited phoneme string."""
    phonemes = req.phonemes.strip()
    if not phonemes:
        raise HTTPException(400, "فونم خالی است")
    if len(phonemes) > MAX_PHONEMES:
        raise HTTPException(400, f"فونم طولانی است (حداکثر {MAX_PHONEMES} نویسه)")
    if any("\u0600" <= ch <= "\u06FF" for ch in phonemes):
        raise HTTPException(400, "در کادر فونم فقط فونم لاتین وارد کنید؛ متن فارسی را در کادر متن بنویسید")

    engine = get_engine()
    pace = float(min(max(req.pace, 0.6), 1.5))
    with _engine_lock:
        audio = engine.synthesize(phonemes, voice_path(req.voice), pace=pace)

    if len(audio) == 0:
        raise HTTPException(400, "صوتی تولید نشد")
    duration = len(audio) / SR
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    audio_id = uuid.uuid4().hex[:12]
    with _store_lock:
        _audio_store[audio_id] = {
            "wav": buf.getvalue(),
            "text": phonemes,
            "voice": req.voice,
            "duration": duration,
            "phonemes": phonemes,
        }
        while len(_audio_store) > 128:
            _audio_store.pop(next(iter(_audio_store)))
    return {
        "id": audio_id,
        "phonemes": phonemes,
        "duration": round(duration, 2),
        "pace": pace,
        "mode": "manual-phonemes",
        "sentences": 1,
        "elapsed": None,
    }


def render_pocket(req, text):
    start = time.perf_counter()
    engine = get_engine()
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pace = req.pace
    phonemes_all, chunks = [], []

    if req.prepare:
        # The chatbot's path (scripts/frontend.py): dates, units, currency,
        # ranges and English numbers read as words in the sentence's own
        # language, then every unit of every sentence in one parallel batch.
        # It reads the RAW text — prepare_persian already spelled the digits,
        # and «۱۴۰۳/۰۵/۲۰» spelled digit-run by digit-run is not a date.
        import frontend
        with _engine_lock:
            try:
                sents, plans = frontend.plan(engine, req.text, pack=req.mode == "pack")
            except ValueError as e:
                raise HTTPException(400, "متن فارسی معتبری پیدا نشد") from e
            if not plans:
                raise HTTPException(400, "متنی برای ساخت صدا پیدا نشد")
            audios = engine.synthesize_many(plans, voice_path(req.voice), pace=pace,
                                            langs=[x.lang for x in sents], seed=req.seed)
        for plan, audio in zip(plans, audios):
            phonemes_all.append(" ".join(p.replace("1", "") for p, _ in plan))
            chunks += [audio, np.zeros(int(PAUSE_S / pace * SR), dtype=audio.dtype)]
    else:
        with _engine_lock:
            engine.rng = np.random.default_rng(req.seed)
            for sent in split_sentences(text):
                # punctuation-aware plan: each phrase (comma/dash/colon-delimited)
                # is a pause unit with its own gap, and short lead-ins ending in
                # strong punctuation ("سؤال اصلی:") stay standalone
                try:
                    plan = plan_phrases(sent, engine._g2p, engine.sp)
                except ValueError as e:
                    raise HTTPException(400, "متن فارسی معتبری پیدا نشد") from e
                if req.mode == "pack":
                    plan = pack_phrases(plan)
                if not plan:
                    continue
                # model card: retry a runaway once (stochastic; 2nd attempt usually ends)
                tokens = sum(len(engine.sp.encode(p.replace("1", ""), out_type=int))
                             for p, _ in plan)
                cap = tokens / engine.tps_est + engine.gen_pad + 1
                for attempt in range(2):
                    audio = engine.synthesize(plan, voice_path(req.voice), pace=pace)
                    if len(audio) / SR <= cap + 2.0:  # multi-chunk texts run longer
                        break
                phonemes_all.append(" ".join(p.replace("1", "") for p, _ in plan))
                chunks.append(audio)
                chunks.append(np.zeros(int(PAUSE_S / pace * SR), dtype=audio.dtype))

    if not chunks:
        raise HTTPException(400, "متنی برای ساخت صدا پیدا نشد")
    audio = np.concatenate(chunks[:-1])  # drop trailing pause
    return audio, SR, {
        "phonemes": " ".join(phonemes_all),
        "sentences": len(phonemes_all),
        "model_name": "pocket-tts-farsi-v2 (ONNX)", "device": "cpu",
        "persian_supported": True, "warnings": [],
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(time.perf_counter() - start, 3),
    }


def validate_text(req):
    text = prepare_persian(req.text, req.prepare)
    if not any(c.isalnum() for c in text):
        raise HTTPException(400, "متن باید حداقل یک حرف یا عدد داشته باشد")
    # Number expansion can multiply the raw request length substantially.
    if len(text) > 6000:
        raise HTTPException(400, "متن پس از آماده‌سازی بسیار طولانی است")
    voice_path(req.voice)
    return text


def reference_audio(voice):
    path = voice_path(voice)
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        wav = f.read(5 * sr, dtype="float32", always_2d=True).mean(axis=1)
    if len(wav) < sr or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
        raise HTTPException(400, "صدای مرجع باید حداقل یک ثانیه و شامل گفتار باشد")
    return wav, sr


def match_level(audio, sr):
    """Speech-frame RMS matching, not a perceptual quality score or denoising."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("صوت معتبر تولید نشد")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        raise ValueError("مدل فقط سکوت تولید کرد")
    hop = max(1, int(sr * .02))
    powers = np.array([np.mean(audio[i:i + hop] ** 2) for i in range(0, len(audio), hop)])
    active = powers[powers > max(float(powers.max()) * .01, 1e-10)]
    rms = float(np.sqrt(active.mean()))
    gain = min(10 ** (-23 / 20) / max(rms, 1e-8), .95 / peak, 10.0)
    return audio * gain, round(20 * np.log10(gain), 3)


def generate(req, model, progress=None):
    text = validate_text(req)
    reference, reference_sr = reference_audio(req.voice)
    with _engine_lock:
        if model == "pocket":
            wav, sr, meta = render_pocket(req, text)
        else:
            wav, sr, meta = _qwen.synthesize(text, reference, reference_sr,
                ref_text=req.ref_text, seed=req.seed, pace=req.pace, progress=progress)
    wav, gain_db = match_level(wav, sr)
    duration = len(wav) / sr
    audio_id = uuid.uuid4().hex[:12]
    result = {"id": audio_id, "model": model, "text": req.text, "prepared_text": text,
              "voice": req.voice, "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
              "reference_seconds": round(len(reference) / reference_sr, 3),
              "sample_rate": sr, "duration": round(duration, 3), "pace": req.pace,
              "mode": req.mode if model == "pocket" else "sentence chunks",
              "seed": req.seed, "prepare": req.prepare, "phonemes": "", **meta,
              "level_gain_db": gain_db, "audio_url": f"/api/audio/{audio_id}"}
    result["rtf"] = round(meta["generation_seconds"] / duration, 3)
    result["elapsed"] = round(meta["load_seconds"] + meta["generation_seconds"], 3)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    with _store_lock:
        _audio_store[audio_id] = {"wav": buf.getvalue(), **result}
        while len(_audio_store) > 128:
            _audio_store.pop(next(iter(_audio_store)))
    return result


@app.post("/api/tts")
def tts(req: TTSRequest):
    try:
        return generate(req, req.model)
    except QwenUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def run_job(job_id, req):
    global _job_running
    def progress(message):
        with _jobs_lock:
            _jobs[job_id]["progress"] = message
    try:
        for model in (["pocket", "qwen"] if req.model == "compare" else [req.model]):
            progress(f"در حال ساخت با {model}…")
            try:
                result = generate(req, model, progress)
            except Exception as exc:
                logging.exception("TTS job %s failed for %s", job_id, model)
                result = {"model": model, "error": str(exc.detail) if isinstance(exc, HTTPException) else str(exc)}
            with _jobs_lock:
                _jobs[job_id]["results"].append(result)
        with _jobs_lock:
            results = _jobs[job_id]["results"]
            _jobs[job_id]["status"] = "complete" if all("error" not in r for r in results) else "partial" if any("id" in r for r in results) else "failed"
            _jobs[job_id]["progress"] = "پایان پردازش"
    finally:
        with _jobs_lock:
            _job_running = False


@app.post("/api/jobs", status_code=202)
def start_job(req: JobRequest):
    global _job_running
    prepared = validate_text(req)
    reference_audio(req.voice)
    with _jobs_lock:
        if _job_running:
            raise HTTPException(409, "یک تولید دیگر در حال اجرا است؛ تا پایان آن صبر کنید")
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {"id": job_id, "status": "running", "progress": "در صف تولید",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request": req.model_dump(), "prepared_text": prepared, "results": [],
            "evaluation": "Human listening ratings; RTF measures speed, not Persian quality.",
            "comparison_notes": ["Both models receive the same prepared text and source reference (first 5 seconds).",
                "Pocket applies its trained G2P, reference cleanup and synthesis retries; Qwen receives Persian script and Auto language.",
                "Outputs use the same speech-frame RMS target with peak protection; this is not LUFS normalization.",
                "The same seed identifies each run; different architectures do not share equivalent random samples.",
                "Qwen reference transcript, when supplied, is extra conditioning unavailable to Pocket."]}
        _job_running = True
        while len(_jobs) > 64:
            _jobs.pop(next(iter(_jobs)))
    threading.Thread(target=run_job, args=(job_id, req), daemon=True).start()
    return {"id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "آزمایش پیدا نشد")
        return copy.deepcopy(job)


@app.get("/api/reference/{voice_id}")
def reference_preview(voice_id: str):
    wav, sr = reference_audio(voice_id)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.get("/api/audio/{audio_id}")
def audio(audio_id: str):
    with _store_lock:
        item = _audio_store.get(audio_id)
    if item is None:
        raise HTTPException(404, "not found")
    return Response(content=item["wav"], media_type="audio/wav")


@app.post("/api/voice/upload")
async def upload_voice(file: UploadFile = File(...)):
    import scipy.signal as sig

    data = await file.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "حداکثر اندازه فایل ۲۰ مگابایت است")
    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "فایل صوتی خوانده نشد — WAV پیشنهاد می‌شود")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        wav = sig.resample_poly(wav, SR // g, sr // g).astype(np.float32)
    if len(wav) < SR:  # < 1 s is too short to clone
        raise HTTPException(400, "صدای مرجع باید حداقل ۱ ثانیه باشد")
    wav = wav[: 5 * SR]  # model card: prompts beyond 5 s are out of distribution
    if not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
        raise HTTPException(400, "صدای مرجع خالی یا نامعتبر است")

    vid = f"upload:{uuid.uuid4().hex[:8]}.wav"
    sf.write(UPLOAD_DIR / vid[7:], wav, SR, format="WAV")
    return {"id": vid, "name": vid[7:], "seconds": round(len(wav) / SR, 1)}


if __name__ == "__main__":
    # defaults keep the demo loopback-only; 0.0.0.0 exposes an
    # unauthenticated demo (disk-writing upload included) — opt in
    host = os.environ.get("PARSIGO_HOST", "127.0.0.1")
    # NOT 8000. That port belongs to the vLLM OpenAI-compatible gateway on the
    # machines this runs on, and an inference gateway losing the bind (or, worse,
    # winning it and answering /v1/chat/completions with this app's 404) takes
    # the whole product down to save a demo. 8770 is also what the chatbot's
    # TTS_PERSIAN_URL defaults to, so the two sides agree without any env var.
    port = int(os.environ.get("PARSIGO_PORT", "8770"))
    print("loading engine (first request may take a moment)...")
    get_engine()
    print(f"demo:  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
