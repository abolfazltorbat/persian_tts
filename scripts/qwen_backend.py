"""Optional local Qwen3-TTS Base backend. No torch import on the ONNX path."""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from persian_text import text_chunks

BASE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BASE / "model" / "qwen3-tts-0.6b"
PERSIAN_WARNING = "فارسی در زبان‌های رسمی Qwen3-TTS نیست؛ تولید با Auto آزمایشی است و ممکن است تلفظ یا زبان اشتباه باشد."


class QwenUnavailable(RuntimeError):
    pass


class QwenBackend:
    def __init__(self):
        self.model = None
        self.device = None
        self.model_path = Path(os.environ.get("PARSIGO_QWEN_MODEL", str(DEFAULT_MODEL))).expanduser().resolve()

    def status(self):
        installed = importlib.util.find_spec("qwen_tts") is not None
        tokenizer = self.model_path / "speech_tokenizer"
        weights = ((self.model_path / "config.json").is_file()
                   and any(self.model_path.glob("*.safetensors"))
                   and (tokenizer / "config.json").is_file()
                   and any(tokenizer.glob("*.safetensors")))
        return {"id": "qwen", "name": "Qwen3-TTS Base", "installed": installed,
                "downloaded": weights, "ready": installed and weights,
                "loaded": self.model is not None, "device": self.device,
                "model_path": str(self.model_path), "persian_supported": False,
                "warning": PERSIAN_WARNING,
                "setup": "docs/comparison.md"}

    def load(self):
        if self.model is not None:
            return 0.0
        status = self.status()
        if not status["ready"]:
            raise QwenUnavailable("Qwen آماده نیست؛ نصب و دریافت مدل: docs/comparison.md. سرور را با .venv-qwen/bin/python اجرا کنید.")
        start = time.perf_counter()
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            self.device = os.environ.get("PARSIGO_QWEN_DEVICE", "auto")
            if self.device == "auto":
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                torch.set_num_threads(int(os.environ.get("PARSIGO_QWEN_THREADS", "4")))
            dtype = torch.float32
            if self.device.startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = Qwen3TTSModel.from_pretrained(
                str(self.model_path), device_map=self.device, dtype=dtype,
                attn_implementation="sdpa", local_files_only=True,
            )
            if self.model.model.tts_model_type != "base":
                self.model = None
                raise QwenUnavailable("برای کلون صدا از مدل Base استفاده کنید؛ CustomVoice و VoiceDesign مناسب این مقایسه نیستند.")
        except QwenUnavailable:
            raise
        except Exception as exc:
            raise QwenUnavailable(f"بارگذاری Qwen ناموفق بود: {exc}") from exc
        return time.perf_counter() - start

    def synthesize(self, text, reference, reference_sr, *, ref_text="", seed=42, pace=1.0, progress=None):
        load_seconds = self.load()
        import torch
        start = time.perf_counter()
        # Calls are serialized by the server lock. Restore RNG state afterward.
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        chunks = text_chunks(text)
        if not chunks:
            raise ValueError("متنی برای ساخت صدا پیدا نشد")
        parts, sr, limit_warnings = [], None, []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=(reference, reference_sr), ref_text=ref_text.strip() or None,
                x_vector_only_mode=not bool(ref_text.strip()),
            )
            for i, chunk in enumerate(chunks):
                if progress:
                    progress(f"Qwen: بخش {i + 1} از {len(chunks)} (روی CPU ممکن است چند دقیقه طول بکشد)")
                token_cap = min(1200, max(180, len(chunk) * 5))
                waves, sr = self.model.generate_voice_clone(
                    text=chunk, language="Auto", voice_clone_prompt=prompt,
                    do_sample=True, top_k=50, top_p=0.9, temperature=0.7,
                    repetition_penalty=1.05, subtalker_dosample=True,
                    subtalker_top_k=50, subtalker_top_p=0.9, subtalker_temperature=0.7,
                    max_new_tokens=token_cap,
                )
                wav = np.asarray(waves[0], dtype=np.float32).reshape(-1)
                if not len(wav) or not np.isfinite(wav).all():
                    raise RuntimeError("Qwen صوت معتبر تولید نکرد")
                if len(wav) / sr >= (token_cap - 2) / 12:
                    limit_warnings.append(f"بخش {i + 1} به سقف طول تولید نزدیک شد؛ پایان جمله را از نظر قطع شدن یا تکرار بررسی کنید.")
                if parts:
                    gap = 0.45 if chunks[i - 1][-1:] in ".!?؟" else 0.16
                    parts.append(np.zeros(int(gap * sr), np.float32))
                parts.append(wav)
        audio = np.concatenate(parts)
        if pace != 1.0:
            from pedalboard import time_stretch
            audio = time_stretch(audio, sr, stretch_factor=pace).reshape(-1)
        provenance = self.model_path / "download.json"
        return audio, sr, {
            "model_name": self.model_path.name, "device": self.device,
            "language": "Auto", "persian_supported": False,
            "reference_mode": "transcript + audio" if ref_text.strip() else "speaker embedding only",
            "sentences": len(chunks), "chunks": chunks, "load_seconds": round(load_seconds, 3),
            "generation_seconds": round(time.perf_counter() - start, 3),
            "warnings": [PERSIAN_WARNING] + limit_warnings + ([] if ref_text.strip() else ["متن دقیق صدای مرجع می‌تواند به تقلید تلفظ کمک کند؛ فقط متن همان کلیپ را وارد کنید."]),
            "provenance": json.loads(provenance.read_text()) if provenance.is_file() else None,
        }
