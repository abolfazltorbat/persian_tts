"""Regression checks for Persian preparation and comparison isolation.

These tests do not download models. Actual audio smoke tests are separate.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from persian_text import prepare_persian, text_chunks
import server
from qwen_backend import QwenBackend, QwenUnavailable


@pytest.mark.parametrize("text, expected", [
    ("قيمت كتاب ۱٬۲۵۰٬۰۰۰ تومان", "قیمت کتاب یک میلیون و دویست و پنجاه هزار تومان"),
    ("ساعت ۱۴:۳۰ و ۰۸:۰۰", "ساعت چهارده و سی دقیقه و هشت"),
    ("دما −۲٫۵ و تخفیف ۱۲٫۵٪", "دما منفی دو ممیز پنج دهم و تخفیف دوازده ممیز پنج دهم درصد"),
    ("شماره ۰۹۱۲", "شماره صفر نه یک دو"),
    ("كتابِ خانهٔ ما می‌رود؛ Python!", "کتابِ خانهٔ ما می‌رود؛ Python!"),
    ("سلام\u200f\nخوبی?", "سلام\nخوبی؟"),
])
def test_persian_preparation(text, expected):
    result = prepare_persian(text)
    assert result == expected
    assert prepare_persian(result) == result
    assert prepare_persian(text, False) == text.strip()


def test_chunking_keeps_content_and_questions():
    text = "سلام، حال شما چطور است؟\n" + "این یک جمله برای ارزیابی است. " * 20
    chunks = text_chunks(text, max_chars=80)
    assert chunks[0].endswith("؟")
    assert " ".join(chunks).split() == text.split()
    assert all(len(c) <= 80 for c in chunks)
    assert text_chunks("... ؟") == []


def test_mixed_g2p_does_not_send_arabic_punctuation_to_model(monkeypatch):
    from speak import _MixedG2P
    import en_phonemes
    monkeypatch.setattr(en_phonemes, "split", lambda _: [("en", "pAyton"), ("fa", "، "), ("en", "linuks")])
    class Inner:
        def phonemise(self, *a, **k):
            pytest.fail("Punctuation-only segments must not reach G2P")
    assert _MixedG2P(Inner()).phonemise("Python، Linux") == "pAyton linuks"


@pytest.fixture
def client():
    # Starlette TestClient can depend on external loop setup; ASGI transport
    # exercises the real routes without opening a network port.
    import httpx
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")


def request(client, method, path, **kwargs):
    import asyncio
    return asyncio.run(client.request(method, path, **kwargs))


def test_config_and_old_default_contract(client):
    r = request(client, "GET", "/api/config")
    assert r.status_code == 200
    assert len(r.json()["cases"]) >= 6
    assert r.json()["models"][1]["persian_supported"] is False
    assert server.TTSRequest(text="سلام", voice="female_narration.wav").model == "pocket"


@pytest.mark.parametrize("extra", [{"model": "invalid"}, {"pace": 0}, {"seed": -1}, {"text": "x" * 801}])
def test_invalid_requests_never_start_generation(client, extra):
    payload = {"text": "سلام", "voice": "female_narration.wav", **extra}
    assert request(client, "POST", "/api/jobs", json=payload).status_code == 422


def test_missing_qwen_has_actionable_error_without_loading_pocket(client, monkeypatch):
    def fail(*args, **kwargs):
        raise QwenUnavailable("install requirements-qwen.txt")
    monkeypatch.setattr(server._qwen, "synthesize", fail)
    r = request(client, "POST", "/api/tts", json={"text": "سلام", "voice": "female_narration.wav", "model": "qwen"})
    assert r.status_code == 503
    assert "requirements-qwen" in r.json()["detail"]


def test_comparison_preserves_success_when_other_model_fails(monkeypatch):
    calls = []
    def generate(req, model, progress):
        calls.append((model, req.text, req.voice, req.seed))
        if model == "qwen":
            raise QwenUnavailable("not installed")
        return {"id": "audio", "model": model}
    monkeypatch.setattr(server, "generate", generate)
    monkeypatch.setattr(server, "_jobs", {"test": {"results": []}})
    monkeypatch.setattr(server, "_job_running", True)
    req = server.JobRequest(text="سلام", voice="female_narration.wav", seed=73)
    server.run_job("test", req)
    assert server._jobs["test"]["status"] == "partial"
    assert server._jobs["test"]["results"][0]["id"] == "audio"
    assert "error" in server._jobs["test"]["results"][1]
    assert calls[0][1:] == calls[1][1:]
    assert server._job_running is False


def test_busy_and_empty_requests(client, monkeypatch):
    monkeypatch.setattr(server, "_job_running", True)
    payload = {"text": "سلام", "voice": "female_narration.wav"}
    assert request(client, "POST", "/api/jobs", json=payload).status_code == 409
    assert request(client, "POST", "/api/jobs", json={**payload, "text": " ؟ "}).status_code == 400


def test_output_uses_backend_sample_rate_and_keeps_metrics(client, monkeypatch):
    sr = 16000
    wav = (.1 * np.sin(np.arange(sr) * .1)).astype(np.float32)
    meta = {"model_name": "test-qwen", "generation_seconds": 2.0, "load_seconds": 3.0}
    monkeypatch.setattr(server._qwen, "synthesize", lambda *a, **k: (wav, sr, meta))
    payload = {"text": "عدد ۱۲", "voice": "female_narration.wav", "model": "qwen"}
    response = request(client, "POST", "/api/tts", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rtf"] == 2.0 and result["elapsed"] == 5.0
    assert result["prepared_text"] == "عدد دوازده"
    data = request(client, "GET", result["audio_url"]).content
    samples, rate = sf.read(io.BytesIO(data))
    assert rate == sr and len(samples) == sr
    assert np.max(np.abs(samples)) <= .95


def test_level_matching_rejects_invalid_audio():
    for wav in [np.zeros(10), np.array([np.nan]), np.array([])]:
        with pytest.raises(ValueError):
            server.match_level(wav, 24000)


def test_qwen_calls_use_script_auto_language_and_exact_reference():
    pytest.importorskip("torch")
    calls, prompts = [], []
    class FakeModel:
        def create_voice_clone_prompt(self, **kwargs):
            prompts.append(kwargs)
            return "prompt"
        def generate_voice_clone(self, **kwargs):
            calls.append(kwargs)
            return [np.full(2400, .1, np.float32)], 24000
    backend = QwenBackend()
    backend.model, backend.device = FakeModel(), "cpu"
    reference = np.ones(24000, np.float32) * .1
    for transcript in ["", "سلام من هستم"]:
        audio, sr, meta = backend.synthesize("سلام. حال شما چطور است؟", reference, 24000, ref_text=transcript)
        assert sr == 24000 and len(audio) > 4800
        assert prompts[-1]["x_vector_only_mode"] is (not bool(transcript))
        assert prompts[-1]["ref_text"] == (transcript or None)
        assert prompts[-1]["ref_audio"][0] is reference
        assert meta["persian_supported"] is False
    assert all(c["language"] == "Auto" for c in calls)
    assert calls[0]["text"] == "سلام."
    assert all(c["max_new_tokens"] <= 1200 for c in calls)
