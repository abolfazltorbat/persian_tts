# Local Persian comparison: Pocket TTS and Qwen3-TTS

The web app has three modes: the existing **Pocket TTS Farsi v2**, **Qwen3-TTS**, and **both models**. Existing `/api/tts` clients still default to Pocket.

**Persian is not an officially supported Qwen3-TTS language.** The upstream model lists Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian. This integration sends Persian script with `language="Auto"`; it does not disguise Persian as Arabic or English or claim fine-tuning. Accent drift, wrong-language speech, missing words and failed generations are possible. Text preparation cannot add language support to the model.

Upstream: [Qwen3-TTS code and supported models](https://github.com/QwenLM/Qwen3-TTS), [0.6B Base model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base), [1.7B Base model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

## Install and run

Keep Qwen's torch/transformers runtime separate from the original lightweight ONNX environment. Commands run from this repository directory:

```bash
python3 -m venv .venv-qwen
# CPU runtime. For NVIDIA, install an appropriate torch/torchaudio pair instead.
.venv-qwen/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt
.venv-qwen/bin/python scripts/download_qwen.py --size 0.6
.venv-qwen/bin/python scripts/server.py
```

Open **http://127.0.0.1:8770**. The model is loaded lazily on the first Qwen request. The download script records the resolved model commit in `model/qwen3-tts-0.6b/download.json`. It includes the speech tokenizer shipped in that repository. First-time installation/download needs internet; generation uses local model files. No hosted TTS or ASR API is used.

The default 0.6B Base checkpoint is the practical CPU option, not a claim of best quality. CPU synthesis may take minutes. To compare the larger checkpoint:

```bash
.venv-qwen/bin/python scripts/download_qwen.py --size 1.7
PARSIGO_QWEN_MODEL="$PWD/model/qwen3-tts-1.7b" .venv-qwen/bin/python scripts/server.py
```

Configuration:

| Variable | Default | Purpose |
|---|---|---|
| `PARSIGO_QWEN_MODEL` | `model/qwen3-tts-0.6b` | Local Base checkpoint directory; restart to switch |
| `PARSIGO_QWEN_DEVICE` | `auto` | CUDA when available, otherwise CPU; can force `cpu` / `cuda:0` |
| `PARSIGO_QWEN_THREADS` | `4` | CPU torch thread count |
| `PARSIGO_HOST` / `PARSIGO_PORT` | `127.0.0.1` / `8770` | Existing server binding |

The Pocket model files from the original installation must still be present. Qwen is optional: missing dependencies or weights are reported on its own card, and a successful Pocket result remains available. Run one server worker: models, jobs and WAVs are in process memory.

## Evaluate Persian

1. Choose **مقایسهٔ هر دو مدل** and a reference voice. `female_narration.wav` is the default; `male_hello_clean.wav` is preferable to the original male greeting with its loud onset.
2. Play the reference clip. Both paths use the first five seconds of the same source recording; Pocket retains its existing onset cleanup. Use clean Persian speech from one speaker, preferably a complete sentence. Avoid music, clipping and long silence.
3. Optionally enter the **exact transcript of the previewed clip** for Qwen. This enables transcript-and-audio conditioning; without it Qwen uses the speaker embedding only. A mismatched transcript can make output worse. The transcript is extra information provided only to Qwen, and the report records it. Use an empty transcript when testing that baseline.
4. Select a test case: everyday speech, ezafe, numbers/time/percentages, Persian names, half-spaces, questions, mixed English or signed numbers. Edit ambiguous words as needed. Automatic vowel guessing is deliberately avoided; Pocket's manual phoneme editor remains available in Pocket-only mode.
5. Generate. The background job reports progress; refresh can reconnect to an active job. Different requests are serialized, and a second background job receives HTTP 409.
6. Listen using the same headphones/volume. Rate pronunciation, intelligibility/completeness, naturalness/prosody and reference-voice similarity from 1 to 5. Leave unjudged criteria blank. Note wrong words and accent changes. These are **your listening ratings, not automatically measured MOS**.
7. Reveal model names and performance, download both WAVs, and export the JSON report **before the next run**. The most recent report is also saved in browser local storage. The report includes errors, original/prepared text, reference hash, seed, preparation setting, model/device, Qwen revision, timings and ratings.

Repeat across test categories and several seeds (for example 42, 43, 44). The same numerical seed does not mean equivalent samples across architectures. One clip is not enough to rank Persian quality. Hidden names reduce name bias but do not make this a controlled blind study; differing failures, speed and accents may still identify models.

`generation_seconds` includes frontend preparation specific to each model, reference conditioning and synthesis, excluding model load and queue time. `load_seconds` is shown separately. `RTF = generation_seconds / audio_duration`: below 1 is faster than real time. Speed is not a pronunciation score. Outputs retain their native sample rates and are matched to the same speech-frame RMS target with peak protection, **not LUFS normalization**. Qwen uses sentence/phrase chunks; the split/pack control is Pocket-specific and is recorded accordingly.

Generated WAVs (last 128) and jobs (last 64) are held in server memory and disappear on restart. Download WAVs separately: JSON stores metadata and audio URLs, not audio bytes. Text/transcripts/ratings in downloaded JSON and browser storage remain on your machine.

## Persian preparation

Enabled by default and switchable for before/after testing:

- Fold Arabic kaf/yeh to Persian forms, remove invisible direction controls and tatweel, preserve meaningful half-spaces, diacritics and Latin words.
- Expand Persian/Arabic/ASCII digits, grouped thousands, decimals, percentages, leading-zero identifiers and signed numbers before sentence/phrase splitting.
- Read `۱۴:۳۰` as `چهارده و سی دقیقه`; dates, equations and ambiguous abbreviations may still need explicit spoken wording.
- Keep paragraph pauses and native punctuation. Qwen gets Persian **text**, never Pocket's Latin phonemes.
- Pocket now shares the mixed Persian/English G2P wrapper already used by `speak.py`, preserving English vowels through CMUdict when available and falling back to Persian transliteration otherwise.
- Existing Pocket ezafe-aware planning, token limits, reference cleanup and synthesis retries remain enabled. Qwen uses bounded chunks, conservative sampling, a token cap and optional exact-reference conditioning. No automatic retry cherry-picks a supposedly better Qwen pronunciation.

Disabling shared preparation does not disable Pocket's existing model-required normalizer. Reports compare complete usable pipelines rather than raw neural network weights. Improvements in text handling are tested; improvements in perceived audio quality require listening evaluation.

## API and checks

```bash
curl -X POST http://127.0.0.1:8770/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"text":"سلام، حال شما چطور است؟","voice":"female_narration.wav","model":"compare","prepare":true,"seed":42,"pace":1,"ref_text":""}'
# Poll GET /api/jobs/{id}; results include /api/audio/{id} URLs.
```

`POST /api/tts` accepts `model: "pocket" | "qwen"` for synchronous callers. `POST /api/jobs` also accepts `"compare"`. Configuration, model availability and test cases are exposed at `GET /api/config`. Reference playback is `GET /api/reference/{voice_id}`. No user-provided filesystem/model paths are accepted by generation endpoints.

```bash
python -m pytest tests/test_comparison.py -q
```

The regression suite checks normalization, content-preserving chunking, API validation, Qwen call arguments, sample rates, optional-dependency errors, job serialization and preservation of the successful result when the other model fails. It uses stubbed synthesis for API tests, so it does not substitute for real-model audio testing.

### Local smoke run (2026-09-29)

Both real models generated WAVs through the comparison UI using `سلام، حال شما چطور است؟`, `female_narration.wav`, seed 42, preparation enabled and pace 1. Qwen used the 0.6B Base checkpoint at commit `5d83992436eae1d760afd27aff78a71d676296fc`, CPU, Auto language and speaker-embedding conditioning. Pocket produced 1.825 seconds of audio in 1.886 seconds; Qwen produced 2.960 seconds in 8.060 seconds, plus 3.814 seconds to load. These single-run timings are not a quality ranking. No listening scores were assigned.

The generated artifacts are in `output/comparison-smoke/` (`pocket.wav`, `qwen.wav`, `comparison.json`, ignored by git). Desktop/mobile browser checks covered generation, hidden/revealed model names, rating/report export, report restoration and horizontal overflow; no JavaScript errors were observed.

The existing Pocket model has its existing noncommercial weight license; Qwen's release is Apache 2.0. Adding Qwen does not change Pocket's licensing.
