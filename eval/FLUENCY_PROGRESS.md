# TTS fluency work — progress ledger (2026-10-23/24)

Goal (owner): default voice «مردانه (تمیزشده)» in Dana, and the most fluent,
human-sounding reading possible — improve in persian_tts first (measured), then
ship the best settings into chatbot.

## RESUME HERE

1. **Sweep 2 may still be running** — `systemctl --user status tts-sweep2`
   (log: `~/.cache/persian_tts_eval/sweep2.log`). It ends with the UTMOS +
   speaker passes and a comparison table. If it died, rerun the missing tags:
   `~/.cache/persian_tts_eval/exp/sweep.sh ~/.cache/persian_tts_eval/exp/spec2.txt`
   (finished tags are skipped; run it as a capped unit — see "How to run").
2. Decide from sweep 2 (rules below), then run
   `python ~/.cache/persian_tts_eval/exp/finalize_engine.py` (makes the
   balanced chunker + 30/4 budget + English 18/2 the ONLY path, deletes the
   greedy packer) and set the chosen temp / room_tone / level_target defaults.
3. Confirmation run, final config vs `base`, on three voices
   (male_hello_clean, female_narration_clean, male_news), all cases, 3 seeds.
   The female voice must not regress.
4. Ship to chatbot (see "Chatbot integration"), build the UI, publish the
   listening page, write `chatbot/CHANGES_2026_10_24_TTS_FLUENCY.md`, commit +
   push both repos to `main` (owner: main is the only branch).

## Done

- chatbot (committed to main by another session's "commit all work" at 01:34,
  pushed): default voice is `male_hello_clean` (`engine/runtime.py`
  BUILTIN_VOICES order; persian_onnx provider uses the same constant); the
  settings picker shows the Persian label of the default; `faulthandler` on in
  `app/main.py` (a native crash now prints every thread's Python stack); disk
  guard in `backend/speech_assets.py` (`_room_for`: a model download needs its
  size + 1 GB free, else an error state the admin panel shows). Tests pass
  (TTS 149, speech_assets 21).
- Whisper large-v3-turbo re-downloaded into `chatbot/storage/models/whisper/`
  (it had been deleted; Dana's dictation needs it too).
- persian_tts commit 136ee50 (pushed): `eval/fluency.py`, `eval/speaker.py`,
  12 paragraph cases, engine candidates OFF by default.

## Measured (male_hello_clean, 60 cases × 3 seeds, Whisper turbo, CUDA)

Single generation vs length (whole sentences, 3 voices): 10–39 tokens → EOS
100 %, CER 4.7–7.7 %, 0 % fails; 40+ tokens hit the decoder cap (128 latents =
10.24 s) → truncated. The old "21+ tokens never terminate" does not hold.

| tag | knobs | CER | fail | UTMOS | spk | mid-clause joins | punct joins |
|---|---|---|---|---|---|---|---|
| base | current engine | 7.1 % | 1.7 % | 3.253 | 0.279 | 99 | 96 |
| bal18 | balanced chunker | 7.2 % | 2.2 % | 3.249 | 0.280 | 87 | 96 |
| **bal30** | balanced + CHUNK_TOKENS 30, HEADROOM 4 | **7.1 %** | 1.7 % | **3.273** | 0.280 | **12** | 93 |
| bal30p | bal30 + pack (commas read through) | 7.3 % | 1.7 % | 3.263 | 0.280 | 21 | 12 |
| t02 | bal30 + temp 0.2 | 7.0 % | 0.6 % | (sweep2) | | | |
| t04 | bal30 + temp 0.4 | 7.8 % | 3.9 % | | | | |
| t05 | bal30 + temp 0.5 | 8.0 % | 3.3 % | | | | |

By category bal30 vs base: Persian 6.8 → 6.3 %, mixed 10.3 → 9.8 %, English
4.9 → **7.6 %** (worse) → fix already in finalize_engine.py: pure-English
sentences keep the 18-token budget. Pitch range (p10–p90) base 7.96 st,
bal30 7.74, t02 7.58 — temp 0.2 is not audibly flatter. Loudness spread
between renders today: 3.4 dB (p10–p90) — `level_target` removes it.

UTMOS caveat: it scores the noisy HUMAN reference clips 1.5–2.9 and our TTS
3–4, so it only ranks renders against each other. Speaker similarity
(WeSpeaker): same speaker/two recordings 0.81–0.88, different speakers
0.02–0.10, renders vs their reference ~0.28.

## Decision rules for sweep 2

- temp: keep 0.3 unless t02's UTMOS is ≥ bal30's (then 0.2: fewer fails).
  0.4/0.5 rejected.
- rl (room_tone + level_target 0.13): adopt if CER/UTMOS are not worse — the
  level target is an objective fix for cross-request loudness steps.
- vA / vB / vC (new reference clips for male_hello_clean, built from the
  model's own best takes; vC = original 1.3 s clip + vA): adopt one only if
  CER ≤ bal30, fail ≤ bal30, spk vs the ORIGINAL clip ≥ ~0.28 (no drift), and
  the short-sentence failure `fa_casual` is not worse. Ship it as
  `voices/male_hello_clean_v2.wav` and map it in chatbot
  `runtime.VOICE_FILES` (the saved id `male_hello_clean` must not change).
  If none qualifies, keep the original clip.

## Chatbot integration (after finalizing persian_tts)

- `cd chatbot/ai_backend && .venv/bin/python scripts/sync_tts_vendor.py`
  then `--check`; tests: `tests/test_tts*.py tests/test_tts_vendor_sync.py`.
- `app/backend/tts/engine/runtime.py::_render`: the sentence silence
  (`np.zeros`) → `eng.pause(...)` so room tone applies there too.
- `app/backend/tts/chunking.py`: `ENGINE_REVISION = "2026.10.24"` and bump
  `FRONTEND_VERSION` in `persian_tts/scripts/frontend.py` (cache keys — else
  users hear stale cached audio).
- Chatbot's own harness: `.venv/bin/python -m eval.speech.run --tag fluency1`
  (voice male_hello_clean) — end-to-end CER/RTF.
- UI: `cd chatbot/front_svelte && npm run build` (the settings label change
  is committed but the served build predates it).
- READMEs: `README.en.md:328` and the same line in README.md still say
  "~18 tokens".

## Listening page

`~/.cache/persian_tts_eval/exp/build_page.py <before_tag> <after_tag>
<voice_before> <voice_after> <out_dir> p_ev_motor,fa_runon,p_neural,p_project,p_chatty,p_battery`
writes MP3s + data.json; fill `page_template.html` (`__DATA__`, `__METHOD__`)
and publish it as an Artifact (files: audio/*.mp3). Has a blind A/B mode.

## How to run (owner rule: never from the IDE's process tree)

One GPU job at a time, capped:
`systemd-run --user --unit=tts-X --collect -p MemoryMax=6G -p MemorySwapMax=1G
 -p StandardOutput=file:LOG -p StandardError=file:LOG <cmd>`.
Venvs: synthesis+ASR `chatbot/ai_backend/.venv` (onnxruntime-gpu, whisper);
UTMOS `PythonProject5/.venv` (CUDA torch; the jit is traced on CUDA);
speaker `persian_tts/.venv-qwen` (torchaudio). Models:
`~/.cache/persian_tts_eval/models/` (utmos.jit 413 MB, WeSpeaker 26 MB) —
delete when done. Disk is near full (1.5 GB free on 10-24): delete the WAVs of
rejected tags (keep report.json).
