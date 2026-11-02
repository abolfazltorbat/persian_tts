# Small-data fine-tune of pocket-tts-farsi-v2 on a 12 GB GPU — ledger (2026-10-31)

Goal (owner): fine-tune on this laptop (RTX 3060 12 GB), beat the current
output (pocket-tts-farsi-v2 + our text front end) with a SMALL dataset, and
write it up for ICEE.

## RESUME HERE

0. 10-31 owner: "don't go ahead and never change the env" → NO venv / pip
   installs. Owner wants the paper's NOVELTY first, preferably in pre- or
   post-processing (model only as fallback). Novelty candidates sent (chat):
   (1) ASR-free loop/failure detector on Mimi latents + targeted retry,
   (2) continuation priming (previous chunk tail / cached carrier as voice
   prompt) for joins + short sentences, (3) ezafe-aware DP chunker (measured),
   (4) English→Persian phoneme front end (measured), (5) script-agnostic CER
   for code-mixed Persian (measured). All need only the installed
   onnxruntime/numpy/faster-whisper. Waiting for the owner's pick.
1. (superseded) BLOCKED 10-31: installing the training code was denied by the auto-mode
   permission check ("code from external"). Asked the owner to approve.
   Needed: venv `~/.cache/persian_tts_ft/venv` (reuses the torch 2.13 cu130 of
   `PythonProject5/.venv` via a .pth file, so no second torch download) +
   `pip install --no-deps -e ~/.cache/persian_tts_ft/src/mallahyari` +
   numpy>=2 pyarrow sphn pyyaml transformers<5 jiwer fastapi uvicorn
   python-multipart.
2. Next: download weights `mehdi-hf/pocket-tts-farsi-v2/model.safetensors`
   (438 MB), data (below), align, phonemise, precompute latents, first run.

## Facts established

- GPU RTX 3060 12 GB, 31 GB RAM, disk 18 GB free (93 % used) — delete
  parquet after transcoding; keep ≤3 checkpoints.
- Code: `~/.cache/persian_tts_ft/src/kyutai` (upstream, 91a9e96) and
  `.../mallahyari` (Farsi fork that trained v2, 3807c20). The fork has the
  whole Farsi pipeline: `training/farsi/prepare_data_fa.py` (manatts, filimo,
  youtube, commonvoice), `v2/phonemize_manifest.py` (Homo-GE2PE), aligner
  `m3hrdadfi/wav2vec2-large-xlsr-persian-v3`, `scripts/precompute_latents.py`.
- The released v2 is the 6-layer **distilled student**; the 24-layer Farsi
  teacher is NOT public (author's GCS bucket). Distillation regressed the
  student's backbone onto the teacher's CFG-combined output (coef 2.0), so a
  plain flow-loss fine-tune moves it towards the *unguided* distribution.
  The author's own anneal recipe kept the distillation objective for this
  reason. → our method: **anchored fine-tune** = flow + EOS loss on new data
  + λ·‖z − z_orig‖² against a frozen copy of v2 (λ = 0 is the plain baseline).
- Data (CC0, no login): Mana-TTS (114 h, 79 parquet ≈ 36 GB, one female
  narrator, formal; v2 already saw 60 h of it), GPTInformal-Persian (6 h,
  5 parquet ≈ 2 GB, short informal clips), Common Voice fa via
  `fsicoli/common_voice_22_0` (56 h with audio, ~4 s clips, many speakers).
  ParsVoice is gated (owner's HF login) and 16 kHz — not used for now.

## Baseline to beat (`output/eval/flu_cur`, 10-24, 79 cases × 3 seeds = 237)

CER 5.7 %, fail 1.7 %, UTMOS 3.284, spk 0.286. By category CER: prose 7.0,
gram 7.0, english 6.4, acro 6.1, mixed 5.6, punct 5.3, numeric 4.8,
paragraph 4.8. Worst renders: fa_casual (short, 6 tokens) 23–46 %;
en_weather / en_acronym repeat words (runaway); register drift — the model
says «رو / دادن / میخواین» for written «را / داده‌اند / می‌خواهند»
(colloquial YouTube/Filimo training data).

## Plan

- Test sets: the 79-case bench (no-regression + categories) + a new
  held-out SHORT set (2–6-word sentences, formal + informal, written fresh so
  nothing overlaps training text). Re-run the baseline in the same session.
- Runs (each ≈1 h train + ≈30 min export/bench): E1 plain FT (λ=0);
  E2 anchored (λ=1); E3 data-size ablation (≈2 / 6 / 15 h); E4 data-type
  ablation (short/informal vs formal). Export to ONNX with
  `scripts/export_unified.py` so the bench runs the identical engine.
- Paper: ICEE, IEEE template (reuse PythonProject1/paper setup: tectonic,
  newtx). Deadline still unknown — ask the owner.
