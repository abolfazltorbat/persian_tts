# Research code for the ICEE paper (prompt self-extension, ASR-free checks, two judges)

Everything runs with the three interpreters that already exist — nothing is installed:

| interpreter | used for |
|---|---|
| `chatbot/ai_backend/.venv/bin/python` | synthesis (ONNX engine), Whisper judge, all analysis |
| `PythonProject5/.venv/bin/python` | UTMOS, figures (matplotlib) |
| `persian_tts/.venv-qwen/bin/python` | WeSpeaker speaker similarity, wav2vec2 second judge |

Heavy jobs run as `systemd-run --user` units capped at 6 GB (see `run_bench.sh`,
`queue*.sh`, `chain.sh`); at most two at once.

## Engine switches (scripts/tts_onnx.py — all OFF by default)

| attribute | meaning |
|---|---|
| `prompt_ext_dir` | directory of `<voice>.ext.npy`; prompt = clip + extension (≤ 5 s, clip first) |
| `prompt_ext_order` | `"ref+ext"` (default; the other order makes the model read the extension aloud) |
| `prompt_ext_en` | `False` = English sentences keep the plain clip (recommended) |
| `prepare_extension(wav, out_dir)` | builds the extension once per voice, no recogniser (median-length take) |
| `max_attempts` | takes per unit in the signal-QC retry loop (default 3) |
| `prime_short_tokens`, `prime_carrier` | carrier (continuation) priming of short units — null result |
| `prime_tail_s`, `prime_tail_ref` | previous unit's tail as prompt (joins study) |

Recommended configuration: `prompt_ext_dir=<dir from prepare_extension>`, `prompt_ext_en=False`.

## Pipeline

1. Sentence sets: `sets/dev_sentences.tsv` (110, tuning only), `sets/short_test.tsv` (60, test only);
   bench = `eval/cases.py` (79). Real short speech for the ASR floor: `fetch_real_short.py` (GPTInformal, CC0).
2. Take corpora (study A): `detect_corpus.py --set dev|bench --tag …` → `output/research/corpus_<tag>/`
   (rows.jsonl + latents). Second judge: `rejudge_corpus.py decode|merge`, `w2v_judge.py`.
   Analysis: `detect_eval.py --dev corpus_dev [--test corpus_test] --label whisper|w2v|cons`.
3. Prompt probes (study B, no retry, paired seeds): `prompt_probe.py --kinds …` →
   `output/research/prompt_<tag>/` (rows + wav); `judge_probe.sh`; tables: `probe_table.py`.
4. Production-path benches: `run_bench.sh` (eval/fluency.py `--set`, knobs, UTMOS, WeSpeaker);
   second judge on kept renders: `judge_bench.sh` / `rescore_w2v.py`; statistics: `paired_stats.py
   [--judge w2v]` (case-bootstrap CI, Wilcoxon).
5. ASR floor: `asr_floor.py`. Figure 1: `fig_judges.py`. Listening tests: `build_ab_page.py`.

Labels: consonant-skeleton CER (scripts/verify.py) per take; a take fails above 0.3;
**consensus** = min(Whisper CER, wav2vec2 CER) — fails only if both recognisers fail it.
Ledger with every number and decision: `PROGRESS.md`.
