#!/usr/bin/env bash
# queue 3: Whisper-only probes re-run WITH audio (for the consensus judge):
# clip-order control and the male_news cut/length control.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
T=$PWD/output/research/voices_trunc
echo "=== order control (ext+ref) $(date +%T)"
$PY research/prompt_probe.py --set dev --cats short --voices male_hello_clean,female_narration_clean --kinds "ext+ref" --takes 6 --tag order
echo "=== length control with audio $(date +%T)"
$PY research/prompt_probe.py --set dev --cats short --voices $T/male_news_w143.wav,$T/male_news_w180.wav,$T/male_news_w248.wav,$T/male_news_w319.wav,male_news --kinds "ref,ref+ext:A" --takes 6 --tag len2
echo "=== single-take variants (compute vs quality) $(date +%T)"
X=output/research/voice_ext_m
KEEP_AUDIO=1 SET=research/sets/short_test.tsv SEEDS=1,2,3,4,5,6 TAGS="b3s_base_a1 b3s_ext_a1" \
  KNOBS_b3s_base_a1='{"max_attempts": 1}' \
  KNOBS_b3s_ext_a1='{"max_attempts": 1, "prompt_ext_dir": "'$X'"}' bash research/run_bench.sh
echo "=== bench with the FINAL config (ASR-free ext, Persian/mixed only) $(date +%T)"
KEEP_AUDIO=1 TAGS="b3f_ext" KNOBS_b3f_ext='{"prompt_ext_dir": "'$X'", "prompt_ext_en": false}' bash research/run_bench.sh
echo "=== queue3 done $(date +%T)"
