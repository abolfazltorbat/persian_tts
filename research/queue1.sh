#!/usr/bin/env bash
# queue 1: test corpus (A), voice extensions + prompt-length control (B3), B3 benches.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
T=output/research/voices_trunc
echo "=== prompt-length control $(date +%T)"
$PY research/prompt_probe.py --set dev --cats short --voices $PWD/$T/male_news_w143.wav,$PWD/$T/male_news_w180.wav,$PWD/$T/male_news_w248.wav,$PWD/$T/male_news_w319.wav,male_news --kinds ref,ref+ext --takes 6 --tag len1
echo "=== B3 short_test base/ext $(date +%T)"
KEEP_AUDIO=1 SET=research/sets/short_test.tsv TAGS="b3s_base b3s_ext" KNOBS_b3s_ext='{"prompt_ext_dir": "output/research/voice_ext"}' bash research/run_bench.sh
echo "=== B3 bench base/ext $(date +%T)"
TAGS="b3_base b3_ext" KNOBS_b3_ext='{"prompt_ext_dir": "output/research/voice_ext"}' bash research/run_bench.sh
echo "=== A test corpus $(date +%T)"
$PY research/detect_corpus.py --set bench --tag test --takes 4
echo "=== queue1 done $(date +%T)"
