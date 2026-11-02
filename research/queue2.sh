#!/usr/bin/env bash
# queue 2 (rewritten 10-31 ~05:40 after the judge finding): the big held-out
# B3 run with audio kept for both judges, then the leak control, then B2.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
X=output/research/voice_ext
S=research/sets/short_test.tsv
echo "=== B3 Persian-only extension on English/acronym cases $(date +%T)"
CATS=english,acro TAGS="b3_extfa" KNOBS_b3_extfa='{"prompt_ext_dir": "'$X'", "prompt_ext_en": false}' bash research/run_bench.sh
echo "=== B3 short_test big run (ASR-free extension, 3 voices x 6 seeds) $(date +%T)"
XM=output/research/voice_ext_m
for V in male_hello_clean female_narration_clean male_news; do
  KEEP_AUDIO=1 VOICES=$V SET=$S SEEDS=1,2,3,4,5,6 TAGS="b3m_base_$V b3m_ext_$V" \
    KNOBS_b3m_ext_male_hello_clean='{"prompt_ext_dir": "'$XM'"}' \
    KNOBS_b3m_ext_female_narration_clean='{"prompt_ext_dir": "'$XM'"}' \
    KNOBS_b3m_ext_male_news='{"prompt_ext_dir": "'$XM'"}' bash research/run_bench.sh
done
echo "=== A leak positive control (ext+ref prompt) $(date +%T)"
$PY research/detect_corpus.py --set dev --cats short,casual --tag leakctl --takes 3 --voices male_hello_clean,female_narration_clean --prompt-kind ext+ref
echo "=== B2 tail priming (multi-unit cats) $(date +%T)"
CATS=prose,paragraph,punct,gram TAGS="b2_t2 b2_t2x" \
  KNOBS_b2_t2='{"prime_tail_s": 2.0}' \
  KNOBS_b2_t2x='{"prime_tail_s": 2.0, "prompt_ext_dir": "'$X'"}' bash research/run_bench.sh
echo "=== queue2 done $(date +%T)"
