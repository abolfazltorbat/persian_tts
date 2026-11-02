#!/usr/bin/env bash
# queue 5 (10-31 ~08:55): held-out confirmation of the SPEAKER-MATCHED extension
# (voice_ext_s, chosen on dev probe ext8), 3 voices x 6 seeds, audio kept.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
XS=output/research/voice_ext_s
S=research/sets/short_test.tsv
for V in male_hello_clean female_narration_clean male_news; do
  KEEP_AUDIO=1 VOICES=$V SET=$S SEEDS=1,2,3,4,5,6 TAGS="b3x_ext_$V" \
    KNOBS_b3x_ext_male_hello_clean='{"prompt_ext_dir": "'$XS'"}' \
    KNOBS_b3x_ext_female_narration_clean='{"prompt_ext_dir": "'$XS'"}' \
    KNOBS_b3x_ext_male_news='{"prompt_ext_dir": "'$XS'"}' bash research/run_bench.sh
done
echo "=== queue5 done $(date +%T)"
