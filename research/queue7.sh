#!/usr/bin/env bash
# queue 7 (10-31 ~09:35): assistant-style replies, today's default vs the RECOMMENDED config
# (median-take extension, Persian/mixed only, single take). Male default voice, 3 seeds.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
XM=output/research/voice_ext_m
KEEP_AUDIO=1 SET=research/sets/assistant_test.tsv SEEDS=1,2,3 TAGS="asst_base asst_rec" \
  KNOBS_asst_rec='{"prompt_ext_dir": "'$XM'", "prompt_ext_en": false, "max_attempts": 1}' bash research/run_bench.sh
echo "=== queue7 done $(date +%T)"
