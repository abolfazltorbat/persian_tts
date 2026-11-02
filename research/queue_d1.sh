#!/usr/bin/env bash
# Study D: product-config renders WITHOUT intonation (offline rule variants are applied to these),
# then the paragraph texts with the study C rule (v1) and the new rule (v2) applied in join.
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
P='"prompt_ext_dir": "output/research/voice_ext_m", "prompt_ext_en": false, "max_attempts": 1'
export VOICES=male_hello_clean,female_narration_clean,male_news SEEDS=1,2 KEEP_AUDIO=1
SET=research/sets/prosody_set.tsv TAGS=d_base KNOBS_d_base="{$P}" bash research/run_bench.sh
SET=research/sets/prosody_set2.tsv TAGS=d2_base KNOBS_d2_base="{$P}" bash research/run_bench.sh
SET=research/sets/prosody_set2.tsv CATS=dyn TAGS="d2_v1 d2_v2" \
  KNOBS_d2_v1="{$P, \"intonation\": 1.0, \"intonation.VERSION\": 1}" \
  KNOBS_d2_v2="{$P, \"intonation\": 1.0, \"intonation.VERSION\": 2}" bash research/run_bench.sh
