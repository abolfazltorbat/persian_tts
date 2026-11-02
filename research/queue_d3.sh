#!/usr/bin/env bash
# Study D: exclamations / surprised questions with the stronger lively style (lv3: +4 st, x1.6, +4 dB), rule OFF.
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
P='"prompt_ext_dir": "output/research/voice_ext_m", "prompt_ext_en": false, "max_attempts": 1, "intonation": 1.0, "intonation.VERSION": 2, "intonation.APPLY": false'
S='"prompt_styles": {"lively": [4.0, 1.6, 4.0], "ask": [2.0, 1.25, 2.0]}'
export VOICES=male_hello_clean,female_narration_clean,male_news SEEDS=1,2 KEEP_AUDIO=1
SET=research/sets/prosody_set.tsv CATS=exc TAGS="d_sty3" KNOBS_d_sty3="{$P, $S}" bash research/run_bench.sh
SET=research/sets/prosody_set2.tsv CATS=qecho,exc TAGS="d2_sty3" KNOBS_d2_sty3="{$P, $S}" bash research/run_bench.sh
