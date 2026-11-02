#!/usr/bin/env bash
# Study D: (1) female exclamations / surprised questions with the high-voice register scaling
# (2) the one-word questions of set 1 (qshort) with the current styles, all voices. Rule OFF.
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
P='"prompt_ext_dir": "output/research/voice_ext_m", "prompt_ext_en": false, "max_attempts": 1, "intonation": 1.0, "intonation.VERSION": 2, "intonation.APPLY": false'
S='"prompt_styles": {"lively": [4.0, 1.6, 4.0], "ask": [2.0, 1.25, 2.0]}'
export SEEDS=1,2 KEEP_AUDIO=1
VOICES=female_narration_clean SET=research/sets/prosody_set.tsv CATS=exc TAGS="d_sty4f" KNOBS_d_sty4f="{$P, $S}" bash research/run_bench.sh
VOICES=female_narration_clean SET=research/sets/prosody_set2.tsv CATS=qecho,exc TAGS="d2_sty4f" KNOBS_d2_sty4f="{$P, $S}" bash research/run_bench.sh
VOICES=male_hello_clean,female_narration_clean,male_news SET=research/sets/prosody_set.tsv CATS=qshort TAGS="d_sty4q" KNOBS_d_sty4q="{$P, $S}" bash research/run_bench.sh
