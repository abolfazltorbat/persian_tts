#!/usr/bin/env bash
# Study D: styled-prompt renders of the non-statement sentences (both sets, dev+test).
#   d_sty  = v2 rule + styles for exc/qecho (lively) and qyn/qwh (ask)
#   d_styx  = v2 rule + styles for exc/qecho only (questions from the neutral prompt)
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
P='"prompt_ext_dir": "output/research/voice_ext_m", "prompt_ext_en": false, "max_attempts": 1, "intonation": 1.0, "intonation.VERSION": 2, "intonation.APPLY": false'
S2='"prompt_styles": {"lively": [3.0, 1.4, 4.0], "ask": [2.0, 1.25, 2.0]}'
S1='"prompt_styles": {"lively": [3.0, 1.4, 4.0]}'
export VOICES=male_hello_clean,female_narration_clean,male_news SEEDS=1,2 KEEP_AUDIO=1
SET=research/sets/prosody_set.tsv CATS=qyn,qshort,qwh,exc TAGS="d_sty" KNOBS_d_sty="{$P, $S2}" bash research/run_bench.sh
SET=research/sets/prosody_set2.tsv CATS=qecho,qalt,qtag,exc TAGS="d2_sty" KNOBS_d2_sty="{$P, $S2}" bash research/run_bench.sh
SET=research/sets/prosody_set.tsv CATS=qyn,qshort,qwh TAGS="d_styx" KNOBS_d_styx="{$P, $S1}" bash research/run_bench.sh
