#!/usr/bin/env bash
# Review point 5 (generalisation): nine more speakers (ShEMO, research/pick_shemo_voices.py).
#   gen_base / gen_ext : short_test (60 sentences) x 2 seeds, plain clip vs self-extension (1 take each)
#   gen_prod / gen_final : held-out prosody sentences, shipped rule vs study D (styles + v2 rule), seed 1
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
V=/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh01.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh02.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh03.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh04.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh05.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh06.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh07.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh08.wav,/home/abolfazl/PycharmProjects/PythonProject5/persian_tts/output/research/voices_shemo/sh09.wav
EXT='"prompt_ext_dir": "output/research/voice_ext_sh", "prompt_ext_en": false, "max_attempts": 1'
run() { echo "=== $1 $(date +%T)"; $PY eval/fluency.py --tag "$1" --voices "$V" "${@:2}" 2>&1 | grep -vE "^phonemes|onnxruntime|provider_bridge|voice prompt" | tail -3; }
mkdir -p output/research/voice_ext_sh
run gen_base --seeds 1,2 --set research/sets/short_test.tsv --knobs '{"max_attempts": 1}'
run gen_ext  --seeds 1,2 --set research/sets/short_test.tsv --knobs "{$EXT}"
for s in 1 2; do
  C=$(cat output/research/gen_cases$s.txt); SETF=research/sets/prosody_set$([ $s = 2 ] && echo 2).tsv
  run gen_prod$s  --seeds 1 --set $SETF --cases "$C" --knobs "{$EXT, \"intonation\": 1.0, \"intonation.VERSION\": 1}"
  run gen_final$s --seeds 1 --set $SETF --cases "$C" --knobs "{$EXT, \"intonation\": 1.0, \"intonation.VERSION\": 2, \"prompt_styles\": {\"lively\": [4.0, 1.6, 4.0]}}"
done
for t in gen_base gen_ext gen_prod1 gen_final1 gen_prod2 gen_final2; do
  bash research/dir_eval.sh output/eval/flu_$t
  $PY research/dnsmos_dir.py output/eval/flu_$t 2>&1 | tail -1
done
echo "=== all done $(date +%T)"
