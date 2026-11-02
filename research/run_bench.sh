#!/usr/bin/env bash
# Fluency-bench variants for the A/B studies, then UTMOS + speaker similarity.
# Same three interpreters as ~/.cache/persian_tts_eval/exp/bench.sh (nothing installed).
# Env: TAGS (required), KNOBS_<tag> (json), SET (tsv; default eval/cases.py),
#      VOICES (male_hello_clean), SEEDS (1,2,3), CATS, KEEP_AUDIO=1 keeps the WAVs.
set -u
TREE=/home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
TORCH=/home/abolfazl/PycharmProjects/PythonProject5/.venv/bin/python
QWEN=$TREE/.venv-qwen/bin/python
M=/home/abolfazl/.cache/persian_tts_eval/models
cd "$TREE"
for TAG in $TAGS; do
  free=$(df --output=avail -m / | tail -1)
  if [ "$free" -lt 1500 ]; then echo "=== STOP: only ${free} MB free"; exit 1; fi
  v="KNOBS_$TAG"; K="${!v:-}"
  echo "=== $TAG set=${SET:-cases.py} voices=${VOICES:-male_hello_clean} knobs=$K $(date +%T)"
  $PY eval/fluency.py --tag "$TAG" --voices "${VOICES:-male_hello_clean}" --seeds "${SEEDS:-1,2,3}" \
      ${SET:+--set "$SET"} ${CATS:+--cats "$CATS"} ${K:+--knobs "$K"} 2>&1 \
      | grep -vE "^phonemes:|onnxruntime|provider_bridge" | tail -6
  $TORCH eval/fluency.py --utmos "$TAG" --utmos-model "$M/utmos.jit" 2>&1 | grep -E "utmos all|Error" | tail -2
  $QWEN eval/fluency.py --speaker "$TAG" --speaker-model "$M/wespeaker_resnet34_LM.onnx" 2>&1 | grep -E "spk all|Error" | tail -2
  [ -z "${KEEP_AUDIO:-}" ] && find "output/eval/flu_$TAG" -name "*.wav" -delete
done
echo "=== done $(date +%T)"
