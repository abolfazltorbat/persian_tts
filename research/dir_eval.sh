#!/usr/bin/env bash
# Study D: every judge on directories of <case>__<voice>__s<seed>.wav renders.
#   research/dir_eval.sh DIR... [ONLY=whisper,w2v,utmos,spk,prosody]
# -> DIR/{whisper.jsonl, w2v_hyps.jsonl, utmos.jsonl, spk.jsonl, prosody.jsonl}; research/dir_compare.py reads them.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
TORCH=/home/abolfazl/PycharmProjects/PythonProject5/.venv/bin/python
QWEN=.venv-qwen/bin/python
ONLY=${ONLY:-whisper,w2v,utmos,spk,prosody}
for D in "$@"; do
  echo "=== $D $(date +%T)"
  [[ $ONLY == *whisper* ]] && $PY research/punct_judge.py "$D" --out "$D/whisper.jsonl" 2>&1 | grep -E "^wrote|Error" | tail -2
  [[ $ONLY == *w2v* ]] && nice -n 5 $QWEN research/w2v_judge.py --wavs "$D" --out "$D/w2v_hyps.jsonl" --threads 4 2>&1 | grep -vi warn | tail -1
  [[ $ONLY == *utmos* ]] && $TORCH research/utmos_dir.py "$D" 2>&1 | tail -1
  [[ $ONLY == *spk* ]] && $QWEN research/spk_dir.py "$D" 2>&1 | tail -1
  [[ $ONLY == *prosody* ]] && $QWEN research/prosody.py "$D" --out "$D/prosody.jsonl" 2>&1 | tail -1
done
echo "=== done $(date +%T)"
