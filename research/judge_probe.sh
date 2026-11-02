#!/usr/bin/env bash
# After probe unit UNIT finishes: second judge on prompt_TAG/wav, then merge.
#   research/judge_probe.sh TAG UNIT
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
TAG=$1; UNIT=$2
while ! systemctl --user is-active --quiet "$UNIT" && pgrep -f "chain.sh [^ ]+ $UNIT " >/dev/null; do sleep 30; done
while systemctl --user is-active --quiet "$UNIT"; do sleep 30; done
nice -n 5 .venv-qwen/bin/python research/w2v_judge.py --wavs output/research/prompt_$TAG/wav \
  --out output/research/prompt_$TAG/w2v_hyps.jsonl --threads 3 2>&1 | grep -vi warn | tail -1
/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python research/rejudge_corpus.py merge-probe prompt_$TAG
echo "judged $TAG"
