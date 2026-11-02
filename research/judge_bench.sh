#!/usr/bin/env bash
# After unit UNIT finishes: second judge on kept bench renders of each TAG, then rescore.
#   research/judge_bench.sh UNIT TAG [TAG ...]
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
UNIT=$1; shift
while ! systemctl --user is-active --quiet "$UNIT" && pgrep -f "chain.sh [^ ]+ $UNIT " >/dev/null; do sleep 30; done
while systemctl --user is-active --quiet "$UNIT"; do sleep 30; done
for T in "$@"; do
  nice -n 5 .venv-qwen/bin/python research/w2v_judge.py --wavs output/eval/flu_$T \
    --out output/eval/flu_$T/w2v_hyps.jsonl --threads 4 2>&1 | grep -vi warn | tail -1
done
/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python research/rescore_w2v.py "$@"
echo "judged bench $*"
