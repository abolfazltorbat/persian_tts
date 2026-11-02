#!/usr/bin/env bash
# After unit UNIT finishes: decode + second judge + merge for corpus NAME, then dev->test detector eval.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
UNIT=$1; NAME=$2
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
while systemctl --user is-active --quiet "$UNIT"; do sleep 30; done
nice -n 5 $PY research/rejudge_corpus.py decode $NAME 2>&1 | tail -1
nice -n 5 .venv-qwen/bin/python research/w2v_judge.py --wavs output/research/$NAME/wav --out output/research/$NAME/w2v_hyps.jsonl --threads 4 2>&1 | grep -vi warn | tail -1
$PY research/rejudge_corpus.py merge $NAME
nice -n 5 $PY research/detect_eval.py --dev corpus_dev --test $NAME --label cons > output/research/detect_eval_devtest_cons.txt 2>&1
echo "judged corpus $NAME"
