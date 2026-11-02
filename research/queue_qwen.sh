#!/usr/bin/env bash
# Second model (Qwen3-TTS-0.6B-Base, CPU torch in .venv-qwen): plain vs. extended prompt, 25 Persian
# short sentences x 2 voices x 1 seed, then both judges + UTMOS + speaker similarity.
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
OMP_NUM_THREADS=6 .venv-qwen/bin/python research/qwen_ext_probe.py --out qwen2 --n 25 --seeds 1
for d in plain ext; do
  $PY research/punct_judge.py output/research/qwen2/$d --set research/sets/short_test.tsv --out output/research/qwen2/$d/whisper.jsonl
  ONLY=w2v,utmos,spk bash research/dir_eval.sh output/research/qwen2/$d
done
echo QWEN_DONE
