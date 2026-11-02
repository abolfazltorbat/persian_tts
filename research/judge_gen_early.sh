#!/usr/bin/env bash
# Judge the nine-speaker robustness renders as soon as they exist (in parallel with the prosody renders).
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
until grep -q "=== gen_prod1" output/research/gen.log; do sleep 30; done
for t in gen_base gen_ext; do ONLY=whisper,w2v bash research/dir_eval.sh output/eval/flu_$t; done
/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python research/short_stats.py \
  output/eval/flu_gen_base output/eval/flu_gen_ext --json output/research/gen_robust.json
echo EARLY_DONE
