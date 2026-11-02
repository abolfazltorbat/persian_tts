#!/usr/bin/env bash
# Judge the nine-speaker prosody renders as each tag finishes (parallel to the main queue).
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
next() { case $1 in gen_prod1) echo gen_final1;; gen_final1) echo gen_prod2;; gen_prod2) echo gen_final2;; gen_final2) echo "output/eval/flu_gen_base";; esac; }
for t in gen_prod1 gen_final1 gen_prod2 gen_final2; do
  n=$(next $t)
  until grep -q "=== $n" output/research/gen.log; do sleep 30; done
  ONLY=whisper,utmos,prosody bash research/dir_eval.sh output/eval/flu_$t
done
echo EXPR_DONE
