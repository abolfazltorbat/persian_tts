#!/usr/bin/env bash
# Copy a running bench tag's WAVs aside before run_bench.sh deletes them.
#   research/keep_wavs.sh TAG [TAG ...]   (exits when all tags have a report + no running bench)
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
while true; do
  for T in "$@"; do
    mkdir -p output/research/keep_wav/$T
    [ -d output/eval/flu_$T ] && cp -n output/eval/flu_$T/*.wav output/research/keep_wav/$T/ 2>/dev/null
  done
  all=1
  for T in "$@"; do
    n=$(ls output/research/keep_wav/$T/*.wav 2>/dev/null | wc -l)
    [ -f output/eval/flu_$T/report.json ] && [ "$n" -gt 0 ] && [ "$(ls output/eval/flu_$T/*.wav 2>/dev/null | wc -l)" -eq 0 ] || all=0
  done
  [ $all -eq 1 ] && break
  sleep 15
done
echo kept
