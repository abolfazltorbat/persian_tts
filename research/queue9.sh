#!/usr/bin/env bash
# queue 9 (10-31 ~10:05): three NEW real clips (GPTInformal, CC0, tightly cut) as voices,
# held-out short set, plain clip vs median-take extension, 3 seeds, audio kept.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
N=$PWD/output/research/voices_new
$PY - <<'PYEOF'
import sys; sys.path.insert(0, 'research')
import core
lab = core.Lab(asr=False)
for v in ("gi1", "gi2", "gi3"):
    f = lab.eng.prepare_extension(f"output/research/voices_new/{v}.wav", "output/research/voice_ext_m")
    print("extension", v, f)
PYEOF
for V in gi1 gi2 gi3; do
  KEEP_AUDIO=1 VOICES=$N/$V.wav SET=research/sets/short_test.tsv SEEDS=1,2,3 TAGS="nv_base_$V nv_ext_$V" \
    KNOBS_nv_ext_gi1='{"prompt_ext_dir": "output/research/voice_ext_m"}' \
    KNOBS_nv_ext_gi2='{"prompt_ext_dir": "output/research/voice_ext_m"}' \
    KNOBS_nv_ext_gi3='{"prompt_ext_dir": "output/research/voice_ext_m"}' bash research/run_bench.sh
done
echo "=== queue9 done $(date +%T)"
