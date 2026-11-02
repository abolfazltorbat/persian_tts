#!/usr/bin/env bash
# queue 8: sentence G (dev-validated) on the held-out short set, male + female, 6 seeds.
set -u
cd /home/abolfazl/PycharmProjects/PythonProject5/persian_tts
PY=/home/abolfazl/PycharmProjects/PythonProject5/chatbot/ai_backend/.venv/bin/python
$PY - <<'PYEOF'
import sys; sys.path.insert(0, 'research')
import core, numpy as np
lab = core.Lab(asr=False)
G = "پدرم هر روز صبح در پارک قدم می‌زند."
ph = " ".join(u[0] for u in lab.units(G))
for v in ("male_hello_clean", "female_narration_clean"):
    f = lab.eng.prepare_extension(lab.voice_path(v), "output/research/voice_ext_g", phonemes=ph)
    print("G extension", v, f, np.load(f).shape)
PYEOF
for V in male_hello_clean female_narration_clean; do
  KEEP_AUDIO=1 VOICES=$V SET=research/sets/short_test.tsv SEEDS=1,2,3,4,5,6 TAGS="b3g_ext_$V" \
    KNOBS_b3g_ext_male_hello_clean='{"prompt_ext_dir": "output/research/voice_ext_g"}' \
    KNOBS_b3g_ext_female_narration_clean='{"prompt_ext_dir": "output/research/voice_ext_g"}' bash research/run_bench.sh
done
echo "=== queue8 done $(date +%T)"
