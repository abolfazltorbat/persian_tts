"""Speaker similarity of every <case>__<voice>__s<seed>.wav in DIR(s) to that
voice's reference clip (voices/<voice>.wav). Runs in .venv-qwen (WeSpeaker ONNX).
    .venv-qwen/bin/python research/spk_dir.py DIR... -> DIR/spk.jsonl"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from speaker import SpeakerEmbedder  # noqa: E402

emb = SpeakerEmbedder(str(Path.home() / ".cache/persian_tts_eval/models/wespeaker_resnet34_LM.onnx"))
refs = {}
for d in map(Path, sys.argv[1:]):
    rows = []
    for w in sorted(d.glob("*.wav")):
        parts = w.stem.split("__")
        if len(parts) < 3:
            continue
        v = parts[1]
        if v not in refs:
            refs[v] = emb.embed(str(ROOT / "voices" / f"{v}.wav"))
        rows.append({"wav": w.name, "spk": round(float(refs[v] @ emb.embed(str(w))), 4)})
    (d / "spk.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(d, len(rows), "mean", round(sum(r["spk"] for r in rows) / max(1, len(rows)), 4))
