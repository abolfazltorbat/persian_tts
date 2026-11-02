"""Voice preparation for prompt self-extension (study B).

A short reference clip (the default male_hello_clean is 1.3 s) gives the model
little context. Once per voice, the model reads a neutral sentence in that
voice; the cleanest of a few fixed-seed takes (judged once, offline) is saved
as RAW latents. The engine (OnnxTts.prompt_ext_dir) then conditions every
take on clip + extension, within the trained 5 s prompt window.

    python research/prep_voice_ext.py --voices male_hello_clean,female_narration_clean,male_news
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from prompt_probe import EXT_TEXT, self_extension  # noqa: E402

DIR = core.OUT / "voice_ext"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="male_hello_clean,female_narration_clean,male_news")
    ap.add_argument("--text", default=EXT_TEXT)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--out", default=str(DIR))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = core.Lab()
    for v in a.voices.split(","):
        raw, info = self_extension(lab, v, a.text, a.tries)
        stem = Path(lab.voice_path(v)).stem
        np.save(out / f"{stem}.ext.npy", raw.astype(np.float32))
        info.update(voice=v, text=a.text, clip_frames=int(len(lab.prompt_latents(v))))
        (out / f"{stem}.ext.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(v, info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
