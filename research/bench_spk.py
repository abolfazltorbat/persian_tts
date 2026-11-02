"""Speaker similarity of kept bench renders to an arbitrary clip (for voices outside voices/).
Runs in .venv-qwen.   .venv-qwen/bin/python research/bench_spk.py CLIP BASE_TAG NEW_TAG"""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from speaker import SpeakerEmbedder  # noqa: E402

clip, base, new = sys.argv[1], sys.argv[2], sys.argv[3]
emb = SpeakerEmbedder(str(Path.home() / ".cache/persian_tts_eval/models/wespeaker_resnet34_LM.onnx"))
r = emb.embed(clip)


def sims(tag):
    out = {}
    for w in sorted((ROOT / "output" / "eval" / f"flu_{tag}").glob("*.wav")):
        case, _voice, seed = w.stem.split("__")
        out[(case, seed)] = float(r @ emb.embed(str(w)))
    return out


a, b = sims(base), sims(new)
keys = [k for k in a if k in b]
d = collections.defaultdict(list)
for k in keys:
    d[k[0]].append(b[k] - a[k])
rng = np.random.default_rng(0)
cs = list(d)
bs = [np.mean([x for j in rng.integers(0, len(cs), len(cs)) for x in d[cs[j]]]) for _ in range(2000)]
print(f"spk {new} vs {base}: n={len(keys)} {np.mean([a[k] for k in keys]):.4f} -> {np.mean([b[k] for k in keys]):.4f} "
      f"diff {np.mean([b[k] - a[k] for k in keys]):+.4f} CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
