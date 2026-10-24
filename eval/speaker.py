"""Speaker similarity: is a render still the reference voice?

    python eval/speaker.py --model wespeaker.onnx voices/male_hello_clean.wav a.wav b.wav

WeSpeaker ResNet34 (VoxCeleb, ONNX) embeddings, cosine similarity against the
first file. Needs torchaudio (Kaldi-compatible fbank) and onnxruntime. Used to
check a candidate reference clip keeps the speaker: same-speaker pairs score
high, different speakers low (see the calibration printed by --calibrate).
"""
from __future__ import annotations

import argparse
from pathlib import Path


class SpeakerEmbedder:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def embed(self, path_or_audio, sr: int | None = None):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        if isinstance(path_or_audio, (str, Path)):
            x, sr = sf.read(str(path_or_audio), dtype="float32")
        else:
            x = np.asarray(path_or_audio, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        w = torch.from_numpy(x).unsqueeze(0)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        feats = torchaudio.compliance.kaldi.fbank(
            w * 32768, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
            sample_frequency=16000, window_type="hamming", use_energy=False)
        feats = feats - feats.mean(dim=0, keepdim=True)
        e = self.sess.run(None, {"feats": feats.unsqueeze(0).numpy()})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    a = ap.parse_args()
    emb = SpeakerEmbedder(a.model)
    r = emb.embed(a.ref)
    for o in a.others:
        print(f"{float(r @ emb.embed(o)):.3f}  {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
