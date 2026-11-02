"""Study D1 — does the voice prompt carry REGISTER and LIVELINESS into the take?

Prompt self-extension (study B) made the prompt end with the model's own
reading of a neutral sentence. Here that reading is re-styled before it is
used: decoded to audio, pitch-shifted / range-expanded / made louder with
TD-PSOLA, and encoded back to Mimi latents. The OUTPUT is generated fresh by
the model and decoded once, so whatever transfers costs no signal processing
on the output. A plain decode->encode round trip (variant "rt") is the
control for the codec pass itself.

    python research/style_probe.py --out style1 [--voices ...] [--seeds 1,2]

Writes output/research/<out>/<variant>/<case>__<voice>__s<seed>.wav and
measures.jsonl (F0 median Hz, F0 range st, speech level dB, duration).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))

# variant -> (register shift st, range scale, gain dB)
VARIANTS = {
    "ext": None,                   # the shipped neutral extension, untouched
    "rt": (0.0, 1.0, 0.0),         # codec round trip only (control)
    "p2": (2.0, 1.0, 0.0),
    "p4": (4.0, 1.0, 0.0),
    "lv": (3.0, 1.4, 4.0),         # livelier: higher, wider, louder
    "lo": (-2.0, 0.8, -4.0),       # calmer: lower, narrower, softer
    # spectral styling: the extension audio moved toward the clip's long-term
    # spectrum (scripts/voice_eq.py curve from 60 plain-clip renders)
    "lv2": (5.0, 1.5, 6.0),        # stronger: ShEMO surprise sits several st above neutral
    "lv3": (4.0, 1.6, 4.0),
    "lv3h": (2.0, 1.6, 4.0),       # lv3 with half the register shift (high voices)
    "lv3r": (4.0, 1.3, 4.0),       # lv3 with less range
    "br": (0.0, 1.0, 0.0, 1.0),
    # tempo: the extension read 10 % / 20 % faster (PSOLA time scale, pitch kept)
    "tf10": (0.0, 1.0, 0.0, 0.0, 1.10),
    "tf20": (0.0, 1.0, 0.0, 0.0, 1.20),
    "brlv": (3.0, 1.4, 4.0, 1.0),
}
EQ_H: dict = {}


def eq_filter(eng, voice: str):
    if voice not in EQ_H:
        import glob
        import soundfile as sf
        import voice_eq as Q
        ref, _ = sf.read(ROOT / "voices" / f"{voice}.wav", dtype="float32")
        bench = [sf.read(f, dtype="float32")[0]
                 for f in sorted(glob.glob(str(ROOT / f"output/eval/flu_c_base/*__{voice}__*.wav")))[:60]]
        f, d = Q.curve(ref, bench, eng.sample_rate)
        EQ_H[voice] = Q.fir(f, d, eng.sample_rate, 1.0)
    return EQ_H[voice]


def restyle(eng, raw: np.ndarray, reg: float, rng: float, gain_db: float, eq: float = 0.0,
            tempo: float = 1.0, voice: str = "") -> np.ndarray:
    """RAW extension latents -> audio -> PSOLA restyle (+ spectral styling) -> RAW latents."""
    import intonation as I
    z = (raw - eng.emb_mean) / eng.emb_std
    audio = eng._decode_all([r.reshape(1, -1) for r in z]).astype(np.float64)
    sr = eng.sample_rate
    if eq:
        import voice_eq as Q
        audio = Q.apply(audio, sr, eq_filter(eng, voice)).astype(np.float64)
    if reg or rng != 1.0:
        f0, hop = I.yin(audio, sr)
        v = f0 > 0
        med = np.median(f0[v])
        st = np.where(v, 12 * np.log2(np.where(v, f0, med) / med), 0.0)
        sh = reg + (rng - 1.0) * np.clip(st, -8, 8)
        ts = np.arange(len(audio)) / hop
        fr = np.arange(len(f0))
        f0s = np.where(np.interp(ts, fr, v.astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
        ratio = 2.0 ** (np.interp(ts, fr, sh) / 12.0)
        audio = I.psola(audio, sr, f0s, ratio).astype(np.float64)
    if tempo != 1.0:
        f0, hop = I.yin(audio, sr)
        ts = np.arange(len(audio)) / hop
        fr = np.arange(len(f0))
        f0s = np.where(np.interp(ts, fr, (f0 > 0).astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
        audio, _ = I.psola_v2(audio, sr, f0s, np.ones(len(audio)), beta=np.full(len(audio), 1.0 / tempo))
    audio = audio * 10 ** (gain_db / 20)
    pk = np.abs(audio).max()
    if pk > 0.98:
        audio *= 0.98 / pk
    n = len(audio) // eng.frame_samples * eng.frame_samples
    lat = eng.s_enc.run(None, {"audio": audio[:n].astype(np.float32)[None, None, :]})[0][0]
    return lat.astype(np.float32)


def measure(x: np.ndarray, sr: int) -> dict:
    import intonation as I
    f0, hop = I.yin(x, sr)
    v = f0 > 0
    e = 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, np.ones(480) / 480, mode="same") + 1e-12)
    sp = e > e.max() - 35
    out = {"dur": round(len(x) / sr, 3), "level_db": round(float(10 * np.log10(np.mean(x[sp] ** 2) + 1e-12)), 2)}
    if v.sum() >= 10:
        st = 12 * np.log2(f0[v] / np.median(f0[v]))
        out.update(f0_med=round(float(np.median(f0[v])), 2),
                   range_st=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 3))
    return out


def main() -> int:
    import soundfile as sf
    import harness
    import frontend
    from core import load_set

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="style1")
    ap.add_argument("--voices", default="male_hello_clean,female_narration_clean,male_news")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--set", default=str(ROOT / "research/sets/prosody_set.tsv"))
    ap.add_argument("--cases", default="dev-st-exc")
    a = ap.parse_args()

    box = harness.Engine()
    eng = box.eng
    eng.prompt_ext_en = False
    eng.max_attempts = 1
    eng.level_target = None           # keep each take's own level (single-chunk sentences)
    eng.intonation = 0.0
    base_dir = ROOT / "output/research/voice_ext_m"
    out = ROOT / "output/research" / a.out
    cases = load_set(a.set)
    if a.cases == "dev-st-exc":
        cases = [c for c in cases if c["cat"] in ("st", "exc") and int(c["id"][1:3]) % 2 == 1]
    voices = a.voices.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for var in a.variants.split(","):
        d = out / "ext" / var
        d.mkdir(parents=True, exist_ok=True)
        for v in voices:
            raw = np.load(base_dir / f"{v}.ext.npy").astype(np.float32)
            if VARIANTS[var] is not None:
                raw = restyle(eng, raw, *VARIANTS[var], voice=v)
            np.save(d / f"{v}.ext.npy", raw)
        eng.prompt_ext_dir = str(d)
        wd = out / var
        wd.mkdir(parents=True, exist_ok=True)
        for c in cases:
            for v in voices:
                for s in seeds:
                    audio, _ = frontend.render(eng, c["text"], box.voice(v), seed=s)
                    name = f"{c['id']}__{v}__s{s}.wav"
                    sf.write(wd / name, audio, eng.sample_rate)
                    r = {"variant": var, "case": c["id"], "cat": c["cat"], "voice": v, "seed": s,
                         **measure(audio, eng.sample_rate)}
                    rows.append(r)
            print(var, c["id"], flush=True)
        (out / "measures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # summary: per variant x voice, mean f0 shift (st) vs "ext", range, level
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["variant"], r["voice"])].append(r)
    ref = {(r["case"], r["voice"], r["seed"]): r for r in rows if r["variant"] == "ext"}
    for (var, v), rs in sorted(by.items()):
        d_st = [12 * np.log2(r["f0_med"] / ref[(r["case"], v, r["seed"])]["f0_med"]) for r in rs
                if "f0_med" in r and (r["case"], v, r["seed"]) in ref and "f0_med" in ref[(r["case"], v, r["seed"])]]
        d_db = [r["level_db"] - ref[(r["case"], v, r["seed"])]["level_db"] for r in rs
                if (r["case"], v, r["seed"]) in ref]
        rg = [r["range_st"] for r in rs if "range_st" in r]
        print(f"{var:4s} {v:24s} dF0 {np.mean(d_st):+.2f} st  range {np.mean(rg):.2f}  dLevel {np.mean(d_db):+.2f} dB  n={len(rs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
