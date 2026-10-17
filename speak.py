"""Minimal wrapper around the ONNX Persian TTS engine (no torch).

    python speak.py "سلام، حال شما چطور است؟"
    python speak.py "متن" -v male_news -o out.wav --pace 1.1 --pack

    from speak import PersianTTS
    tts = PersianTTS()                      # loads the ONNX graphs once
    tts.say("سلام دنیا", "output/hi.wav")

Handles Persian, mixed Persian+English and pure English text. English is
rendered with the Persian phoneme inventory (see en_phonemes.py) — the model
has no /th/, /w/ or /ng/, so English always sounds Persian-accented.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "scripts"))

# Measured intelligibility (whisper transcript vs target text, n=12 runs):
#   female_narration 0.82 | male_hello_clean 0.78 | male_news 0.76 | male_hello 0.51
# male_hello.wav opens with a loud isolated burst that derails the model — it
# drops words or speaks the reference clip's own sentence in ~2/3 of runs.
# male_hello_clean.wav is the same voice with that burst trimmed off.
# female_narration_clean.wav (2026-10-11) is the 2 s crop «فهمه و آزاده کجا
# رفتن» of female_narration: the full clip's own sentence surfaced inside
# 4/66 harness renders, the crop in 0/66 (eval/harness.py, CER 9.2 → 7.9 %).
VOICES = {"female_narration_clean", "female_narration", "male_news",
          "male_hello_clean", "male_hello"}
DEFAULT_VOICE = "female_narration_clean"


# kept under the old name: scripts/server.py imports it from here
from frontend import MixedG2P as _MixedG2P  # noqa: E402


class PersianTTS:
    def __init__(self, voice=DEFAULT_VOICE, seed=None):
        from tts_onnx import OnnxTts
        from g2p_onnx import OnnxG2P
        self.engine = OnnxTts(seed=seed)
        self.engine._g2p = _MixedG2P(OnnxG2P(self.engine.dir))
        self.voice = voice
        self.sample_rate = self.engine.sample_rate

    def _voice_path(self, voice):
        v = voice or self.voice
        p = Path(v)
        return p if p.suffix else BASE / "voices" / f"{v}.wav"

    def phonemes(self, text):
        """Text (any script) -> the phoneme string the model will speak."""
        import frontend
        return " ".join(self.engine._g2p.phonemise(s.text)
                        for s in frontend.speakable(text)).strip()

    def say(self, text, out="output/out.wav", voice=None, pace=1.0,
            pack=False, seed=None):
        """Text -> WAV file. Returns the output path."""
        import numpy as np, soundfile as sf
        audio = self._render(text, str(self._voice_path(voice)), pace, pack, seed)
        out = Path(out)
        if not out.is_absolute():
            out = BASE / out
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, self.sample_rate)
        return str(out)

    def say_phonemes(self, phonemes, out="output/out.wav", voice=None,
                     pace=1.0, seed=None):
        """Speak a phoneme string directly (manual pronunciation fixes)."""
        import numpy as np, soundfile as sf
        if seed is not None:
            self.engine.rng = np.random.default_rng(seed)
        audio = self.engine.synthesize(phonemes, str(self._voice_path(voice)),
                                       pace=pace)
        out = Path(out)
        if not out.is_absolute():
            out = BASE / out
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, self.sample_rate)
        return str(out)

    def _render(self, text, voice, pace, pack, seed):
        """The chatbot's path (scripts/frontend.py): digits/dates/units/
        currency → words in the sentence's language, then every sub-chunk of
        every sentence in one parallel batch. Always the TEXT branch — the
        engine's own synthesize_text sends text with no Persian character to
        its PHONEME branch, and English came out as phoneme letters."""
        import frontend
        audio, _ = frontend.render(self.engine, text, voice, seed=seed,
                                   pace=pace, pack=pack)
        return audio


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Persian/English text -> speech (WAV)")
    ap.add_argument("text", nargs="?", default="سلام، حال شما چطور است؟")
    ap.add_argument("-o", "--out", default="output/out.wav")
    ap.add_argument("-v", "--voice", default=DEFAULT_VOICE,
                    help=f"builtin: {', '.join(sorted(VOICES))} — or a WAV path (<=5s)")
    ap.add_argument("--pace", type=float, default=1.0, help="0.6-1.5")
    ap.add_argument("--pack", action="store_true", help="fluent mode (fewer comma pauses)")
    ap.add_argument("--seed", type=int, default=None, help="reproducible output")
    ap.add_argument("--phonemes", action="store_true",
                    help="treat the input as a phoneme string, not text")
    a = ap.parse_args()

    tts = PersianTTS(voice=a.voice, seed=a.seed)
    if a.phonemes:
        print(tts.say_phonemes(a.text, a.out, pace=a.pace, seed=a.seed))
    else:
        print("phonemes:", tts.phonemes(a.text))
        print(tts.say(a.text, a.out, pace=a.pace, pack=a.pack, seed=a.seed))


if __name__ == "__main__":
    main()
