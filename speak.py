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
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "scripts"))

# Measured intelligibility (whisper transcript vs target text, n=12 runs):
#   female_narration 0.82 | male_hello_clean 0.78 | male_news 0.76 | male_hello 0.51
# male_hello.wav opens with a loud isolated burst that derails the model — it
# drops words or speaks the reference clip's own sentence in ~2/3 of runs.
# male_hello_clean.wav is the same voice with that burst trimmed off.
VOICES = {"female_narration", "male_news", "male_hello_clean", "male_hello"}
DEFAULT_VOICE = "female_narration"


class _MixedG2P:
    """OnnxG2P + English. Same interface, so the engine's punctuation-aware
    phrase planning is reused unchanged. English words resolvable by CMUdict
    are phonemised directly; anything else is left in the text for the
    engine's own Latin->Persian transliteration."""

    # a leftover segment with no letters/digits is punctuation debris; the
    # G2P raises ValueError("normalisation emptied the text") on it
    _CONTENT = re.compile(r"[\u0600-\u06FF A-Za-z0-9\u06F0-\u06F9]".replace(" ", ""))

    def __init__(self, inner):
        self.inner = inner

    def phonemise(self, text, keep_ezafe=False):
        import en_phonemes
        out = []
        for kind, s in en_phonemes.split(text):
            if kind == "en":
                out.append(s)
            elif self._CONTENT.search(s):
                p = self.inner.phonemise(s, keep_ezafe=keep_ezafe)
                if p:
                    out.append(p)
        return " ".join(out)


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
        from tts_onnx import split_sentences
        return " ".join(self.engine._g2p.phonemise(s)
                        for s in split_sentences(text)).strip()

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
        # Mirrors OnnxTts.synthesize_text's TEXT branch, but always takes it.
        # The engine picks the branch by looking for a Persian character
        # (tts_onnx.py:474), so pure-English text falls through to the
        # PHONEME branch and is spoken as phoneme letters — mush.
        # It also drops seed= on that branch (:492), so seed the rng here.
        import numpy as np
        from tts_onnx import (split_sentences, plan_phrases, pack_phrases,
                              SENTENCE_GAP)
        if seed is not None:
            self.engine.rng = np.random.default_rng(seed)
        sentences = []
        for sent in split_sentences(text):
            plan = plan_phrases(sent, self.engine._g2p, self.engine.sp)
            if pack:
                plan = pack_phrases(plan)
            if plan:
                sentences.append(plan)
        if not sentences:
            return np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(SENTENCE_GAP * self.sample_rate), np.float32)
        parts = []
        for plan in sentences:
            parts.append(self.engine.synthesize(plan, voice, pace=pace))
            parts.append(silence)
        return np.concatenate(parts[:-1]) if len(parts) > 1 else parts[0]


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
