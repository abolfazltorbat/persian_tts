# Pure-ONNX inference engine for pocket-tts-farsi-v2.
#
# The WHOLE pipeline — Persian text -> phonemes (G2P) -> speech — runs on
# onnxruntime + numpy + sentencepiece + pure-python helpers. NO torch.
# Persian text is auto-detected (Arabic-script codepoints) and phonemised
# with the ONNX G2P (scripts/g2p_onnx.py); phoneme strings pass through.
#
# Package: model/onnx/ (see manifest.json)
#
# CLI:
#   python scripts/tts_onnx.py "salAm hAle SomA Cetor ?ast" voices/male_hello.wav out.wav
import argparse
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _sibling(name: str):
    """A sibling module, imported the same way here and in the vendored copy."""
    import importlib
    if __package__:
        return importlib.import_module("." + name, __package__)
    return importlib.import_module(name)
import sentencepiece as spm

BASE = Path(__file__).resolve().parent.parent
PKG = BASE / "model" / "onnx"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
# phrase joints: after punctuation, and around parentheticals — "(...)" is a
# prosodic unit a reader sets off with small pauses on both sides
_PHRASE_SPLIT = re.compile(r"(?<=[،؛:—–,;:)])\s+|\s+(?=\()")


def session_providers() -> list[str]:
    """CPU by default. PARSIGO_DEVICE=cuda asks for the GPU (onnxruntime-gpu),
    filtered to what this onnxruntime build really has, so a CPU-only install
    keeps working when the variable is set by mistake."""
    if os.environ.get("PARSIGO_DEVICE", "cpu").lower() not in ("cuda", "gpu"):
        return ["CPUExecutionProvider"]
    # the CUDA EP prints a ScatterND advisory on EVERY graph run (dozens per
    # sentence) from the default logger; errors still show at severity 3
    ort.set_default_logger_severity(3)
    have = set(ort.get_available_providers())
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in have]


def split_sentences(text: str) -> list[str]:
    return [t.strip() for t in _SENTENCE_SPLIT.split(text.strip()) if t.strip()]


def split_phrases(sentence: str) -> list[str]:
    """Punctuation-aware phrase units. G2P discards punctuation, so a chunker
    working on phonemes alone cannot see where the writer paused (model card:
    'anything chunking the result cuts on token count alone'). Splitting the
    TEXT first keeps commas / dashes / colons as the pause points they are.
    Phrases with no letters/digits are debris — "— —" typed as two dashes
    leaves a lone "—", and a standalone "..." sentence is all punctuation —
    and would otherwise reach the G2P as an empty string ("normalisation
    emptied the text" -> HTTP 400 on perfectly valid Persian)."""
    return [t for t in _PHRASE_SPLIT.split(sentence.strip())
            if t.strip() and _letter_words(t)]


# A phrase ending in one of these is a lead-in ("سؤال اصلی:") whose whole
# purpose is the pause that follows it.
_STRONG_LEADIN = (":", "؛", "—", "–")

# pause lengths shared by the plan/pack layers (seconds): a strong lead-in
# (colon/semicolon/dash) gets a real stop, a comma a short breath; the
# sentence pause is inserted by the caller — server and CLI both read
# SENTENCE_GAP so the two paths cannot drift apart. These are what is
# INSERTED; every chunk already ends with OnnxTts.tail_keep (0.20 s) of its own
# decay and quiet, so a listener hears comma 0.24 s, colon 0.34 s, sentence
# 0.53 s, paragraph 0.88 s — the lengths of the 60 ms-tail design, kept when
# the tail grew (2026-10-24).
_STRONG_GAP = 0.12
_PHRASE_GAP = 0.02
SENTENCE_GAP = 0.31
# after the last sentence of a paragraph (a blank line in the text): readers
# stop longer between paragraphs than between sentences, and without it an
# answer's structure — intro, list, conclusion — ran together at one rhythm
PARAGRAPH_GAP = 0.66
# gap at a fresh-chunk boundary inside one sentence: none beyond the chunk's
# own tail (an unrelated knob from the _compress_pauses keep=0.12, which
# shortens leftover dead air)
_CHUNK_GAP = 0.0

# ---- chunking token budget: single source for the measured model limits ----
# One generation is one breath. Every chunk boundary is a pause, a pitch reset
# and an utterance-final fall in the middle of a sentence — the "reading a
# list" sound. The old budget (18 tokens; "21+ stop terminating") was measured
# with the defective male_hello prompt. Re-measured 2026-10-23 on three clean
# voices, whole sentences as ONE generation, Whisper-scored: 10–39 tokens end
# on EOS 100 % of the time at CER 4.7–7.7 %, no worse than short chunks. The
# real ceiling is the decoder (max_latents: 10.24 s of audio), reached ~40+.
# So a chunk is <= 30 tokens (+4 for a bound pair): most sentences are one
# generation, and mid-clause joins on eval/fluency.py fell 99 -> 12.
CHUNK_TOKENS = 30
CHUNK_HEADROOM = 4
# Pure-English sentences keep the short budget. They are read through mapped
# phonemes the model never trained on, and long runs of those went CER
# 4.9 -> 7.6 % while Persian went 6.8 -> 6.3 % and mixed 10.3 -> 9.8 %.
EN_CHUNK_TOKENS = 18
EN_CHUNK_HEADROOM = 2
# minimum viable chunk — below this the model reads badly ("har taklif rA"
# came out as garbage on whole voices)
MIN_CHUNK_WORDS = 3
MIN_CHUNK_TOKENS = 6


def _letter_words(tp: str) -> list[str]:
    """Words with at least one letter/digit — a lone "—" or "..." is not a
    word, and counting it as one breaks the phoneme-word-count check."""
    return [w for w in tp.split() if any(ch.isalnum() for ch in w)]


def merge_short_phrases(phrases: list[str], min_words: int = 2,
                        keep_standalone=None) -> list[str]:
    """Punctuation is a pause: a phrase with >= min_words words keeps its
    own pause slot (the reader asked for it with the comma/paren/colon).
    Only 1-word fragments merge forward — they read badly standalone and
    GE2P is noisy on 1-word inputs. Phrases ending in strong punctuation
    ("سؤال اصلی:", "…است —") stay standalone however short, and
    `keep_standalone(p)` lets the caller veto that for an unclean
    phonemisation. Must run on TEXT (before phonemisation): phonemes carry
    no punctuation."""
    def _protected(p: str) -> bool:
        return p.rstrip().endswith(_STRONG_LEADIN) and (
            keep_standalone is None or keep_standalone(p))

    out: list[str] = []
    for p in phrases:
        if out and len(_letter_words(out[-1])) < min_words and not _protected(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    if (len(out) >= 2 and len(_letter_words(out[-1])) < min_words
            and not _protected(out[-2])):
        # NB: pop BEFORE the store — `out[-2] += " " + out.pop()` would
        # resolve the store index AFTER the list shrank, silently
        # overwriting the phrase one slot earlier (dropped + duplicated
        # phrases, heard as missing words).
        last = out.pop()
        out[-1] += " " + last
    return out


# Text-layer prepositions that start a DETACHABLE adjunct ("برای X", "بدون
# X"). Tight-binding ones (از/با/در/به/تا — "از آن"، "به دست آورد"، "۱ تا ۲")
# are deliberately excluded: cutting before them reads far worse than wherever
# the phoneme packer lands.
_TEXT_PREPS = {"برای", "بدون", "درباره", "مثل"}


def split_long_phrase(tp: str, max_words: int = 9) -> list[str]:
    """A phrase longer than max_words needs 2+ chunks anyway (18-token
    budget); cut it before its last preposition (>=4 words before, >=3
    after) so the packer's inevitable break lands at a natural joint like
    '…سیم‌کشی | برای ساختن…' instead of inside 'یک شبکهٔ عصبی'."""
    words = tp.split()
    if len(words) <= max_words:
        return [tp]
    cut = None
    for i, w in enumerate(words):
        if w in _TEXT_PREPS and i >= 4 and len(words) - i >= 3:
            cut = i
    if cut is None:
        return [tp]
    return (split_long_phrase(" ".join(words[:cut]), max_words)
            + split_long_phrase(" ".join(words[cut:]), max_words))


# Light verbs in TEXT form (ZWNJ/space-stripped). A phrase must not START
# with one: the TTS model refuses to lead an utterance with a bare verbal
# enclitic and drops the word — "…جدیدی — | می‌شود شبکه را…" loses «می‌شود»
# on every voice — while «دیده می‌شود» at a chunk end reads fine. Kept in
# sync with the phoneme-side _LIGHT_VERBS (داد/دارد/می‌دهد families included).
_TEXT_LIGHT_VERBS = {
    "است", "هست", "هستم", "هستی", "هستیم", "هستید", "هستند",
    "بود", "بودم", "بودی", "بودیم", "بودید", "بودند", "باشد", "باشند",
    "شد", "شدم", "شدی", "شدیم", "شدید", "شدند", "شود", "شوند",
    "کرد", "کردم", "کردی", "کردیم", "کردید", "کردند", "کرده",
    "کنم", "کنی", "کند", "کنیم", "کنید", "کنند",
    "داد", "دادم", "دادی", "دادیم", "دادید", "دادند", "داده",
    "بدهد", "بدهند",
    "میشود", "میشوند", "میکند", "میکنند", "میکرد", "میکردند",
    "میداد", "میدادند", "میدهد", "میدهند",
    "دارد", "دارم", "داری", "داریم", "دارید", "دارند", "داشت",
    "میباشد", "میباشند",
}


def merge_leading_light_verbs(phrases: list[str]) -> list[str]:
    """Merge any phrase that starts with a light verb — or with the object
    marker «را», which clings to the previous phrase's noun — into the
    previous phrase, so the clitic follows its host and is actually spoken.
    The dash/colon pause that preceded it gives way to a small intra-chunk
    gap after the verb — a minor prosody cost against a dropped word."""
    out: list[str] = []
    for p in phrases:
        words = _letter_words(p)
        first = words[0].strip("«»()\"'.,;:!?،؛:-") if words else ""
        key = first.replace("\u200c", "")
        if out and (key in _TEXT_LIGHT_VERBS or key == "را"):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def plan_phrases(sentence: str, g2p, tokenizer=None) -> list[tuple[str, float]]:
    """One text sentence -> [(phonemes, gap_before_seconds), ...].
    Punctuation-aware splitting AND tiny-phrase merging happen here, where
    the punctuation is still visible. Gaps mirror how a reader delivers
    the text: 0.16 s at a comma (a short breath), 0.26 s after a strong
    lead-in (colon/semicolon/dash). The sentence-final pause (0.45 s) is
    inserted by the caller, outside the per-sentence synthesis.
    `tokenizer` (the engine's SentencePiece model) gates split_long_phrase
    on the real budget — phoneme TOKENS, not text words: a 10-word phrase
    of short words is ~17 tokens and fits one chunk, so splitting it only
    inserted a pause the writer never asked for. Without it, the old
    word-count heuristic applies."""
    cache: dict[str, bool] = {}

    def clean(tp: str) -> bool:
        # GE2P is noisy on 1-word inputs ("نکته:" -> "nokte nokte"): a
        # lead-in may only stand alone if its phoneme word count matches
        # its text word count; otherwise it merges forward like any other
        # tiny phrase (merged text gives the G2P the context it needs).
        # The text side is transliterated AND normalised first — "self-
        # recurrency" is one text word but two Persian words, and "۱۳۱۳:"
        # is one text word but five G2P words (numbers expand); the raw
        # count wrongly vetoed such lead-ins.
        if tp not in cache:
            from g2p_onnx import transliterate_text  # puts model/v2 on sys.path
            from normalize_fa import normalize_for_model
            want = len(_letter_words(
                normalize_for_model(transliterate_text(tp))
                .replace("؟", "").replace("?", "").replace(":", "")))
            ph = g2p.phonemise(tp, keep_ezafe=True)
            cache[tp] = bool(ph) and len(ph.split()) == want
        return cache[tp]

    merged = merge_short_phrases(split_phrases(sentence), keep_standalone=clean)
    merged = merge_leading_light_verbs(merged)
    out: list[tuple[str, float]] = []
    prev_tp: str | None = None
    for tp in merged:
        ph_tp = g2p.phonemise(tp, keep_ezafe=True)
        if (tokenizer is not None and ph_tp
                and len(tokenizer.encode(ph_tp, out_type=int)) <= CHUNK_TOKENS + CHUNK_HEADROOM):
            parts = [tp]          # fits one chunk even with bound-pair slack
        else:
            parts = split_long_phrase(tp)
        for q in parts:
            ph = ph_tp if q == tp else g2p.phonemise(q, keep_ezafe=True)
            if not ph:
                continue
            strong = prev_tp is not None and prev_tp.rstrip().endswith(_STRONG_LEADIN)
            out.append((ph, _STRONG_GAP if strong else _PHRASE_GAP))
            prev_tp = q
    return out


def pack_phrases(plan: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """PACK mode (the UI's «یکپارچه» option): merge adjacent phrases into
    one breathing unit unless a strong lead-in (gap 0.26 — colon/semicolon/
    dash) demands its pause. The comma pauses give way to the chunker's
    shorter joints, so the sentence flows in fewer, longer breaths — the
    model ends every chunk with its own sentence-final fall, so fewer
    chunks mean fewer artificial sentence-ends. Split mode (the plan as-is)
    keeps every punctuation pause."""
    groups: list[tuple[str, float]] = []
    for ph, gap in plan:
        if groups and gap != _STRONG_GAP:
            groups[-1] = (groups[-1][0] + " " + ph, groups[-1][1])
        else:
            groups.append((ph, gap))
    return groups


def trim_hot_onset(audio: np.ndarray, sr: int, head_ms: int = 300,
                   thresh_db: float = 4.0,
                   min_keep_s: float = 1.5) -> tuple[np.ndarray, int]:
    """Drop a hot opening from a reference-voice prompt. -> (audio, dropped_ms)

    A prompt whose first syllable is much louder than the rest of the sample
    makes the model replay that onset instead of the first word of a chunk.
    With voices/male_hello.wav (first 300 ms = +9.3 dB over the sample's own
    speech level) the 4-token tail chunk «شبکه شکننده میشود» opened with a
    280 ms burst: peak 1.39 (past full scale), +8 dB over the chunk's own body
    and 0.86-correlated with the prompt's first 250 ms — so the word «شبکه» is
    masked and the peak guard ducks the whole render by ~3 dB. The other two
    builtin references start with ~250 ms of near-silence (-48 / -36 dB) and
    never do this. Measured remedy: drop the leading 300 ms (four seeds and
    the full sentence come out clean; head peak 1.385 -> 0.025, head rms
    9 -> 27 dB below the chunk body).

    The attack is located relative to the SPEECH START, not to t=0: uploads
    routinely carry leading silence, and one real one (a clip opening with
    ~0.75 s of digital silence and then a full-scale clipped «سلام» at +6-9 dB
    over its own body) sailed past the old first-300 ms window untrimmed —
    every chunk then opened with that greeting while the rest of the render
    truncated (3.09 s vs 4.22 s for the same sentence+seed; head-vs-burst
    spectral cos 0.83 vs the 0.46 healthy baseline). Two fire paths, because
    the burst shapes differ (measured over all five current references):

    - attack MEAN well above the body (male_hello +5.8 dB, male_news +6.5 dB):
      the syllable sustains, the 300 ms window carries it;
    - the file CLIPS (peak >= 0.99) while the attack pokes above the body
      (the upload: peak 1.00, attack max +8.2 dB but mean only -0.1 dB — its
      ~150 ms burst is diluted by the body inside the window). Clipping is
      the tell: healthy references peak at 0.37-0.79, and a normal consonant
      transient can spike a frame as hard as a clipped burst does (59da's
      attack max is +8.8 dB at peak 0.37 and it is perfectly fine).

    When a path fires, everything from t=0 through attack-end is dropped —
    the leading silence goes with it, which also hands the encoder a denser
    prompt. A prompt that opens at a normal level is returned untouched, so
    this costs nothing for voices that are already fine. Prompts too short
    to lose the attack keep it (min_keep_s): a 0.9 s prompt would hurt the
    clone more than the leak."""
    if len(audio) < sr:                       # too short to judge
        return audio, 0
    h = max(1, int(0.01 * sr))
    nf = len(audio) // h
    peak = float(np.abs(audio).max())
    if nf < 8 or peak < 1e-6:
        return audio, 0
    frames = np.sqrt(np.array([(audio[i * h:(i + 1) * h] ** 2).mean()
                               for i in range(nf)], dtype=np.float64))
    loud = frames > 0.02 * peak               # frames carrying real audio
    speech = frames[loud]
    if not len(speech):
        return audio, 0
    typical = float(np.median(speech))
    onset = int(np.argmax(loud))              # first frame with real audio
    attack = frames[onset : onset + max(1, int(head_ms / 10))]
    loud_enough = typical * 10 ** (thresh_db / 20)
    hot = (float(np.sqrt((attack ** 2).mean())) > loud_enough
           or (peak >= 0.99 and float(attack.max()) > loud_enough))
    if not hot:
        return audio, 0
    drop = onset * h + int(head_ms / 1000 * sr)
    if len(audio) - drop < min_keep_s * sr:
        return audio, 0
    out = audio[drop:].copy()
    f = max(1, int(0.02 * sr))                # the new start may be mid-wave
    if len(out) > 2 * f:
        out[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    return out, drop * 1000 // sr


class OnnxTts:
    # voice-prompt KV cache budget: ~19 MB per entry -> ~76 MB held
    VOICE_MEMO_MAX = 8

    def __init__(self, pkg_dir=PKG, seed=None):
        # seed=None draws OS entropy: every run differs, so the README's
        # "rebuild a bad generation" advice actually works. Pass an int
        # (CLI --seed) to reproduce a run exactly.
        self.dir = Path(pkg_dir)
        man = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        c = man["constants"]
        self.ldim, self.dim = c["ldim"], c["dim"]
        self.L, self.H, self.D, self.CAP = c["layers"], c["heads"], c["dim_per_head"], c["cache_capacity"]
        self.sample_rate = c["sample_rate"]
        self.steps_per_latent = c["mimi_steps_per_latent"]
        self.temp = c["temp"]
        self.tps_est = c["tokens_per_second_estimate"]
        self.gen_pad = c["gen_seconds_padding"]
        self.frame_rate = c["frame_rate"]
        self.frame_samples = int(round(self.sample_rate / self.frame_rate))  # 1920

        w = np.load(self.dir / "weights.npz")
        self.lut = w["lut_weight"]
        self.spk_proj = w["speaker_proj"]          # [dim, ldim]
        self.bos_voice = w["bos_before_voice"][0]  # [1, dim]
        self.emb_std, self.emb_mean = w["emb_std"], w["emb_mean"]

        self.sp = spm.SentencePieceProcessor(model_file=str(BASE / "model" / "v2" / "tokenizer_ph.model"))
        self.rng = np.random.default_rng(seed)
        # voice prompt KV caches, memoised by (path, mtime): encoding a
        # reference voice costs ~0.5 s and synthesis runs per SENTENCE —
        # a 6-sentence paragraph paid it 6 times. Entries are ~19 MB each,
        # so VOICE_MEMO_MAX caps the cache at ~150 MB (LRU; a voice can hold
        # three: extended, plain for English, and one prosody-styled prompt).
        self._voice_memo: dict = {}
        self._voice_raw: dict = {}      # id(voice cache) -> raw prompt latents
        self._carrier_memo: dict = {}   # id(voice cache) -> (carrier tokens, latents)
        self._ext_lock = threading.Lock()  # prompt extensions are built once per voice
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # chunks generate in parallel (4 workers); the models are small and
        # batch-1 steps barely use extra threads — cap per-session threads
        # to avoid 4x oversubscription on an 8-core box
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        prov = session_providers()
        self.s_flow = ort.InferenceSession(str(self.dir / "flow_lm_step.onnx"), opts, providers=prov)
        self.s_enc = ort.InferenceSession(str(self.dir / "mimi_encoder.onnx"), opts, providers=prov)
        self.s_dec = ort.InferenceSession(str(self.dir / "mimi_decoder_step_kv.onnx"), opts, providers=prov)

        # decoder state layout: inputs s0..sN follow the export's dec_keys
        # order; npz keys carry "module.state_key" so we can find the caches.
        self.dec_in = [i.name for i in self.s_dec.get_inputs()]
        self.dec_out = [o.name for o in self.s_dec.get_outputs()]
        init = np.load(self.dir / "decode_state_init.npz")
        self.dec_init = [init[k] for k in init.files]  # npz order == s0..sN
        # the decoder's KV cache is linear (2048 steps, 16 per latent): one
        # generation can decode at most 128 latents = 10.24 s of audio, and a
        # longer one dies in _decode_all with a broadcast error
        dec_steps = min(init[k].shape[2] for k in init.files if k.endswith(".cache"))
        self.max_latents = dec_steps // self.steps_per_latent - 1
        n_state = len(self.dec_in) - 1
        self.cache_slots = [i for i, k in enumerate(init.files) if k.endswith(".cache")]
        self.small_slots = [i for i in range(n_state) if i not in self.cache_slots]
        self.dec_out_small = [i for i, n in enumerate(self.dec_out) if n.endswith("o") and n != "audio"]
        self.dec_out_kv = [i for i, n in enumerate(self.dec_out) if n.startswith("kv")]

    # ------------------------------------------------------------------
    def _flow_step(self, seq, text_emb, offset, noise, cache):
        S = seq.shape[1] + text_emb.shape[1]
        if offset + S > self.CAP:
            raise RuntimeError(
                f"flow cache overflow: offset {offset} + {S} new tokens exceeds "
                f"capacity {self.CAP} — the text chunk is too long for one "
                f"generation pass (max {CHUNK_TOKENS + CHUNK_HEADROOM} phoneme tokens per chunk)")
        lat, eos, kv = self.s_flow.run(None, {
            "sequence": seq, "text_emb": text_emb,
            "offset": np.array(offset, dtype=np.int64),
            "noise": noise, "cache": cache})
        kv = kv.reshape(self.L, 2, -1, self.H, self.D)
        S = kv.shape[2]
        # write in place: every caller owns its cache exclusively (the
        # shared voice cache is copied per attempt in _generate_with_retry)
        # — a defensive copy here cost ~5.4 ms x ~45 steps per chunk
        cache[:, :, offset : offset + S] = kv
        return lat, eos, cache, offset + S

    def voice_cache(self, wav_path, extend: bool = True, style: str | None = None):
        """Voice prompt -> seeded flow KV cache. Pure ONNX + numpy.
        Memoised by (path, mtime): the returned cache is SHARED and must
        never be written to — callers copy it before generating (they do:
        _generate_with_retry copies per attempt). `extend=False` gives the
        plain clip even when prompt self-extension is on (English sentences,
        and building the extension itself) — without touching shared state,
        so concurrent requests on one engine cannot see each other's mode.
        `style` names an entry of `prompt_styles`: the extension is re-styled
        (register / range / level) before it conditions the model."""
        import soundfile as sf
        from scipy.signal import resample_poly

        p = Path(wav_path)
        style = style if (extend and style and self.prompt_ext_dir and self.prompt_styles
                          and style in self.prompt_styles) else None
        key = (str(p.resolve()), p.stat().st_mtime, self.prompt_ext_dir if extend else None,
               self.prompt_ext_order, self.prompt_ext_min_s,
               (style, tuple(self.prompt_styles[style])) if style else None)
        hit = self._voice_memo.pop(key, None)
        if hit is not None:
            self._voice_memo[key] = hit  # refresh recency
            return hit

        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != self.sample_rate:
            g = np.gcd(int(sr), self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)
        # model card: prompts beyond 5 s are out of distribution (the
        # upload endpoint already trims; this guards the CLI/engine path)
        if len(audio) > 5 * self.sample_rate:
            audio = audio[: 5 * self.sample_rate]
        audio, dropped = trim_hot_onset(audio, self.sample_rate)
        if dropped:
            print(f"voice prompt {p.name}: dropped the first {dropped} ms — a "
                  f"hot opening syllable gets replayed as a burst over the "
                  f"first word of every chunk")

        lat = self.s_enc.run(None, {"audio": audio[None, None, :]})[0]  # [1,T,ldim]
        if extend:
            lat = self._extend_prompt(p, lat, style)
        cond = (lat[0] @ self.spk_proj.T).astype(np.float32)            # [T,dim]
        text_emb = np.concatenate([self.bos_voice, cond])[None]         # [1,T+1,dim]

        cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
        _, _, cache, off = self._flow_step(
            np.zeros((1, 0, self.ldim), np.float32), text_emb, 0,
            np.zeros((1, self.ldim), np.float32), cache)
        self._voice_memo[key] = (cache, off)
        # the prompt's raw latents, for continuation priming (prompt = clip + tail)
        self._voice_raw[id(cache)] = lat[0].astype(np.float32)
        while len(self._voice_memo) > self.VOICE_MEMO_MAX:
            old, _ = self._voice_memo.pop(next(iter(self._voice_memo)))
            self._voice_raw.pop(id(old), None)
            self._carrier_memo.pop(id(old), None)
        return cache, off

    #: Prompt self-extension (research/README.md, study B; None = off): a
    #: directory of `<voice stem>.ext.npy` files — RAW latents of the model's
    #: own reading of a neutral sentence in that voice (research/prep_voice_ext.py).
    #: The prompt becomes clip + extension, cut to the trained 5 s window.
    prompt_ext_dir: str | None = None
    #: "ref+ext" (clip first) or "ext+ref" (clip last, next to the take)
    prompt_ext_order: str = "ref+ext"
    #: apply the (Persian) extension to English sentences too
    prompt_ext_en: bool = True
    #: seconds of extension guaranteed even for a long clip (its start is cut)
    prompt_ext_min_s: float = 0.0

    #: Prosody-styled prompts (research/PROGRESS.md, study D; None = off):
    #: name -> (register st, range scale, level dB). The voice prompt carries
    #: register, pitch range and level into the take — a re-styled extension
    #: (decoded, TD-PSOLA, re-encoded; the OUTPUT is never processed) makes the
    #: model itself speak higher and livelier. frontend.render picks a style
    #: per sentence from its written type (intonation.STYLE_OF_KIND).
    prompt_styles: dict | None = None
    #: a high voice (extension median F0 above this) gets only this share of a
    #: style's register shift: +4 st on the female voice cost 0.4 UTMOS on
    #: exclamations, +2 st 0.12 (research/PROGRESS.md, study D)
    style_high_voice_hz: float = 165.0
    style_high_voice_reg: float = 0.5
    VOICE_STYLE_MEMO_MAX = 16

    #: Sentence-type intonation (research/PROGRESS.md, study C; 0 = off): the
    #: model never sees punctuation, so «رفتی؟» renders exactly like «رفتی.».
    #: frontend.join restores the contour per sentence from its written mark —
    #: final rise for yes/no questions, no final fall for wh-questions and
    #: exclamations (scripts/intonation.py). The value scales every shift.
    intonation: float = 0.0

    #: the neutral sentence the model reads to extend a prompt: «امروز هوا کمی
    #: سرد است و باد می‌وزد.» (G2P output; chosen on the dev set, research/PROGRESS.md)
    PROMPT_EXT_PHONEMES = "?emruz havA kami sard ?ast va bAd mivazad"

    def prepare_extension(self, wav_path, out_dir, tries: int = 4,
                          phonemes: str | None = None, name: str | None = None) -> Path:
        """Build `<out_dir>/<voice stem>.ext.npy` for prompt self-extension,
        with no recogniser: the model reads the neutral sentence `tries` times
        from the plain clip (fixed seeds), the take of median speech length is
        kept (speech span + one frame of its own trailing quiet) as RAW latents.
        Run once when a voice is added; synthesis then only reads the file."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ph = phonemes or self.PROMPT_EXT_PHONEMES
        cache, off = self.voice_cache(wav_path, extend=False)
        takes = []
        for s in range(tries):
            lat, _, _ = self._generate_chunk(cache.copy(), off, ph, rng=np.random.default_rng(500 + s))
            idx = self._speech_frames(self._decode_all(lat), len(lat))
            if len(idx):
                z = np.concatenate([x.reshape(1, -1) for x in lat[max(0, int(idx[0]) - 1): int(idx[-1]) + 2]])
                takes.append((len(idx), z))
        if not takes:
            raise RuntimeError("prompt extension: no take carried speech")
        takes.sort(key=lambda t: t[0])
        z = takes[len(takes) // 2][1]
        f = out_dir / (name or f"{Path(wav_path).stem}.ext.npy")
        tmp = f.with_name(f.name + ".tmp.npy")
        np.save(tmp, (z * self.emb_std + self.emb_mean).astype(np.float32))
        os.replace(tmp, f)                      # a reader never sees half a file
        # cached prompts are stale now (ids key the raw/carrier side tables)
        self._voice_memo.clear()
        self._voice_raw.clear()
        self._carrier_memo.clear()
        return f

    def _ext_file(self, wav: Path) -> Path | None:
        """The extension for this clip: `<stem>.<content hash>.ext.npy`, built
        on first use (once per voice; ~4 short generations) — so a voice a user
        uploads gets one too, and an uploaded clip that happens to share a
        built-in's name can never pick up the built-in's extension. A legacy
        `<stem>.ext.npy` (the research directories) is used when present."""
        d = Path(self.prompt_ext_dir)
        digest = hashlib.sha1(wav.read_bytes()).hexdigest()[:12]
        keyed = d / f"{wav.stem}.{digest}.ext.npy"
        if keyed.is_file():
            return keyed
        legacy = d / f"{wav.stem}.ext.npy"
        if legacy.is_file():
            return legacy
        with self._ext_lock:                    # two first requests build it once
            if not keyed.is_file():
                try:
                    self.prepare_extension(wav, d, name=keyed.name)
                except Exception as exc:        # no extension is the old behaviour
                    print(f"prompt extension for {wav.name} not built: {exc}")
                    return None
        return keyed

    def _styled_ext(self, f: Path, ext: np.ndarray, style: str) -> np.ndarray:
        """The extension re-styled: RAW latents -> audio -> TD-PSOLA register /
        range shift and level -> Mimi encoder -> RAW latents. Memoised per
        (file, style); ~0.1-0.2 s once."""
        reg, rng_scale, gain_db = (float(v) for v in self.prompt_styles[style])
        key = (str(f), f.stat().st_mtime, style, reg, rng_scale, gain_db)
        memo = self.__dict__.setdefault("_style_memo", {})
        hit = memo.get(key)
        if hit is not None:
            return hit
        into = _sibling("intonation")
        z = (ext - self.emb_mean) / self.emb_std
        audio = self._decode_all([r.reshape(1, -1) for r in z]).astype(np.float64)
        sr = self.sample_rate
        if reg or rng_scale != 1.0:
            f0, hop = into.yin(audio, sr)
            v = f0 > 0
            if v.sum() >= 10:
                med = float(np.median(f0[v]))
                if med > self.style_high_voice_hz:
                    reg *= self.style_high_voice_reg
                st = np.where(v, 12 * np.log2(np.where(v, f0, med) / med), 0.0)
                sh = reg + (rng_scale - 1.0) * np.clip(st, -8, 8)
                ts = np.arange(len(audio)) / hop
                fr = np.arange(len(f0))
                f0s = np.where(np.interp(ts, fr, v.astype(float)) > 0.5, np.interp(ts, fr, f0), 0.0)
                audio = into.psola(audio, sr, f0s, 2.0 ** (np.interp(ts, fr, sh) / 12.0)).astype(np.float64)
        audio = audio * 10 ** (gain_db / 20)
        pk = float(np.abs(audio).max()) if len(audio) else 0.0
        if pk > 0.98:
            audio *= 0.98 / pk
        n = len(audio) // self.frame_samples * self.frame_samples
        out = self.s_enc.run(None, {"audio": audio[:n].astype(np.float32)[None, None, :]})[0][0]
        out = out.astype(np.float32)
        memo[key] = out
        while len(memo) > self.VOICE_STYLE_MEMO_MAX:
            memo.pop(next(iter(memo)))
        return out

    def _extend_prompt(self, wav: Path, lat, style: str | None = None):
        if not self.prompt_ext_dir:
            return lat
        f = self._ext_file(wav)
        if f is None:
            return lat
        ext = np.load(f).astype(np.float32)
        if style:
            ext = self._styled_ext(f, ext, style)
        ref = lat[0]
        cap = int(5 * self.frame_rate)
        # a long clip leaves no room: keep at least prompt_ext_min_s of the
        # extension by dropping the clip's START (the prompt is "the audio
        # before the target", so its beginning is the part that matters least)
        keep_ext = min(len(ext), int(round(self.prompt_ext_min_s * self.frame_rate)))
        if len(ref) + keep_ext > cap:
            ref = ref[-(cap - keep_ext):] if cap > keep_ext else ref[:0]
        room = cap - len(ref)
        if room <= 0:
            return lat
        ext = ext[-room:]
        both = np.concatenate([ref, ext] if self.prompt_ext_order == "ref+ext" else [ext, ref])
        return both[None].astype(np.float32)

    def _prompt_state(self, raw):
        """RAW prompt latents [T, ldim] -> (flow KV cache, offset), the
        arithmetic of voice_cache without the audio front."""
        cond = (raw @ self.spk_proj.T).astype(np.float32)
        text_emb = np.concatenate([self.bos_voice, cond])[None]
        cache = np.zeros((self.L, 2, self.CAP, self.H, self.D), np.float32)
        _, _, cache, off = self._flow_step(
            np.zeros((1, 0, self.ldim), np.float32), text_emb, 0,
            np.zeros((1, self.ldim), np.float32), cache)
        return cache, off

    # ------------------------------------------------------------------
    def _decode_all(self, latents):
        st = [a.copy() for a in self.dec_init]
        out = []
        off = 0
        for lt in latents:
            un = (lt.reshape(1, 1, self.ldim) * self.emb_std + self.emb_mean).astype(np.float32)
            feeds = {"latent": un}
            for i in range(len(st)):
                feeds[self.dec_in[1 + i]] = st[i]
            res = self.s_dec.run(None, feeds)
            out.append(res[0])
            for j, o in enumerate(self.dec_out_small):
                st[self.small_slots[j]] = res[o]
            for j, o in enumerate(self.dec_out_kv):
                st[self.cache_slots[j]][:, :, off : off + self.steps_per_latent] = res[o]
            off += self.steps_per_latent
        return np.concatenate(out, axis=2)[0, 0]

    def synthesize_text(self, text, voice_wav, seed=None, pace=1.0, mode="split"):
        """Persian text OR phonemes -> audio. Persian is auto-detected;
        Persian text is split into sentences and punctuation-delimited phrases
        BEFORE phonemisation so pauses land where the writer put them.
        mode="pack" merges comma-delimited phrases into longer breathing
        units (pack_phrases) — punctuation pauses survive only at strong
        lead-ins (colon/semicolon/dash)."""
        if any("\u0600" <= ch <= "\u06FF" for ch in text):
            from g2p_onnx import OnnxG2P

            if not hasattr(self, "_g2p"):
                self._g2p = OnnxG2P(self.dir)
            # per-sentence phrase plans: phrase gaps (0.16/0.26 s) stay
            # clearly shorter than sentence pauses (0.45 s), mirroring the
            # server — a reader breathes at a comma but stops at a period
            sentences = []
            for sent in split_sentences(text):
                plan = plan_phrases(sent, self._g2p, self.sp)
                if mode == "pack":
                    plan = pack_phrases(plan)
                if plan:
                    sentences.append(plan)
            print("phonemes:", " ".join(" ".join(p for p, _ in s) for s in sentences))
            parts, silence = [], np.zeros(int(SENTENCE_GAP * self.sample_rate), np.float32)
            for plan in sentences:
                parts.append(self.synthesize(plan, voice_wav, pace=pace))
                parts.append(silence)
            if not parts:  # text was all punctuation debris
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(parts[:-1]) if len(parts) > 1 else parts[0]
        return self.synthesize(text, voice_wav, seed=seed, pace=pace)

    # Phonetic prepositions: a break right before one starts a prepositional
    # phrase — one of the joints a reader breathes at.
    _PREPS = {"?az", "bA", "dar", "be", "barAye", "tA", "ruye", "bedune", "vase"}

    def chunk_phonemes(self, phonemes: str, lang: str = "fa") -> list[str]:
        """One phrase -> word-boundary chunks, each within the generation
        budget (CHUNK_TOKENS; EN_CHUNK_TOKENS for an English sentence). The
        ezafe marker "1" guides the breaks and is stripped from the result."""
        words = [w for w in phonemes.split() if w]
        if lang == "en":
            budget, cap = EN_CHUNK_TOKENS, EN_CHUNK_TOKENS + EN_CHUNK_HEADROOM
        else:
            budget, cap = CHUNK_TOKENS, CHUNK_TOKENS + CHUNK_HEADROOM
        return [c.replace("1", "") for c in self._chunk_balanced(words, budget, cap, lang)]

    # English joints, in en_phonemes' phoneme spellings. The Persian break
    # costs key on ezafe, «را» and Persian verbs, which an English sentence
    # has none of, so its breaks fell wherever the token count put them:
    # "…in twenty twenty | three and now…" — and a generation that starts
    # mid-number garbles its first words (en_numbers: 4 of 6 takes).
    _EN_NUMBER = {"ziro", "van", "tu", "tri", "for", "fAyv", "siks", "seven", "?eyt",
                  "nAyn", "ten", "?ileven", "tvelv", "tertin", "fortin", "fiftin",
                  "sikstin", "seventin", "?eytin", "nAyntin", "tventi", "terdi", "forti",
                  "fifti", "siksti", "seventi", "?eyti", "nAynti", "handred", "tAvzend",
                  "milyen", "bilyen", "trilyen", "poynt", "?o", "mAynes", "persent"}
    # articles, prepositions, conjunctions, possessives: never end a chunk
    _EN_DANGLING = {"de", "?ey", "?an", "?av", "tu", "?in", "?An", "?at", "for", "vid",
                    "bAy", "fram", "?intu", "?ebAvt", "?end", "?or", "bat", "mAy", "yor",
                    "hiz", "her", "?its", "?Aver", "der", "dis", "diz", "doz"}
    # a reader breathes BEFORE these
    _EN_BREAK_BEFORE = {"?end", "?or", "bat", "?in", "?An", "?at", "for", "vid", "bAy",
                        "fram", "tu", "dat", "hu", "bikoz", "so", "ven", "vAyl", "?if",
                        "den", "viC"}

    def _chunk_balanced(self, words: list[str], budget: int, cap: int,
                        lang: str = "fa") -> list[str]:
        """All of a phrase's chunk breaks chosen together (dynamic programming
        over word boundaries), instead of greedily.

        The greedy packer fills each chunk to the budget and hands the rest
        on, so whatever is left at the end is a runt: a 49-token phrase came
        out 14 + 14 + 19 + 2, and the 2-token «پیدا کنیم» was read alone —
        an isolated word with its own intonation after a pause, and the model
        is unreliable that short (two of three attempts failed QC). Chosen
        together, the same phrase is three even chunks.

        Cost = per chunk (one unit each, so a phrase is never split more than
        it must be; extra for a runt or for using the +2 headroom) + per break
        (the greedy rules as costs: never strand an ezafe head, never open a
        chunk with a light verb, «را» or a conjunction, never end on a
        dangling preposition; a joint right after a light verb or «را» is the
        one a reader would breathe at)."""
        n = len(words)
        if n <= 1:
            return [" ".join(words)] if words else []
        clean = [w.replace("1", "") for w in words]
        memo: dict[tuple[int, int], int] = {}

        def ntok(i: int, j: int) -> int:
            if (i, j) not in memo:
                memo[(i, j)] = len(self.sp.encode(" ".join(clean[i:j]), out_type=int))
            return memo[(i, j)]

        def chunk_cost(i: int, j: int) -> float:
            # 4 per chunk: more than any joint bonus, so a phrase that fits is
            # never split for a "nicer" joint, but less than a bad break, so
            # one extra chunk may buy a phrase out of «هر | کدام»
            t, nw = ntok(i, j), j - i
            c = 4.0
            if nw < n:
                if t < MIN_CHUNK_TOKENS or nw < MIN_CHUNK_WORDS:
                    c += 8.0                # runt
                elif t < 9:
                    c += 1.5                # short: reads as a fragment
            if t > budget:
                c += 0.75 * (t - budget)
            return c

        def break_cost(k: int) -> float:
            prev, nxt = words[k - 1], words[k]
            p, q = clean[k - 1], clean[k]
            if lang == "en":
                c = 0.0
                if p in self._EN_NUMBER and q in self._EN_NUMBER:
                    c += 6.0                # inside a number: "twenty | five"
                if p in self._EN_DANGLING:
                    c += 5.0                # "the | company", "in | twenty"
                elif q == "?av":
                    c += 2.0                # "of" completes its left word: "a lot | of"
                elif q in self._EN_BREAK_BEFORE:
                    c -= 1.0                # "…percent | in twenty twenty three"
                return c
            clause_end = self._is_finite_verb(p)
            c = 0.0
            if prev.endswith("1") or p.endswith("ye"):
                c += 6.0                    # ezafe head stranded («دادههای | جمع‌آوری‌شده»)
            if self._is_light_verb(q) or q == "rA":
                c += 6.0                    # «تشکیل | شده‌اند», «کتاب | را»
            if p in self._FUNCTION_WORDS or p in self._DETERMINERS:
                c += 5.0                    # dangling preposition / «هر | کدام»
            if q in self._CONJUNCTIONS and not clause_end:
                c += 3.0                    # «A و B»: «و» binds to its left operand
            if nxt.endswith("1"):
                c += 2.0                    # ezafe phrase detached from its host
            if p.endswith("e") and q.startswith("?"):
                # unmarked compound («qAbele | ?e?temAd») — weak: every vowel-
                # initial word starts with «?», so this also fires on ordinary
                # word pairs
                c += 1.0
            if q == "?az":
                c += 3.0                    # «از» completes its left word: «استفاده | از»
            elif clause_end or p == "rA":
                c -= 2.0                    # end of a clause («…کند | که…») / of the object NP
            elif q in self._PREPS:
                c -= 0.5                    # start of a prepositional phrase
            return c

        INF = float("inf")
        best = [INF] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                if ntok(i, j) > cap and j - i > 1:
                    break                   # longer spans only grow
                c = best[i] + chunk_cost(i, j) + (break_cost(i) if i else 0.0)
                if c < best[j]:
                    best[j], back[j] = c, i
        out, j = [], n
        while j > 0:
            out.append(" ".join(words[back[j]:j]))
            j = back[j]
        return out[::-1]

    #: Determiners and quantifiers: «هر کدام», «یک سیستم», «این روش» — they
    #: open a noun phrase and must not end a chunk without it.
    _DETERMINERS = {"har", "yek", "?in", "?An", "hame", "hameye", "Cand", "hic", "barxi"}

    _VERB_FORM = re.compile(r"^(?:mi|nemi|be|na|ne)\w{2,}(?:am|i|ad|im|id|and)$")

    # Light-verb stems with their prefixes and endings — the closed set that
    # forms compound verbs («تشکیل شده‌اند», «وجود دارد», «پیدا کنیم»). A fixed
    # list of forms missed the perfect ones (Sodeand, kardeand).
    _LV_FORM = re.compile(r"^(?:mi|nemi|na|ne|be)?(?:Sod|Sav|kard|kon|dAd|dAh|dAr|dASt"
                          r"|gereft|gir|zad|yAft|yAb|bud|bAS|nemud)e?(?:am|i|ad|im|id|and|ast|an)?$")

    def _is_light_verb(self, w: str) -> bool:
        return w in self._LIGHT_VERBS or bool(self._LV_FORM.match(w))

    def _is_finite_verb(self, w: str) -> bool:
        """A clause-final verb in GE2P phonemes: a light verb, or a prefixed
        (mi-/be-/na-) stem with a person ending — «besAzim», «mitavAnand».
        A break after one is the clause boundary a reader breathes at."""
        return self._is_light_verb(w) or w == "bAyad" or bool(self._VERB_FORM.match(w))

    # Persian function words that must not dangle at a chunk end: a
    # preposition/conjunction without its object makes the model pause after
    # it, which the listener hears as a strange mid-phrase stop. "beture"/
    # "besurate" ("به‌طور/به‌صورت X") always need the word they qualify.
    # "rA" is NOT here: it never dangles — it clings to the noun before it
    # and is handled as a left-binding clitic.
    _FUNCTION_WORDS = {
        "dar", "be", "?az", "tA", "va", "ke", "bA", "bedune",
        "age", "vali", "yA", "barAye", "vase", "dAr", "mi",
        "beture", "besurate",
    }
    # Left-binding words: a chunk must not START with one — "A va B" and
    # "ketAb rA" belong together, so the previous chunk's last word moves
    # down to keep the pair intact.
    _CONJUNCTIONS = {"va", "yA", "vali", "amA", "hattA", "ke", "rA"}
    # Persian light verbs complete the previous word's compound verb
    # ("Sekannde miSavad", "neSAn dAdand"); breaking right before one splits
    # the verb. NB: GE2P writes long-a as "A" — the dah-family is "dAd…",
    # never "dad…" (a lowercase typo there silently disables the rule).
    _LIGHT_VERBS = {
        "miSavad", "miSavand", "miSavam", "miSavid", "miSavim",
        "Savad", "Savand", "Sod", "Sodand", "Sodan",
        "mikonad", "mikonand", "mikonam", "mikonid", "mikonim",
        "kard", "karde", "konad", "konand",
        "dAd", "dAde", "dAdand", "dAdam", "dAdi", "dAdim", "dAdid",
        "dAhad", "dAhand", "midAd", "midAdand",
        "dArad", "dArand", "dAsht", "Ast", "?ast", "bud", "budand",
    }

    def synthesize(self, phonemes, voice_wav, seed=None, pace=1.0):
        """phonemes: a phoneme string OR a list of phrase strings. In list mode
        every phrase is a pause unit (punctuation-aware splitting is done in
        the text layer); a phrase over the token budget is sub-chunked. Each
        chunk is generated from the pristine voice state (mirrors production:
        copy_state=True — a continued 100+ position context is out of
        distribution and causes early EOS = dropped words). `pace`
        (0.6..1.5) time-stretches the final audio uniformly."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.synthesize_many([phonemes], voice_wav, pace=pace)[0]

    def _jobs_for(self, phonemes, lang: str = "fa") -> list[tuple[str, float]]:
        """One sentence's phrase plan -> generation units [(chunk, gap_before)]."""
        if isinstance(phonemes, str):
            phrases = [(phonemes, None)]
        else:
            phrases = [p if isinstance(p, tuple) else (p, None) for p in phonemes]
        phrases = [(p.strip(), g) for p, g in phrases if p and p.strip()]
        # NOTE: tiny-phrase merging is deliberately NOT done here. It needs
        # the original punctuation, so it lives in the text layer
        # (plan_phrases / merge_short_phrases); merging again here would
        # re-absorb a short colon lead-in that the caller kept standalone.
        jobs = []
        for pi, (phrase, pgap) in enumerate(phrases):
            for ci, chunk in enumerate(self.chunk_phonemes(phrase, lang)):
                if pi == 0 and ci == 0:
                    gap = 0.0
                elif ci == 0:
                    gap = pgap if pgap is not None else _PHRASE_GAP
                else:
                    gap = _CHUNK_GAP
                jobs.append((chunk, gap))
        return jobs

    #: Parallel generation units. Chunks are independent (each starts from
    #: the pristine voice state) and a batch-1 flow step barely saturates
    #: the cores, so they run on a thread pool; onnxruntime releases the GIL
    #: inside Session.run.
    workers = min(4, os.cpu_count() or 4)

    def synthesize_many(self, plans, voice_wav, pace=1.0, langs=None, seed=None, styles=None) -> list:
        """Several sentences at once -> one audio array per sentence.

        All generation units of ALL sentences go into one pool, so a
        multi-sentence request keeps every worker busy instead of waiting
        sentence by sentence (a 3-sentence paragraph used to run as three
        serial rounds of 2-3 parallel chunks). Each unit gets its own rng
        drawn from the engine rng — or from `seed`, which callers sharing
        one engine across threads pass so the take is reproducible — so
        results do not depend on scheduling. `styles` (one name or None per
        sentence) picks a prosody-styled prompt (`prompt_styles`)."""
        pace = float(np.clip(pace, 0.6, 1.5))
        voice_cache, voice_off = self.voice_cache(voice_wav)
        # English sentences may keep the plain clip (prompt_ext_en=False): the
        # extension is a Persian sentence, and on the bench it helped Persian
        # units while English ones drifted (research/PROGRESS.md)
        plain = None
        if self.prompt_ext_dir and not self.prompt_ext_en and langs and "en" in langs:
            plain = self.voice_cache(voice_wav, extend=False)

        # styled prompts are built here, before the worker threads start
        styled = {st: self.voice_cache(voice_wav, style=st)
                  for st in sorted({x for x in (styles or []) if x})
                  if self.prompt_ext_dir and self.prompt_styles and st in self.prompt_styles}

        def cache_for(lang, si=None):
            if plain is not None and lang == "en":
                return plain
            st = styles[si] if (styles and si is not None) else None
            return styled.get(st, (voice_cache, voice_off)) if st else (voice_cache, voice_off)
        jobs, owner = [], []
        for si, ph in enumerate(plans):
            for job in self._jobs_for(ph, langs[si] if langs else "fa"):
                jobs.append(job)
                owner.append(si)
        if not jobs:
            return [np.zeros(0, dtype=np.float32) for _ in plans]
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        seeds = rng.integers(0, 2**63, size=len(jobs))

        def run(i):
            lang = langs[owner[i]] if langs else "fa"
            vc, vo = cache_for(lang, owner[i])
            return self._generate_with_retry(
                vc, vo, jobs[i][0],
                rng=np.random.default_rng(int(seeds[i])), lang=lang,
                prime=self._short_prime(vc, vo, jobs[i][0], lang))

        if self.prime_tail_s > 0:
            audios = self._run_primed(jobs, owner, langs, seeds, voice_cache, voice_off,
                                      len(plans), cache_for=cache_for)
        else:
            audios = self._run_units(run, range(len(jobs)))
        if self.verifier is not None:
            audios = self._verify_and_repair(jobs, owner, audios, voice_cache,
                                             voice_off, langs, rng, len(plans),
                                             cache_for=cache_for)

        out = []
        for si in range(len(plans)):
            segments = [(audios[i], jobs[i][1]) for i in range(len(jobs)) if owner[i] == si]
            if not segments:
                out.append(np.zeros(0, dtype=np.float32))
                continue
            audio = self._stitch(segments)
            if abs(pace - 1.0) >= 0.03:
                from pedalboard import time_stretch

                audio = np.ascontiguousarray(audio, dtype=np.float32)
                audio = time_stretch(audio, self.sample_rate,
                                     stretch_factor=pace).reshape(-1)
            out.append(audio.astype(np.float32))
        return out

    def _run_units(self, run, indices) -> list:
        """run(i) for each index, on the worker pool; results in index order."""
        idx = list(indices)
        if len(idx) > 1 and self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(self.workers, len(idx))) as ex:
                return list(ex.map(run, idx))
        return [run(i) for i in idx]

    # ── continuation priming (research/README.md, study B; off by default) ──
    #: Short-unit carrier priming: a Persian unit of at most this many phoneme
    #: tokens is generated as the continuation of a carrier phrase whose
    #: latents (the model's own reading of it, made once per voice) are
    #: teacher-forced; the carrier's audio is cut away. 0 = off.
    prime_short_tokens: int = 0
    prime_carrier: str = "xob"
    #: frames of the carrier's trailing quiet kept before the unit starts
    prime_carrier_gap: int = 2
    #: Tail priming: a unit after the first of its sentence is conditioned on
    #: the last `prime_tail_s` seconds of the previous unit's speech — the
    #: model's own training cut (prompt = the audio before the cut, <= 5 s).
    #: 0 = off. Units of one sentence then run in order (sentences in parallel).
    prime_tail_s: float = 0.0
    #: keep the reference clip in front of the tail (prompt = clip + tail, <= 5 s)
    prime_tail_ref: bool = True
    #: also prime a sentence's first unit with the previous sentence's tail
    prime_across_sentences: bool = False

    def _speech_frames(self, audio, n):
        """Indices of the latent frames carrying speech (within 30 dB of the
        take's loudest frame)."""
        fs = self.frame_samples
        db = np.full(n, -120.0)
        for k in range(n):
            seg = audio[k * fs:(k + 1) * fs]
            if len(seg):
                db[k] = 20 * np.log10(max(1e-6, float(np.sqrt((seg.astype(np.float64) ** 2).mean()))))
        return np.where(db > db.max() - 30.0)[0] if n else np.zeros(0, int)

    def _carrier(self, voice_cache, voice_off):
        """(token ids, latents) of the carrier as this voice reads it: four
        fixed-seed takes, the median-length one, cut after its last speech
        frame + `prime_carrier_gap` frames of its own quiet."""
        key = id(voice_cache)
        hit = self._carrier_memo.get(key)
        if hit is not None:
            return hit
        takes = []
        for s in range(4):
            lat, _, _ = self._generate_chunk(voice_cache.copy(), voice_off, self.prime_carrier,
                                             rng=np.random.default_rng(1000 + s))
            idx = self._speech_frames(self._decode_all(lat), len(lat))
            if len(idx):
                takes.append((int(idx[-1]), lat))
        if not takes:
            return None
        takes.sort(key=lambda t: t[0])
        last, lat = takes[len(takes) // 2]
        z = np.concatenate([x.reshape(1, -1) for x in lat[: last + 1 + self.prime_carrier_gap]])
        out = (self.sp.encode(self.prime_carrier, out_type=int), z.astype(np.float32))
        self._carrier_memo[key] = out
        return out

    def _short_prime(self, voice_cache, voice_off, chunk, lang):
        if not self.prime_short_tokens or lang == "en":
            return None
        if len(self.sp.encode(chunk, out_type=int)) > self.prime_short_tokens:
            return None
        return self._carrier(voice_cache, voice_off)

    def _run_primed(self, jobs, owner, langs, seeds, voice_cache, voice_off, n_sent,
                    cache_for=None):
        """Tail priming: each sentence's units in order, every unit after the
        first conditioned on the previous unit's speech tail (+ the clip)."""
        raw_ref = self._voice_raw.get(id(voice_cache))
        if self.prime_across_sentences:
            groups = [list(range(len(jobs)))]
        else:
            groups = [[i for i in range(len(jobs)) if owner[i] == si] for si in range(n_sent)]
            groups = [g for g in groups if g]
        tail_n = max(1, int(round(self.prime_tail_s * self.frame_rate)))
        max_prompt = int(5 * self.frame_rate)
        audios = [None] * len(jobs)

        def run_group(g):
            prev = None
            for i in g:
                lang = langs[owner[i]] if langs else "fa"
                r = np.random.default_rng(int(seeds[i]))
                prime = None
                if prev is not None and len(prev):
                    tail = prev[-tail_n:]
                    if self.prime_tail_ref and raw_ref is not None:
                        raw = np.concatenate([raw_ref[: max(0, max_prompt - len(tail))], tail])
                    else:
                        raw = tail
                    cache, off = self._prompt_state(raw)
                else:
                    cache, off = cache_for(lang) if cache_for else (voice_cache, voice_off)
                    prime = self._short_prime(cache, off, jobs[i][0], lang)
                speech, (lat, audio) = self._generate_with_retry(
                    cache, off, jobs[i][0], rng=r, lang=lang, prime=prime, want_latents=True)
                audios[i] = speech
                idx = self._speech_frames(audio, len(lat))
                if len(idx):
                    z = np.concatenate([x.reshape(1, -1) for x in lat[: int(idx[-1]) + 1]])
                    prev = (z * self.emb_std + self.emb_mean).astype(np.float32)
                else:
                    prev = None

        self._run_units(lambda gi: run_group(groups[gi]), range(len(groups)))
        return audios

    def _verify_and_repair(self, jobs, owner, audios, voice_cache, voice_off,
                           langs, rng, n_sent, cache_for=None):
        """Transcribe each stitched sentence ONCE, charge the errors to the
        units that made them, regenerate only those, keep each unit's best
        take. A take can be loud, the right length and wrong — the reference
        clip's sentence replayed, a word dropped or swapped — and no signal
        check sees that; the recogniser does. A failing recogniser leaves the
        signal-checked takes in place: it may only ever improve a reading."""
        v = self.verifier
        stats = self.qc_stats
        best = list(audios)
        best_err = [float("inf")] * len(jobs)
        sentences = [[i for i in range(len(jobs)) if owner[i] == si] for si in range(n_sent)]
        todo = [si for si in range(n_sent) if sentences[si]]
        for rnd in range(self.repair_rounds + 1):
            bad = []
            for si in todo:
                idx = sentences[si]
                lang = langs[si] if langs else "fa"
                audio = self._stitch([(audios[i], jobs[i][1]) for i in idx])
                try:
                    total, per_unit = v.check_sentence(audio, self.sample_rate,
                                                       [jobs[i][0] for i in idx], lang)
                except Exception:
                    if stats is not None:
                        stats["asr_errors"] = stats.get("asr_errors", 0) + 1
                    continue
                if stats is not None:
                    stats["checks"] = stats.get("checks", 0) + 1
                for i, e in zip(idx, per_unit):
                    if e < best_err[i]:
                        best[i], best_err[i] = audios[i], e
                if total <= getattr(v, "sentence_ok", 0.0):
                    continue
                bad += [i for i in idx if best_err[i] > v.threshold]
            if not bad or rnd == self.repair_rounds:
                break
            if stats is not None:
                stats["repaired"] = stats.get("repaired", 0) + len(bad)
            fresh = rng.integers(0, 2**63, size=len(bad))
            seed_of = dict(zip(bad, fresh))

            def again(i):
                vc, vo = (cache_for(langs[owner[i]] if langs else "fa", owner[i]) if cache_for
                          else (voice_cache, voice_off))
                return self._generate_with_retry(
                    vc, vo, jobs[i][0], attempts=2,
                    rng=np.random.default_rng(int(seed_of[i])),
                    lang=langs[owner[i]] if langs else "fa")

            for i, a in zip(bad, self._run_units(again, bad)):
                audios[i] = a
            todo = sorted({owner[i] for i in bad})
        return best

    @staticmethod
    def _window_mean(x: np.ndarray, w: int) -> np.ndarray:
        """'same'-mode moving average in O(n): np.convolve with a w-sample
        kernel is O(n*w) — ~0.4 s on 20 s of audio — and this runs on every
        retry attempt and again when stitching. Output length is always
        len(x) (np.convolve 'same' returns max(len(x), w), which let index
        math run past the signal on inputs shorter than the window)."""
        n = len(x)
        c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
        k = np.arange(n + w - 1)
        lo = np.maximum(k - w + 1, 0)
        hi = np.minimum(k + 1, n)
        full = (c[hi] - c[lo]) / w
        start = (w - 1) // 2
        return full[start: start + n].astype(np.float32)

    def _dead_air_regions(self, p, rel_floor=0.03, win=0.25, density=0.90):
        """Spans of dead air the model inserted mid-chunk, as (start, end)
        sample indices. Windowed quiet-density based: the model scatters
        tiny blips through its silences, which chop a 1.2 s pause into sub-
        threshold runs that defeat any longest-run check. A region needs
        >=density of a win-second window below rel_floor*rms, then expands
        to its true quiet boundaries."""
        rms = float(np.sqrt((p ** 2).mean()))
        if rms < 1e-6 or len(p) < 3:
            return []
        sr = self.sample_rate
        quiet = np.abs(p) < rel_floor * rms
        w = max(1, int(win * sr))
        dens = self._window_mean(quiet.astype(np.float32), w)
        solid = dens >= density
        edges = np.diff(np.concatenate(([0], solid.astype(np.int8), [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        out = []
        for a, b in zip(starts, ends):
            i, j = int(a), int(b)
            while i > 0 and quiet[i - 1]:
                i -= 1
            while j < len(p) and quiet[j]:
                j += 1
            out.append((i, j))
        merged: list[tuple[int, int]] = []
        for i, j in out:
            if merged and i <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], j))
            else:
                merged.append((i, j))
        return merged

    def _solid_fraction(self, speech, floor=0.05, win=0.04) -> float:
        """Fraction of the chunk's duration carrying real speech energy
        (windowed rms >= an absolute floor). A degraded generation produces
        one loud burst and then near-silence: chunk-level rms and duration
        checks both pass, but only 10-40% of the audio is speech — the
        words are simply not there (confirmed by ASR)."""
        if len(speech) < 3:
            return 1.0
        w = max(1, int(win * self.sample_rate))
        n = len(speech) // w
        if n == 0:
            return 1.0
        solid = sum(1 for i in range(n)
                    if float(np.sqrt((speech[i * w:(i + 1) * w] ** 2).mean())) >= floor)
        return solid / n

    #: Optional ASR self-check (scripts/verify.py AsrVerifier): after the
    #: signal checks, every SENTENCE is transcribed once and the units it
    #: charges with errors are regenerated (`_verify_and_repair`). None keeps
    #: the signal-only checks — the behaviour without an ASR model.
    verifier = None
    #: repair rounds per request when a verifier is set
    repair_rounds = 2
    #: counters: units, attempts, checks, repaired, asr_errors
    qc_stats: dict | None = None

    #: takes per unit allowed by the signal-QC retry loop (research: with prompt
    #: self-extension the retries' extra compute bought little — research/PROGRESS.md)
    max_attempts: int = 3

    def _generate_with_retry(self, voice_cache, voice_off, chunk, attempts=None,
                             rng=None, lang="fa", prime=None, want_latents=False):
        """Generate one chunk, retrying when quality is bad. `rng` lets
        parallel callers give each chunk its own independent noise stream
        (the engine's shared rng is not thread-safe). Failure modes,
        all stochastic per the model card; a fresh attempt usually lands
        clean:
        (1) near-silent output (a generation that EOS'd into nothing) —
            caught by an ABSOLUTE rms floor, because every relative check
            is blind on a silent chunk (its own rms is ~0, so all of it
            looks "loud" and none of it looks "quiet");
        (2) burst-then-silence degradation — one loud syllable then
            mumbling: caught by the solid-speech fraction;
        (3) speech too short = early EOS = dropped words, or too long =
            runaway (the manifest's tps_est=3 is ~2x conservative vs the
            real 4-7 tokens/s, so the window is wide);
        (4) a long mid-chunk dead-air stretch — the model sometimes goes
            quiet for ~1 s at a random word, heard as a weird mid-phrase
            stop (measured windowed, so noise blips inside the silence
            cannot hide it).
        Preference order when no attempt is fully clean: loud beats silent,
        enough actual speech beats garbage, more speech beats less (dropped
        words read short), then less dead air. Returns trimmed speech
        (with `want_latents`: (speech, the kept take's latents))."""
        tokens = len(self.sp.encode(chunk, out_type=int))
        expected_speech = tokens / self.tps_est  # ~seconds (conservative)
        stats = self.qc_stats
        if stats is not None:
            stats["units"] = stats.get("units", 0) + 1
        best, best_key, best_lat = None, None, None
        n_prime = len(prime[1]) if prime is not None else 0
        if attempts is None:
            attempts = self.max_attempts
        for _ in range(attempts):
            if stats is not None:
                stats["attempts"] = stats.get("attempts", 0) + 1
            latents, _, _ = self._generate_chunk(voice_cache.copy(), voice_off, chunk,
                                                   rng=rng, prime=prime)
            if n_prime:
                # the causal decoder needs the carrier as context; its samples are cut
                audio = self._decode_all(list(prime[1]) + latents)[n_prime * self.frame_samples:]
            else:
                audio = self._decode_all(latents)
            s0, e0 = self._speech_bounds(audio, tail_keep=self.tail_keep)
            speech = audio[s0:e0]
            # strip model dead air BEFORE measuring (and returning): a
            # chunk often says its words then trails ~1 s of pre-EOS
            # silence, which used to fail the density gate and trigger
            # two pointless retries — 3x the cost on clean chunks
            speech = self._compress_pauses(speech, max_pause=0.30, keep=0.12)
            dur = len(speech) / self.sample_rate
            rms = float(np.sqrt((speech ** 2).mean()))
            loud_ok = rms >= 0.02
            frac = self._solid_fraction(speech)
            # actual speech seconds: a short chunk often says its words fast
            # and then trails ~1 s of pre-EOS silence — total duration and
            # solid fraction both look bad while the WORDS are fine, and a
            # dense garbage attempt used to win the ranking on them.
            solid_dur = frac * dur
            dur_ok = 0.30 * expected_speech <= solid_dur <= 1.6 * expected_speech
            regions = self._dead_air_regions(speech)
            dead = max((j - i) for i, j in regions) / self.sample_rate if regions else 0.0
            key = (loud_ok, dur_ok, solid_dur, 0.0 if dead <= 0.35 else -dead)
            if best_key is None or key > best_key:
                best, best_key, best_lat = speech, key, (latents, audio)
            if loud_ok and dur_ok and frac >= 0.55 and dead <= 0.35:
                break
        if want_latents:
            return best, best_lat
        return best

    def _speech_bounds(self, p, rel_floor=0.03, head_keep=0.02, tail_keep=0.06,
                       min_run=0.03):
        """Indices of actual speech: the model pads generations with near-
        silence (up to ~2 s) before EOS, and that dead air is what makes
        chunked output feel disjointed. Threshold is relative to the chunk's
        own RMS so quiet voices are not clipped; onsets must be sustained
        (>= min_run) so a single murmur sample does not count as speech."""
        rms = float(np.sqrt((p ** 2).mean()))
        if rms < 1e-6:
            return 0, len(p)
        loud = np.abs(p) >= rel_floor * rms
        # windowed: a position is speech if >=60% of its 2*min_run window is loud
        w = max(1, int(min_run * self.sample_rate))
        dens = self._window_mean(loud.astype(np.float32), 2 * w)
        solid = dens >= 0.6
        idx = np.where(solid)[0]
        if len(idx) == 0:
            return 0, len(p)
        sr = self.sample_rate
        start = max(0, idx[0] - int(head_keep * sr))
        end = min(len(p), idx[-1] + int(tail_keep * sr))
        return start, end

    def _stitch(self, segments):
        """Join chunk audios into one continuous-sounding piece: loudness
        matched to the MEDIAN chunk level (chunks are generated fresh and
        their levels differ wildly — a female_narration segment measured at 3x
        its median), plus 8 ms declick fades and fixed short pauses instead
        of the variable 1-2 s of model-generated dead air. The gain clip is
        deliberately wide (0.4-2.5): the old ±35% bound left such an outlier
        2.25x above the rest — a 7 dB jump between adjacent segments; the
        final peak guard still protects the output. `segments` is a list of
        (audio, gap_before) pairs — a phrase start (punctuation position)
        gets a slightly longer pause than an intra-phrase chunk boundary."""
        if not segments:
            return np.zeros(0, dtype=np.float32)
        levels = [float(np.sqrt((p ** 2).mean())) for p, _ in segments if len(p)]
        target = float(np.median(levels)) if levels else 0.0
        if self.level_target and target > 1e-6:
            # one fixed level for every request, not each request's own median:
            # an answer is several requests played back to back, and each one
            # matched only to itself stepped in level at every request join
            target = self.level_target
        f = max(1, int(0.008 * self.sample_rate))
        out = []
        for p, gap in segments:
            rms = float(np.sqrt((p ** 2).mean()))
            if rms > 1e-6 and target > 1e-6:
                p = p * float(np.clip(target / rms, 0.4, 2.5))
            if len(p) > 2 * f:
                p = p.copy()
                p[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
                p[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
            if gap > 0:
                out.append(self.pause(gap))
            out.append(p)
        audio = np.concatenate(out)
        # peak guard: loudness matching can push peaks past full scale. With a
        # limiter after the join (limiter_ceiling) the peaks are left for it,
        # so a sentence is not scaled down as a whole
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 0.98 and not self.limiter_ceiling:
            audio = audio * (0.98 / peak)
        return self._compress_pauses(audio)

    #: Audio kept after a chunk's last solid speech (s). 60 ms cut endings in
    #: the middle of their decay; 0.20 s with the inserted gaps shortened by
    #: the same 0.14 s (pauses as long as before): UTMOS +0.008, 95 % CI
    #: +0.001..+0.014, CER unchanged (eval/fluency.py, 49 cases × 3 seeds).
    #: Longer pauses on top scored higher still (+0.019) but slow every answer.
    tail_keep = 0.20

    #: Fixed speech RMS for every request (None: each request's own median).
    #: Renders of one voice differed by 3.4 dB (p10-p90) on eval/fluency.py,
    #: so a long answer read as several requests stepped in loudness at each
    #: join. With 0.13 and room_tone: CER 7.1 -> 6.9 %, UTMOS unchanged.
    level_target: float | None = 0.13
    #: Peak ceiling of the limiter frontend.join runs on the finished answer
    #: (None: each chunk / sentence is scaled down when it would clip — the
    #: research default). Set with a higher level_target to speak louder.
    limiter_ceiling: float | None = None
    #: Fill inserted pauses with a quiet noise floor instead of digital zero.
    #: The model's own quiet sits near -59 dBFS; an inserted gap of exact
    #: zeros drops below it, which on headphones is the sound switching off.
    room_tone: bool = True
    _ROOM_TONE_RMS = 0.0011

    def pause(self, seconds: float):
        """An inserted pause: silence, or room tone (see `room_tone`)."""
        n = max(0, int(seconds * self.sample_rate))
        if not self.room_tone or n == 0:
            return np.zeros(n, dtype=np.float32)
        # deterministic per length — a pause must not change the take's seed
        r = np.random.default_rng(n)
        x = r.standard_normal(n + 64)
        x = np.convolve(x, np.ones(8) / 8, mode="same")[32:32 + n]   # dull, like the floor
        x *= self._ROOM_TONE_RMS / max(float(np.sqrt((x ** 2).mean())), 1e-9)
        return x.astype(np.float32)

    def _compress_pauses(self, audio, max_pause=0.38, keep=0.28, rel_floor=0.03):
        """Remaining dead-air stretches (whatever survived the retry in
        _generate_with_retry) longer than `max_pause` are shortened to
        `keep` seconds with small fades, so the flow of speech stays
        continuous. Regions come from windowed quiet density — the model
        peppers its silences with tiny blips that would chop a 1 s pause
        into sub-threshold runs. Deliberate intra-sentence pauses — chunk tail
        plus inserted gap: strong lead-in 0.34 s, phrase 0.24 s, chunk 0.22 s
        — stay below `max_pause` untouched; the sentence pause is inserted by
        the caller, outside this pass."""
        regions = [r for r in self._dead_air_regions(audio, rel_floor=rel_floor)
                   if (r[1] - r[0]) > int(max_pause * self.sample_rate)]
        if not regions:
            return audio
        sr = self.sample_rate
        f = max(1, int(0.008 * sr))
        out, pos = [], 0
        for a, b in regions:
            out.append(audio[pos:a])
            seg = audio[a : min(a + int(keep * sr), b)].copy()
            if len(seg) > 2 * f:
                seg[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
            out.append(seg)
            nxt = audio[b : b + f]
            if len(nxt) == f:
                nxt = (nxt * np.linspace(0.0, 1.0, f, dtype=np.float32)).astype(np.float32)
            out.append(nxt)
            pos = b + f
        out.append(audio[pos:])
        return np.concatenate(out)

    def _generate_chunk(self, cache, off, chunk, rng=None, prime=None):
        """One take. `prime` = (carrier token ids, carrier latents [K, ldim]):
        the text prefix becomes carrier+chunk and the carrier's latents are
        teacher-forced after BOS, so the take starts mid-utterance (continuation
        priming, see `prime_short_tokens`). The returned latents are the
        chunk's only; the carrier's are not repeated."""
        tokens = self.sp.encode(chunk, out_type=int)
        ptoks, plat = prime if prime is not None else ([], np.zeros((0, self.ldim), np.float32))
        text_emb = self.lut[np.asarray(list(ptoks) + list(tokens))][None].astype(np.float32)

        r = rng if rng is not None else self.rng
        scale = self.temp ** 0.5
        noise = (r.standard_normal((1, self.ldim)) * scale).astype(np.float32)
        seq = np.concatenate([np.full((1, self.ldim), np.nan, np.float32),
                              np.asarray(plat, np.float32).reshape(-1, self.ldim)])[None]
        lat, _, cache, off = self._flow_step(seq, text_emb, off, noise, cache)
        latents = [lat]

        words = len(chunk.split())
        frames_after_eos = (3 if words <= 4 else 1) + 2
        max_gen_len = int(np.ceil((len(tokens) / self.tps_est + self.gen_pad) * self.frame_rate))
        # the BOS step already produced one latent; never decode past capacity
        # (a carrier's frames are decoded together with the take's)
        max_gen_len = min(max_gen_len, self.max_latents - 1 - len(plat))

        eos_step = None
        for step in range(max_gen_len):
            noise = (r.standard_normal((1, self.ldim)) * scale).astype(np.float32)
            lat, eos, cache, off = self._flow_step(
                latents[-1].reshape(1, 1, self.ldim),
                np.zeros((1, 0, self.dim), np.float32), off, noise, cache)
            if bool(eos.reshape(-1)[0]) and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            latents.append(lat)
        return latents, cache, off


def main():
    ap = argparse.ArgumentParser(
        description="Persian TTS via the pure-ONNX engine (no torch)")
    ap.add_argument("text", nargs="?", default="سلام، حال شما چطور است؟",
                    help="Persian text or phonemes")
    ap.add_argument("voice", nargs="?",
                    default=str(BASE / "voices" / "male_hello.wav"),
                    help="reference voice WAV")
    ap.add_argument("out", nargs="?",
                    default=str(BASE / "output" / "tts_onnx.wav"),
                    help="output WAV path")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproduce a run exactly")
    ap.add_argument("--pack", action="store_true",
                    help="merge comma phrases into longer breaths")
    args = ap.parse_args()

    eng = OnnxTts(seed=args.seed)
    t0 = time.perf_counter()
    audio = eng.synthesize_text(args.text, args.voice,
                                mode="pack" if args.pack else "split")
    dt = time.perf_counter() - t0

    import soundfile as sf

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, eng.sample_rate)
    print(f"text: {args.text}")
    print(f"generated {len(audio)/eng.sample_rate:.2f}s audio in {dt:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
