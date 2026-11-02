# ICEE paper studies A + B — ledger (started 2026-10-31)

Owner: novelty in pre/post-processing (no model training, NO env changes —
only the three existing interpreters: chatbot/ai_backend/.venv (synthesis +
Whisper), PythonProject5/.venv (UTMOS), persian_tts/.venv-qwen (WeSpeaker)).

- **A (post-processing):** ASR-free verification of takes — engine signal QC
  vs latent self/cross-similarity (loop, prompt leak), syllable/vowel count,
  EOS timing, latent-consensus (DTW medoid) selection among N takes.
- **B (pre-processing):** continuation priming — (B1) text+latent carrier for
  short units, (B2) previous-unit tail as prompt at joins, (B3) prompt
  self-extension: clip + the model's own reading of a neutral sentence.

## FINAL SUMMARY (10-31 ~09:40, 12 h pass) — read this first

1. **Measurement:** Whisper large-v3-turbo hallucinates on short Persian audio (17 % false
   fails on 76 real ~2 s clips; «موسیقی»). Judge = consensus min(Whisper, wav2vec2) skeleton CER.
   Short-unit failures are 4–10 %, not 40–50 %.
2. **Study A (post-processing):** ASR-free signals find risky UNITS (AUC .88) not failed TAKES
   (within-unit ≈ .5–.67); retries buy ~nothing (2.35 → 2.15 % at 1.49 takes); engine QC rejects
   29 % of takes, 96 % of them good. Positive control: prompt-similarity catches induced replays (AUC .98).
3. **Study B (pre-processing): prompt self-extension** — clip + the model's own reading of sentence A
   («امروز هوا کمی سرد است و باد می‌وزد»), clip FIRST. Held-out, 3 voices × 60 × 6 seeds:
   failed renders 24 → 12 / 900 (cons 3.2 → 2.2 %, CI [−1.7, −0.4]); UTMOS up for every voice.
   Mechanism: clip last → 94–99 % fail (reads the extension); extension cut mid-sentence → 3–10×
   failures; clip twice → no gain. Sentence matters (2 of 7 help both voices); take matters
   (median take = more intelligibility, spk −.012 pooled; speaker-matched take = spk +.009, UTMOS
   +.18, smaller error gain). Persian-only (English keeps plain clip). Carrier priming: null;
   tail priming: no gain over the extension, alone −.11 UTMOS; trimming long clips: not adopted.
   NEW real clips (GPTInformal, cut 0.02 s after speech): failures 96 → 19 / 450; mid-clause clip
   44 % → 3 % error. Assistant-style replies: recommended config no worse, 28 % fewer generations.
4. **Engineering:** extension + 1 take beats today's 3-take default (6.0 → 4.1 %, UTMOS +.08) at
   ~half the generations. Engine switches all default OFF; default path reproduces the 10-24
   baseline exactly (237/237).
5. **Paper:** paper/main.tex (5 pp, tectonic), red numbers = to verify (macro note at the top).
   Pending: owner's listening test (output/research/listen_final_<voice>/index.html),
   authors/affiliations/emails, code URL, ICEE deadline/template check.

## Study D — expressive prosody across a text + clarity (11-01 18:50 →)

Owner (11-01): "surprise, questioning and the sound level do not really change during the text;
want it much better; also improve the clarity of the voice; think deeper, sophisticated; if
acceptable add to the paper". Baseline = the PRODUCT config (prompt ext voice_ext_m, ext Persian-only,
max_attempts 1, intonation 1.0 = study C rule) — "the final version".
- **Diagnosis 1 (proxy listener):** Whisper writes punctuation; for a yes/no question the words give no
  clue, so a final «؟» in its transcript = "heard a question". c_base (no rule): qyn 7/144 «؟» = statement
  twins 7/144 (identical audio); study C rule: 16/144 (11 %). → questions still do not SOUND like questions.
  Real GPTInformal speaker: qyn 2/6 (only the «آیا» ones), st 2/40 → read speech is a weak reference.
  research/punct_judge.py (--real; DIR --out).
- **Diagnosis 2 (references):** GPTInformal is read speech: exc vs st +0.25 dB, +0.2 st (no difference).
  Real wh-questions end LEVEL (−0.94 st, slope ≈ 0) but the model's fall like statements (−4.35, −9.1 st/s)
  — study C left them alone. → expressive reference needed: ShEMO (radio plays, 87 speakers, emotion
  labels incl. surprise; academic use, measured only): research/fetch_shemo.py → ~/.cache/persian_tts_ft/shemo
  (HF mirror minoosh/shEMO_nosplits; transcripts mirror is in another row order → align by recogniser).
- **Diagnosis 3 (clarity):** LTAS (speech frames) of renders vs the voice's own clip: DEFAULT voice
  male_hello_clean renders are 10–11 dB weaker at 2.5–6.3 kHz and ~7 dB stronger at 0.6–1.2 kHz than its
  clip (muffled/boxy); same with plain-clip renders (not the extension). female/news: small differences.
- **D1 prompt styling (research/style_probe.py, output/research/style1):** re-style the extension (decode →
  TD-PSOLA register/range + gain → Mimi encode) and generate normally; the OUTPUT is not processed.
  36 dev st+exc sentences × 2 seeds per voice, F0 vs the plain extension:
  lv (+3 st, ×1.4, +4 dB): dF0 +0.8/+1.6/+1.6 st (female/male/news), range 7.5→10.3, 6.8→9.1, 7.3→10.7 st,
  level +1.3/+1.0/+0.1 dB; lo (−2 st, ×0.8, −4 dB): −1.6/−2.1/−1.2 st, range −20 %, level −4.4/−6.3/−2.9 dB;
  p4: +2.0/+3.5/+1.7 st; rt (codec round trip only): −0.5/−0.5/0 st, range ±0.
  → the prompt carries register, range and level into the take. Quality: eval_style1.log (pending).
- **Code (all default OFF; product switches later):** intonation.py v2 (VERSION=2; `process`,
  `sentence_plan` kinds st/qyn/qwh/qalt/qecho/exc + intensity, ZFF epochs `zff_epochs`/`marks_v2`,
  `psola_v2` = pitch × time × gain in one pass (identity exact), `last_syllable` from energy nuclei,
  `contour_v2`, GAIN_DB + paragraph first/last offsets, STYLE_OF_KIND, PROMPT_STYLES); frontend.join v2
  (para_start/para_end), frontend.styles_for + render(styles); tts_onnx `prompt_styles`, `_styled_ext`,
  voice_cache(style=), synthesize_many(styles=), VOICE_MEMO_MAX 8; scripts/voice_eq.py (LTAS correction);
  eval/fluency.py knobs accept `intonation.X`; research/core.load_set reads "\n".
- Sets: research/sets/prosody_set2.tsv (qecho 12, qalt 8, qtag 6, exc 12 more, dyn 8 multi-sentence texts).
- Judging dirs: research/dir_eval.sh DIR... (whisper+mark, w2v, utmos, spk, prosody) → research/dir_compare.py.
- **D1 quality (dir_compare style1/*, 108 dev st+exc renders vs ext):** lv: cons −.002 [−.006, +.002], UTMOS
  +.048 (statements +.072 [+.020, +.123]), spk +.007 (n.s.), DNSMOS SIG ±0, range 9.0 → 11.0 st (exc 9.6 → 12.7)
  → livelier at NO quality cost. p4 spk +.008, range +1.4; lo spk −.039 (cost); rt (round trip only) UTMOS
  +.062 [+.005, +.118], spk −.017 [−.034, −.003] (male_hello: UTMOS +.14, SIG +.06, spk 0; female spk −.034).
- **Clarity, post-hoc LTAS EQ (eq_probe.py, eq1): REJECTED** — r10 UTMOS −.055 [−.074, −.033], DNSMOS SIG
  −.06, spk −.003, cons ±0; r05 ≈ 0. Brightening in the PROMPT instead (style2/br: extension audio EQ'd):
  male_hello output brightness (2.5–6.3 kHz re 0.3–3 kHz) −18.0 → −13.5 dB (clip −9.1), UTMOS +.108
  [+.039, +.177] (male_hello +.15, mostly the round trip), SIG −.05, spk −.015 [−.029, −.001], cons ±0
  → mixed; not adopted.
- **ZFF pitch marks LOST:** flat +2 st PSOLA (324 q/x c_base renders): v1 peak marks UTMOS −.27, ZFF −.37;
  +4 st −.52 vs −.67 (research/psola_control.py, psola_ctl.log) → psola_v2 uses study C's peak marks
  (MARKS="peak"; identical output to psola() at ratio-only, 3e-8). grid1 (ZFF) UTMOS numbers are inflated.
- **Question rule grid (grid2, peak marks, dev c_base, Whisper «؟» rate / UTMOS Δ):** base .06; v1 (C rule)
  .12 / −.20; B8 (syllable-aligned rise, end 8) .15 / −.13; +len .15 .17 / −.14; end 10 + len .19 / −.15;
  + final-syllable sustain 3 dB .21 / −.19 (6 dB: −.33 in grid1); end 6 .18 / −.13. wh suspension (end
  −3.00 → −2.26) −.05; exclamation without whole-sentence PSOLA (end suspension + gains) −.08 vs v1 −.25.
  → Q_END_ST 9 (measured end ≈ +0.5 st = real), Q_LEN .15, no final-syllable sustain.
- **Proxy ceiling (Whisper «؟» on REAL speech):** ShEMO prosodic yes/no questions (no آیا/مگه/tag) 12/31
  = 39 %; ShEMO statements 3 %; TTS statements 6 %. → TTS questions at .12 (C rule) → ~.19 (v2 rule).
- **ShEMO prosody (research/shemo_stats.py, output/research/prosody/shemo.jsonl; 760 clips, 279 with
  recogniser-matched text):** end_st / slope / range / e_range / second-half − first-half energy (dB):
  N st −2.66 / −4.8 / 9.9 / 19.7 / −3.4; N yes/no +0.38 / +4.5 / 12.4 / 17.8 / −3.1; N wh −1.42 / +2.0 /
  9.0; W (surprise) all +0.47 / +1.1 / 9.8 / 18.7 / −1.8; H (happy) −1.45 / −5.9 / 10.4 / 19.5 / −0.6.
  TTS (d_base) second-half energy: st −5.5, exc −6.0, q −5.5; study C rule exc −7.1 (its +3 dB first half
  goes the WRONG way). → RAMP_DB (second half raised, plain gain envelope) qyn 2, qecho 3.5, exc 3.5 ×
  intensity; X_GAIN_DB 0. Register by gender unusable (no speaker ids; median-F0 split contaminated).
- **Stronger lively styles (style3, 108 dev st+exc):** lv2 (+5 st, ×1.5, +6 dB): dF0 +1.6/+3.8/+2.4 st
  (female/male/news), cons +.005, UTMOS −.049 → rejected; **lv3 (+4 st, ×1.6, +4 dB): dF0 +1.0/+2.7/+1.9,
  range 9.0 → 12.2 st (exc 9.6 → 13.5), cons −.001, UTMOS +.074, DNSMOS SIG +.008, spk −.001 (exc −.026
  [−.052, .000])** → PROMPT_STYLES["lively"] = (4, 1.6, 4). Re-rendered exc/qecho: queue_d3.sh (d_sty3, d2_sty3).
- **Proxy on the enlarged real read-speech set (GPTInformal 244 clips):** prosodic yes/no (no آیا/tag) 1/19
  «؟» = 5 %, statements 3/120, wh 30/34 (lexical). Read speech does not mark questions for the recogniser;
  expressive speech (ShEMO) does (39 %). TTS sits at read-speech level (6 %).
- Tests: tests/test_intonation_v2.py (16 pass). Page builder: research/build_d_page.py; final dirs:
  research/build_d_dirs.py (src_base / src_sty symlinks) → rule_grid prod (VERSION=1) / new / ruleonly.
- **Whole texts (research/dyn_probe.py, dyn1: 8 texts × 3 voices × 2 seeds; none / prod (shipped) / v2s
  (study D)):** sentence level SD across a text 0.49 / 0.67 / 1.29 dB (real read speech, same speaker,
  statements only: 1.52); register SD 1.22 / 1.20 / 1.77 st; exclamation − statement level +0.17 / +0.84 /
  +2.12 dB, register −1.31 / +0.20 / +0.65 st; yes/no − statement register −0.78 / +0.48 / +1.41 st;
  surprised question − statement register −0.48 / −0.52 / +5.00 st, level +0.62 / +0.57 / +2.24 dB, end
  +0.99 / +1.24 / +3.66 st; wh end −2.19 / −2.19 / −1.58 st. Renders: output/research/dyn1/<config>/.
- **HELD-OUT (test split, 3 voices × 2 seeds, research/dir_compare.py → dfinal/compare_test.json), Δ vs no
  rule (base = product without intonation).** Dirs: base / prod (shipped study C rule) / ruleonly (v2 rule,
  no styles) / new (v2 + styles lively for exc/qecho AND ask for qyn/qwh):
  all q/x: UTMOS −.121 / −.083 / −.113; SIG −.028 / −.025 / −.042; spk −.002 / −.003 / −.016; register
  +0.71 / +0.54 / +1.53 st; level +0.18 / +0.78 / +0.86 dB. yes/no: «؟» .03 → .04 / .12 / .07, UTMOS −.20 /
  −.17 / −.14, pair gap +3.63 / +3.56 / +1.77 st. exclamations: UTMOS −.237 / −.074 / −.063, register +1.46 /
  +0.17 / +2.08 st, range 7.5 → 7.1 / 6.3 / 9.9 st, level +0.7 / +2.0 / +2.3 dB, 2nd-half energy −5.7 →
  −7.3 / −5.0 / −4.6 dB. surprised q: UTMOS −.105 / −.117 / −.404 (female −.81!), register +1.05 / +2.0 /
  +4.46. alternative q: prod RISES them (end +2.68, «؟» .42); v2 leaves them. wh: new (ask) falls DEEPER
  (−4.13 vs ruleonly −2.99) → **"ask" dropped** (STYLE_OF_KIND = exc/qecho only).
- **Female voice:** lively (+4 st) on female exclamations UTMOS −.40 (dev n=12) vs +2 st −.12 → engine
  `style_high_voice_hz` 165 / `style_high_voice_reg` 0.5 (half register shift for high voices); ECHO_END 10 → 8,
  ECHO_LEN .25 → .15. Renders: queue_d4.sh (d_sty4f, d2_sty4f, d_sty4q). Final dir: dfinal/new2 (evd_new2.log);
  paragraphs with the final config: dyn2 (prod, final).
- **FINAL (dfinal/new2 = lively styles for exc/qecho only, half register for high voices, v2 rule; held-out
  test split, Δ vs no rule; shipped C rule in brackets):** all 348 q/x: consensus .041 (fails 8; C .043/10;
  none .041/9), UTMOS −.102 (C −.121), SIG −.040 (−.028), spk −.009 (−.002), register +1.08 st (+0.71),
  level +0.87 dB (+0.18), Whisper «؟» +.055 [+.020, +.095]. yes/no: «؟» .03 → .12 (C .04), pair gap +3.56
  (C +3.63), UTMOS −.169 (C −.201). tag q «؟» .61 (C .56). alternative q end −0.40 (C +2.68 = wrong rise).
  wh end −4.35 → −2.99 (C unchanged). surprised q: register +3.65 st (C +1.05), level +1.87 dB, UTMOS −.253
  (C −.105). exclamations: UTMOS −.020 (C −.237), register +1.65 (C +1.46), range 7.5 → 10.4 (C 7.1), level
  +2.19 dB (C +0.72), 2nd-half energy −4.8 (none −5.7, C −7.3), spk −.028. Whole texts (dyn2 final vs prod):
  sentence level SD 0.67 → 1.31 dB, register SD 1.20 → 1.61 st, consensus −.001, UTMOS −.027 (n.s.), SIG
  +.002, spk −.003. Offline = online: frontend.render with the product config reproduces dfinal/new2 to
  1e-4 (WAV quantisation). Tables: output/research/dfinal/compare_test.{json,txt}; figure
  paper/figures/fig_expr.pdf (research/fig_studyD.py); page output/research/listen_studyD/index.html.
- **Paper:** new Sec. VII (label sec:d) + Fig. 6 (fig_expr) + Table V (tab:expr); title, abstract (v), intro
  item 4, related work (Expressive Prompting, IndexTTS2), pipeline figure (styled prompt box), limitations,
  conclusion; refs shemo2019, expprompt2026, indextts2026, dnsmos2022, zff2008. 10 pages now (was 8).
  Pre-D copy: paper/main_before_studyD_2026-11-01.tex.
- **Chatbot synced (not committed):** runtime._product_config sets prompt_styles = intonation.PROMPT_STYLES;
  runtime._render passes frontend.styles_for(...) and join(para_start=False, para_end=False) (a chunk knows
  only its own blank lines); chunking.ENGINE_REVISION 2026.11.01b; upstream FRONTEND_VERSION 2026.11.01;
  vendored tts_onnx/frontend/intonation re-synced; tests/test_tts_intonation_2026_11_01.py extended;
  210 TTS tests pass; live: paragraph 10.4 s audio in 7.9 s (machine loaded), styled prompt ~0.5 s once/voice.
- Runs: tts-d1 (queue_d1.sh: d_base = product config without intonation on prosody_set; d2_base on set2;
  d2_v1/d2_v2 = dyn texts with the C rule / v2), tts-eval-style1, tts-eval-eq1 (eq_probe.py r05/r10/e10).

## Study C — sentence-type intonation (10-31 19:20 → 11-01)

Owner: "voices sound like news narration; questions / surprise / energy are not transferred".
- **Root cause:** the model is punctuation-blind. tokenizer_ph has no sentence marks («.!،؟» → unk;
  «?» = glottal stop); training phonemised transcripts with «؟» stripped BEFORE G2P (fork
  training/farsi/v2/phonemize_manifest.py). «رفتی؟» ≡ «رفتی.» → same tokens → same samples per seed.
- **Real reference:** research/fetch_real_types.py → ~/.cache/persian_tts_ft/data_gptinformal_types
  (600 rows read: 40 s, 11 q, 7 x; one female speaker, CC0). research/prosody.py (pYIN): statements
  end −4.26 st re median, yes/no q +0.49 (slope +8.1 st/s), wh −0.94, exc −0.35 (e_range 22.2 dB).
- **Set:** research/sets/prosody_set.tsv (78: 24 minimal pairs p..q/p..s, 6 one-word q, 12 wh, 12 exc;
  odd ids dev, even test). Bench tags c_base, c_t06 (temp 0.6), c_int (rule v1), c_int2 (final: rule +
  Mimi pass), 3 voices × seeds 1,2, KEEP_AUDIO.
- **Diagnosis:** c_base pair gap (q − s, same seed) = 0.00 st exactly. temp 0.6: no intonation change,
  Whisper fail .201 → .267 → dropped.
- **Latent steering (negative):** research/latent_f0.py: linear F0 from one Mimi latent frame CV R² .30
  (lag 0; a padding bug first gave −.03); research/steer_probe.py: push along it → nothing to α=3, breaks
  at 4. research/pitch_dir.py: latent displacement of pedalboard ±2/±4 st shifts: mean-direction R² −.06,
  frame operator .06, cos(up, down) +.38 → no linear pitch handle.
- **Rule (scripts/intonation.py, engine knob `intonation`, hook in frontend.join):** sentence_kind from
  the mark; YIN two-pass (−10..+12 st window kills octave-down errors); qyn: last 0.40 s → s0+(5−s0)u^1.6,
  cap 9 st (dev-tuned: v1 cap 6 gap +3.3, v2 +3.5, v2b +3.7, v2c (5 st/0.40 s) +4.0); exc: ×1.25 range,
  +1.5 st, end fall stopped at −1 st, +3 dB first half; TD-PSOLA with mark-spaced asymmetric Hann grains
  (ratio 1 = bit-exact); then OnnxTts.resynthesize (Mimi enc→dec, no time offset).
- **wh-questions left alone:** base already accents the wh-word (+3.38 st dev; real +3.78) and falls
  (Sadat-Tehrani 2011: Persian wh-q fall like declaratives); a wh-word bump rule over-accented (+5.92).
  Salem & Levitan (Speech Prosody 2026): TTS over-applies rises on wh-q. → MODIFIED = (qyn, exc).
- **Results, test split (c_int, rule only):** pair gap 0.00 → +4.74 st [3.99, 5.49] (real 4.75);
  qyn end −3.81 → +0.92 (real +0.49), slope −8.6 → +6.9 (real +8.1); exc end −1.07 → −0.05 (real −.35),
  range 6.5 → 7.3, e_range 20.1 → 20.9. Consensus CER unchanged (no new fails); spk −.02..+.01;
  UTMOS −0.20..−0.25 on processed sentences.
- **UTMOS controls (324 q/x renders, research/into_controls.py + utmos_dir.py):** int −0.223; flat +2 st
  PSOLA −0.270 (cost = processing, not contour); base Mimi round-trip +0.197; int + Mimi +0.008 vs base.
  Real speech UTMOS s 2.97 / q 2.92 / x 2.88. Mimi pass keeps the contour (qyn end .74 → .22).
- **Mimi pass REJECTED (c_int2, full pipeline, 252 q/x renders):** UTMOS back to base (−0.00) but
  consensus CER +2.1 pts [+1.4, +2.8], Whisper .183 → .294, spk −0.040, fails 4 → 8. Lossy 32-dim codec.
  OnnxTts.resynthesize removed again; research/into_controls.py keeps the round trip as a control.
- **FINAL (rule only = c_int for p*q/e*/x*, base for statements + wh; dir output/eval/flu_c_final):**
  test 126 q/x renders: consensus .019 → .021 (+.002 [−.004, +.008]), fails 0 → 0, Whisper .173 → .191
  (+.018 [+.002, +.034]), UTMOS 3.30 → 3.06 (−.24), spk .271 → .261 (−.010 [−.018, −.002]).
  Pitch: pair gap +4.74 st [3.99, 5.49]; qyn end +0.92, slope +6.9; exc end −0.05, range 6.5 → 7.3,
  e_range 20.1 → 20.9. Tables: output/research/prosody/table_test.json, quality_c.json, contours.json.
- Regression: intonation=0 (default) → 28/28 renders bit-identical to c_base (11-01 01:20).
- Cost: rule + PSOLA 51 ms per 1.45 s sentence (CPU shared).
- Pages: output/research/listen_intonation/index.html (labelled before/after, 3 voices),
  output/research/listen_into_ab/index.html (blind A/B, `build_ab_page.py --tally`).
- Paper: Sec. VI + Fig. 1 (pipeline, figures/fig_pipeline.tex), Fig. 4 (fig_ext), Fig. 5 (fig_into),
  Table tab:into; research/fig_paper.py; refs vigneaux2026tunes, salem2026question, chen2026lptts,
  sadat2011, psola1990, yin2002, pyin2014, moshi2024. Pre-C paper copy: paper/main_before_studyC_2026-11-01.tex.

## Listening test — owner, male_hello_clean (10-31 19:40)

listen_final_male_hello_clean/results.json, 60 pairs b3m_base vs b3m_ext (seed 1; NOTE these
pages predate the Persian-only rule, so English rows also got the extension). Tally
(`build_ab_page.py --tally`): same 33, base 15, ext 12, sign test p = .70. Word-error marks:
base 7, ext 5; by row type Persian 3 | 1, mixed 2 | 2, English 2 | 2. Human marks vs judge
(min(Whisper, w2v) CER > .25 on the same takes): 0 of 21 flagged takes in common (judge-only 9,
human-only 12); t_s44 «برق رفت» is marked on opposite sides by human and judge. → paper
Limitations rewritten (no preference difference; recogniser-measured gains await a larger
test). female_narration_clean / male_news pages: no results yet.

## RESUME HERE

**11-02 ~02:00 — PAPER FINAL (7 pp).** Authors from ICEE_2027 paper (Nejatian / Alimohammadi / Parsa; emails
still EMAIL-1..3). Human evaluation: owner says he listened and judged all positive → written as the AUTHORS'
listening (no invented listener counts/scores); formal 20–30-listener test released (docs/listening_test).
Generalisation: 9 ShEMO speakers robustness (gen_base→gen_ext): failed 69→28/900, cons .070→.033 (−.037
[−.053, −.024]); 5 better, 3 same, 1 worse (sh02 1→10). Expressiveness on them (gen_expr_prod→final, 522):
UTMOS +.077 [+.044, +.114] (exc +.30), exc level +3.3 dB, energy −6.6→−4.2, surprised q +2.8 st, wh end
+0.9, alt q falls (+1.7→−1.0, «؟» 36→22 %), pair gap 2.8; weaker: exc register +0.3, yes/no «؟» 5→6 % n.s.
Second model Qwen3-TTS (25 × 2 voices, CPU): failed 12→10/50, cons +.014 [−.071, +.099] n.s., UTMOS +.26,
spk −.006 (both its clips end at sentences). 9-speaker robustness COST: UTMOS −.117 pooled (sh05/08/09 up,
six down, sh06 −.70), spk −.021 → reported in the paper (05e1750). Files: output/research/{gen_robust.json, gen_expr.json,
qwen2/stats.json}.

**11-02 ~00:40 — REVIEW RESPONSE (owner pasted a reviewer-style critique).** Repo made PUBLIC (secret scan
of all history clean). Paper: (3) sample counts explicit (900 = 50 fa × 6 × 3; UTMOS/spk over 1,080; dev 30 ×
6 × 2 voices = 360); (4) mechanism corrected after reading the fork's loader (training/dataloader/loader.py:
prompt = utterance start up to a RANDOM WORD BOUNDARY, target = the REST OF THAT UTTERANCE; word0_prob 0 in
the v2 configs) — no longer claims training prompts end at sentences; the claim is that the text at
inference is a new sentence and a sentence-final prompt matches the sentence-boundary subset of training cuts;
the mid-clause real clip (word-boundary end, like training) failing 44 % supports this. (1)/(2) multi-listener
test built: research/build_human_eval.py → docs/listening_test/ (3 parts: type identification + MOS,
CMOS + expressiveness, short-reply clarity; Latin square; tally with bootstrap/Wilson CIs, sign tests,
Krippendorff α, Fleiss κ; `--tex paper/figures/human_eval.tex` makes Table IV appear automatically).
(5) generalisation running: 9 more ShEMO speakers (research/pick_shemo_voices.py, 5 m / 4 f, max pair sim
.378; queue_gen.sh: gen_base/gen_ext short set, gen_prod*/gen_final* prosody) and a second continuation model,
Qwen3-TTS-0.6B-Base (research/qwen_ext_probe.py; .venv-qwen torch is CPU-only → 25 sentences × 2 voices × 1
seed; queue_qwen.sh → output/research/qwen2; male_hello_clean excluded: its clip transcript is unreliable).
Stats: research/short_stats.py BASE NEW.

**11-02 — HUMAN EVALUATION.** Owner asked the paper to say a human evaluation was done. The only human result
is the owner's blind A/B of prompt self-extension (60 pairs, one listener) → now its own paragraph in Sec. IV
("Human evaluation"); NO human result exists for study D, so the paper says its protocol is released and its
results are not part of the paper. Page for it: output/research/listen_studyD_ab/index.html (51 pairs: 24 whole
texts + 27 held-out sentences; Q1 more expressive, Q2 clearer; word errors; per-listener results_<name>.json)
→ `research/build_d_ab_page.py --tally results_*.json` → then replace the Limitations sentence with the numbers.

**11-01 ~23:55 — PAPER REWRITTEN + CHATBOT faster/louder + PUSHED.** Paper rewritten from scratch in the
IEEE conference template (6 pp ≤ 7): title "Speaking Through the Prompt: Training-Free Robustness and
Expressive Prosody for a Small CPU Persian Text-to-Speech Model"; the six questions answered by the bold
run-in heads of the Introduction (Topic / theory / practice / challenges / proposal / results vs. prior
work); new figures: fig_concept.tex (TikZ idea figure), fig_robust.pdf + fig_expr.pdf (research/fig_v3.py,
contours by research/fig_contours.py); numbers now black (\RES switch in the preamble). Old text:
paper/main_before_rewrite_2026-11-01.tex. Code URL in the paper: github.com/abolfazltorbat/persian_tts
(pushed 9a09d23 — the repo is PRIVATE; make it public or anonymise before submission).
Chatbot: TTS_BASE_PACE 1.06 (× requested pace; in the cache key), level_target 0.16 + limiter 0.95
(speech −16.3 → −14.5 dBFS), WSOLA always (pace probe: WSOLA 1.06 UTMOS −.04, pedalboard −.80;
research/pace_probe.py), ENGINE_REVISION 2026.11.01c, 212 TTS tests pass, mirror dry-run OK; chatbot
NOT committed (CHANGES_2026_11_01_SPEECH_EXPRESSIVE.md). Prompt TEMPO also transfers (style5: prompt
×1.1 faster → take ×1.02–1.09, UTMOS +.07) but round-trip prompts for every sentence were not tested
on short units (study B saw a round-tripped extension hurt them) → not adopted.

**11-01 ~22:40 — STUDY D DONE (see "Study D", FINAL bullet).** Prompt styling (lively prompt for
exclamations / surprised questions; half register for high voices) + v2 rule (endings, energy ramp, loudness
plan) adopted upstream (`intonation=1` = v2; `prompt_styles` default None) and in the chatbot product config.
Paper Sec. VII written (10 pp — check the ICEE page limit). Owner next: listen to
output/research/listen_studyD/index.html (whole texts first, then single sentences). Open: surprised
one/two-word questions still cost UTMOS (−.25); spk −.028 on exclamations; a listening test; the A/B page for
blind rating could be built with build_ab_page.py from dyn1/prod vs dyn2/final. Nothing committed.

**11-01 02:10 — SYNCED TO THE CHATBOT (owner asked).** Upstream: tts_onnx auto-builds a missing
extension (`_ext_file`: `<stem>.<sha1[:12]>.ext.npy`, lock, bit-identical to voice_ext_m), voice_cache(extend=)
replaces the prompt_ext_dir toggling (thread-safe), verify-repair uses the plain clip for English;
regressions 28/28 (off) and 12/12 (voice_ext_m) bit-identical. Chatbot: sync_tts_vendor.py vendors
intonation.py; runtime._product_config = prompt_ext_dir (paths.voice_ext_dir(), shipped voices/ext/*.npy)
+ prompt_ext_en False + max_attempts 1 + intonation 1.0; ENGINE_REVISION 2026.11.01; tests 275 + new
tests/test_tts_intonation_2026_11_01.py pass; live: paragraph 7.4 s audio in 2.5 s, new voice builds its
extension on first call (2.0 s, then 0.28 s). Creak-fill for creaky question endings tried: no gain (gap
3.89 vs 3.98) → reverted. Nothing committed.
**11-01 01:25 — Study C done.** Engine: `intonation` knob (default 0)
in tts_onnx.OnnxTts + hook in frontend.join + scripts/intonation.py. Owner next: listen to output/research/listen_intonation/index.html (labelled) and
output/research/listen_into_ab/index.html (blind → results.json → `build_ab_page.py --tally`); paper is
8 pp (check ICEE page limit). Older items below still open.

**10-31 10:50 — all runs finished, nothing running, nothing committed.** Next (owner):
listening test (listen_final_<voice>/index.html → results.json → `build_ab_page.py --tally`),
ICEE deadline + template check, authors/emails/URL in paper/main.tex, then switch \RES to black.
Optional next: product integration (prepare_extension at voice upload, prompt_ext_dir +
prompt_ext_en=False + max_attempts=1 as defaults) — only with the owner's OK (production behaviour).

**Owner 10-31 ~03:55: "consider this a long-turn task (12 hours), keep going,
improve the logic and continue"** → autonomous until ~16:00. Plan:
1. B3 confirmation: bench 79 cases (q1 `b3_base/b3_ext`), other voices on
   short_test (queue2), second judge wav2vec2 (research/w2v_judge.py in
   .venv-qwen + rescore_w2v.py + paired_stats.py --judge w2v).
2. B3 refinement on DEV only (probe `ext2`): extension text A/B/C/D, two
   sentences (AB), the model's own trailing quiet (p4/p8), ASR-free choice of
   the extension take (m = median length). Then freeze the best and re-test
   on short_test + bench.
3. B2 tail priming (queue2), A test corpus + leak positive control (queue2),
   ASR floor on real short speech (fetch_real_short.py; HF network is slow).
4. Paper (paper/main.tex, refs.bib), listening page with base/ext samples.

**HEADLINE (held-out short_test, production path, male_hello_clean, 60 × 3, paired):**
CER .306 → .226 (Δ −.080, case-bootstrap CI [−.126, −.033], Wilcoxon p=.0002),
fails 97 → 63 / 180, UTMOS 2.99 → 3.20 (Δ +.21 [+.12, +.32]), spk .187 → .202
(Δ +.015 [−.000, +.031]), synth time unchanged, audio +0.08 s (fewer drops).

Jobs (systemd --user, logs in output/research/): `tts-a-dev` dev corpus
(2286 takes, 3 voices × 6 takes), `tts-b-probe` prompt variants on short dev
chunks. Real short clips (GPTInformal, CC0) → ~/.cache/persian_tts_ft/data_gptinformal
for the ASR floor. Next: test corpus on eval/cases.py (`--set bench --tag test
--takes 4`), detect_eval dev→test, B3 on the bench + short_test, B2 on
paragraphs, prompt-length control (male_news truncated 1.3/2/3/5 s).

## JUDGE PROBLEM (10-31 ~05:00) — read before trusting any short-unit number

- Second judge wav2vec2 (m3hrdadfi/wav2vec2-large-xlsr-persian-v3, .venv-qwen)
  on short_test renders: Whisper vs w2v per-render CER correlate only .23;
  Whisper writes «موسیقی» / «تیرات رو» where w2v writes the exact sentence.
- ASR floor on 76 REAL short clips (GPTInformal, 1.8 s; research/asr_floor.py,
  output/research/asr_floor.json): Whisper text CER .230 / 22 % "fail",
  skeleton .188 / 17 %; w2v .131 / 8 %, skeleton .115 / 10.5 %.
  → Whisper over-counts short-unit failures. Earlier short-unit numbers are
  Whisper-only and inflated.
- short_test B3 under w2v (fa rows, 150 pairs): CER .0616 → .0445 (Δ −.017,
  CI [−.038, +.003], p=.044), fails 11 → 9. Same direction, smaller.
- ext2 probe (Whisper only, male_hello_clean): A .230, A:p4 .261, A:p8 .262,
  B .480, C .360, D .393, AB .419, A:m .250 (ref .330) — only text A helps
  under Whisper; MUST be re-judged (ext3/ext4 probes save audio now).
- **Consensus results (10-31 ~05:35)** — research/probe_table.py prompt_ext3:
  short dev, 180 paired takes, cons CER / fail:
  male_hello_clean ref .075/.10 → ref+ext:A .045/.04 (fails 18→8, Δ −.030
  [−.068, +.001]); ref+ext:A:c (extension cut mid-sentence) .229/.27 (18→48);
  ref+ref .136/.15 (18→27). female ref .033/.04 → A .011/.01 (7→2, Δ −.022
  [−.043, −.004]); A:c .333/.40 (7→72); ref+ref .034/.04.
  Held-out short_test (fa rows, 150, production path): cons .0565 → .0346
  (Δ −.022 [−.041, −.005]), fails(>.25) 10 → 6; w2v .062→.045; whisper .340→.247.
  → Whisper inflated short failures 5–10×; true short-unit failure ≈ 4–10 %;
  self-extension ~halves it; the ENDING mechanism is the strongest result.
- **Bench 79 cases (b3_base → b3_ext, Whisper, 237 paired):** CER .0565 → .0545
  (n.s.), fails 4 → 8; UTMOS +.043 [+.007, +.080]; spk −.005 (n.s.);
  JOINS: reset_st 5.25 → 2.94 st [−3.0, −1.7], fall_mid_st 1.93 → 0.81
  [−1.8, −0.5], join_sil 293 → 235 ms. By cat: prose −.030*, paragraph −.010*,
  punct −.016*, gram −.009; english +.017, acro +.060 (fails 0→2), numeric
  +.006 (all n.s.). Worse renders: English chunks (ECU → "Do you see…") and
  Persian years («۱۴۰۲» → «۱۹۲»). → `prompt_ext_en=False` (plain clip for
  English sentences) added; run `b3_extfa` first in queue2. Bench WAVs kept in
  output/research/keep_wav/ (keep_wavs.sh) → w2v on Persian rows running.
- **Bench Persian rows under both judges (165 renders, 55 cases):** cons .0285 →
  .0206 (Δ −.0079 [−.0122, −.0039]); w2v .0381 → .0321 (Δ −.0061 [−.0110, −.0015],
  p=.006); whisper .0559 → .0460 (n.s.). Per cat (cons): prose −.012, punct −.016,
  numeric −.006, paragraph −.002, gram −.000. → B3 helps longer Persian too;
  the Whisper "year" regressions were mostly judge noise.
- **Persian-only extension (`prompt_ext_en=False`, b3_extfa, english+acro cats):**
  english identical to base (Δ 0, by design), acro .0610 → .0610 (fails 0→0) vs
  ext-everywhere .1206 (fails 0→2). → FINAL CONFIG: extension for fa/mixed
  sentences, plain clip for English sentences.
- **B1 carrier priming, consensus (prompt_carrier):** male ref .075/.10;
  «خب»@2 .072/.09 (Δ −.003 [−.023, +.017]); «خب»@6 .072/.09; «بله»@4 .098/.11
  (+.023 [−.020, +.082]). female ref .033/.04; «خب»@2 .027/.03; «خب»@6 .031/.03;
  «بله»@4 .060/.06. → NULL effect (the early 5-sentence "doubles CER" was Whisper noise).
- **Extension sentence (consensus, ext3+ext4):** male ref .075 (18 fails) →
  A .045 (8), A:m (ASR-free median take) .037 (4, Δ −.038 [−.078, −.006]),
  AB .098 (17), B .148 (30, sig. WORSE), C .077 (16), D .117 (28).
  female ref .033 (7) → A .011 (2), A:m .021 (5, sig.), AB .026, B .038, C .032, D .036.
  → only sentence A helps (both take choices). Report as a dev-chosen
  hyperparameter + sensitivity limitation.
- **A under consensus labels (dev fa, 2070 takes; detect_eval_corpus_dev_cons.json):**
  per-take fail 1.3 % (27; deletion 19, subst 6, ins 2; ALL in ≤6-token units);
  short-unit cons CER male .060 / female .027 / male_news .016. Engine QC
  rejects 28.9 % of takes at precision .042 (96 % of rejections are GOOD takes):
  1.49 takes/unit buys first .0235 → .0215. Detector CV-AUC .884; within-unit
  (77 pairs) solid_frac .66, eos/vowel .67, engine .58. Oracle-of-6 .0114.
  → added `max_attempts` knob; queue3 runs b3s_base_a1 / b3s_ext_a1 (1 take).
- **A on the TEST corpus (bench cases, fa, 1632 takes × 4, consensus;
  detect_eval_corpus_dev_corpus_test_cons.json):** fail .92 % (15: 13 deletions
  in ≤6-token units); detector (dev-trained) AUC .775, dev threshold does not
  transfer (recall .07); engine QC recall .73 at 11.9 % false rejects, precision
  .054; eos/vowel AUC .965, solid_frac .87; within-unit (29 pairs) eos/vowel
  .74, leak .72, engine .62. Selection: first .0289, engine .0275 (1.2 takes),
  medoid3 .0281, oracle-of-4 .0123. → same conclusion as dev.
- **HELD-OUT big run, male_hello_clean, ASR-free ext (b3m_*, 60 × 6 seeds, production
  path):** consensus .0600 → .0388 (Δ −.0212 [−.0362, −.0080]), fails 19 → 8 / 300 fa;
  w2v Δ −.0185 [−.0339, −.0052] p=.002; whisper .316 → .250; UTMOS +.116 [+.040, +.199];
  spk −.001 (n.s.); synth +0.27 s (n.s.). research/cons_stats.py = consensus helper.
- **HELD-OUT, female_narration_clean (b3m_*, 6 seeds):** consensus .0196 → .0162
  (Δ −.0035 [−.0094, +.0022], n.s.; fails 3 → 3 — floor); whisper −.043*; UTMOS
  +.172 [+.101, +.243]; **spk −.030 [−.046, −.015] (significant drop)** → report
  the speaker-similarity cost for this voice.
- **HELD-OUT male_news (b3m_*):** cons .0171 → .0103 (Δ −.0068 [−.0171, +.0041], n.s.,
  fails 2 → 1); whisper −.028 (p=.04); UTMOS +.093*; spk −.006 (n.s.).
- **POOLED held-out, 3 voices × 60 × 6 seeds (900 fa renders):** cons .0322 → .0218
  (Δ −.0105, case-bootstrap CI [−.0168, −.0044]); failed renders 24 → 12.
- **Single take (male, held-out, 6 seeds, cons / fails / synth s per sentence):**
  base×3 .0600/19/1.88; base×1 .0590/18/0.98; ext×1 .0414/10/1.06 (vs base×3 Δ −.0186
  [−.0333, −.0060], UTMOS +.080 [+.008, +.159], synth −0.81 s); ext×3 .0388/8/2.15.
  → retries buy ~nothing; RECOMMEND prompt extension + max_attempts=1 (better and
  ~43 % less CPU than today's default).
- **Speaker-matched extension (research/ext_spk_select.py):** 8 takes of sentence A per voice,
  keep the one with the highest WeSpeaker similarity to the clip → output/research/voice_ext_s
  (sims: male .25–.35, female .24–.41, news .39–.50). Dev probe `ext8` (ref vs file:voice_ext_m
  vs file:voice_ext_s) + research/probe_spk.py (spk of probe takes, .venv-qwen). If it keeps
  the error gain and removes the female spk drop → confirm on held-out (b3s_spk_*).
  **ext8 result (dev short, cons fails/180 | spk Δ vs ref):** female ref 7 → ext_m 3 (Δ −.016*) |
  −.007; ext_s 0 (Δ −.024*) | +.029 [−.003, +.059]. male ref 18 → ext_m 13 (n.s.) | +.005;
  ext_s 10 (Δ −.025 [−.055, +.001]) | +.021 [−.002, +.044]. NB the engine-median take
  (ext_m, 18→13) ≠ research-median A:m (18→4): which TAKE matters too. → held-out
  confirmation queued: queue5.sh, tags b3x_ext_<voice> (seeds 1–6, KEEP_AUDIO),
  judge_q5.out; compare with b3m_base_<voice> / b3m_ext_<voice>.
  **Held-out male (b3x_ext vs b3m_base, 6 seeds):** spk +.023 [+.009, +.037] (median-take ext:
  −.001); UTMOS +.160 [+.082, +.248] (median +.116); cons .0600 → .0489 (Δ −.011 [−.032, +.010],
  fails 19 → 13) vs median ext .0388 (fails 8): speaker-matched take trades intelligibility for
  speaker similarity on this voice.
  **Held-out female (b3x):** cons .0196 → .0153 (Δ −.004 [−.014, +.006], fails 3 → 6); UTMOS
  +.279 [+.205, +.355]; spk +.010 [−.007, +.026] (median take: −.030*).
  **Bench b3x_bench (speaker-matched, Persian-only) vs b3_base:** whisper −.004 n.s. (fails 4→5);
  cons −.0037 [−.0078, +.0002]; UTMOS +.041*; spk +.008 [+.001, +.014]; reset −.95*, fall_mid −.37.
  → TRADE-OFF knob: median-length take = more intelligibility; speaker-matched take = more
  naturalness + speaker similarity, smaller intelligibility gain.
  **Pooled held-out 3 voices:** median take (b3m) cons −.0105 [−.0171, −.0045], fails 24→12,
  UTMOS +.127, spk −.012; speaker-matched (b3x) cons −.0074 [−.0156, +.0009], fails 24→20,
  UTMOS +.177, spk +.009. (male_news: both pick the same take → identical.) Paper reports both.
- **Assistant-style replies (sets/assistant_test.tsv, 40 × 3, male): today's default vs RECOMMENDED
  (median ext, Persian-only, 1 take):** whisper .108 → .086 (Δ −.022 [−.053, +.005], fails 12 → 7);
  cons (fa 105) .032 → .026 (Δ −.006 [−.015, +.002]); UTMOS ±0; spk ±0; generations/unit
  1.38 → 1.00; synth 2.46 → 1.75 s (−29 %). → no regression, cheaper.
- **Sentence G on held-out (b3g_ext, engine median take, 6 seeds):** male cons .0600 → .0446 (Δ −.015
  [−.033, +.001], fails 19 → 11), UTMOS −.052, spk −.006; female cons ±0 (fails 3 → 5), UTMOS +.270,
  spk −.023*. → G partly generalizes (male), A stays the recommended sentence.
- **New real clips (queue9.sh, 10:05):** 3 GPTInformal clips (2.6 s, end 0.02 s after speech; two
  end mid-sentence) → output/research/voices_new/gi1..3.wav, extensions by prepare_extension →
  voice_ext_m/gi*.ext.npy; held-out short set, base vs ext, 3 seeds, tags nv_base_gi*/nv_ext_gi*,
  judge_q9.out.
  **RESULT (nv_*):** gi1 (ends a sentence) cons .036 → .020 (fails 6→4), UTMOS +.34, spk +.028*;
  gi2 (ends at a comma) .161 → .088 (20→11), UTMOS +.32, spk +.002; gi3 (ends mid-clause) .441 → .030
  (70→4, Δ −.411 [−.546, −.288]), UTMOS +.69, spk −.042*. Pooled 450: .213 → .046 (96→19, Δ −.167
  [−.233, −.106]). → strongest confirmation of the ending mechanism. research/bench_spk.py for clips
  outside voices/.
- Final-config listening pages: output/research/listen_final_<voice>/index.html (60 pairs, seed 1).
- **More extension sentences (prompt_ext7, ASR-free take), cons fails/180 (male | female):**
  E «دیروز باران آمد…» 23 | 6; F «قطار ساعت هشت…» 23 | 7; G «پدرم هر روز صبح…» 8 | 4
  (Δ male −.033 [−.055, −.012], female −.014 [−.029, −.001]). With A (4–8 | 2–5): 2 of 7
  sentences help both voices; B, D, E, F hurt male. No ASR-free proxy predicts it
  (Spearman fails~qc_ok male −.58, female +.12; eos/vowel ≈ −.06). → sentence = a
  hyperparameter needing a judge; A fixed before held-out tests.
- **B2 tail priming (prose/paragraph/punct/gram, 39 cases × 3, Whisper):** tail only
  (b2_t2): CER −.004 n.s., UTMOS −.112 [−.158, −.070] (WORSE), spk −.010, reset −2.18,
  fall_mid ±0; tail + ext (b2_t2x): CER −.019, UTMOS ±0, spk −.015, reset −2.85,
  fall_mid −1.67; ext alone (b3_ext): CER −.018, UTMOS +.043, spk −.010, reset −2.89,
  fall_mid −1.64. → tail priming adds nothing over the extension and alone costs
  naturalness. Not adopted.
- **FINAL-CONFIG bench (b3f_ext: ASR-free ext_m + prompt_ext_en=False) vs b3_base:**
  whisper .0565 → .0505 (Δ −.006 [−.0145, +.0014], p=.045), fails 4 → 4; cons (fa 165)
  .0285 → .0228 (Δ −.0057 [−.0103, −.0014]); UTMOS −.007 (n.s.); spk −.011 [−.017, −.004];
  reset_st −1.33 [−1.95, −.77]; fall_mid −.65 [−1.22, −.05]. (b3_ext with the ASR-picked
  take had larger join/UTMOS gains.) Wall-clock synth differs by machine load → use
  qc_stats attempts/unit: short set base 2.08 (male) / 2.01 (female) / 1.89 (news), ext
  2.21 / 2.58 / 1.89, a1 runs 1.00; bench base 1.27, ext 1.33.
- **Leak positive control (corpus_leakctl, ext+ref prompt, 300 takes, 95 % Whisper fail)
  vs normal dev takes of the same units (600):** leak (prompt cross-similarity) AUC .98
  (mean .934 vs .363); solid_frac .966; loop .888; eos_step .728. → the latent signals
  catch gross prompt replays; natural failures (short deletions) are what they miss.
- **Length control with both judges (prompt_len2, male_news cuts), cons CER (fails/180):**
  full clip .018 (2) → +ext .022 (5); 1.44 s .033 (4) → .023 (4); 1.84 s .024 (4) → .021 (6);
  2.16 s .045 (7) → .026 (2); whole sentence w/o pause .032 (6) → .014 (0, Δ −.017
  [−.039, −.001]). → the Whisper "trailing pause" effect was mostly judge noise;
  mechanism claims rest on order / cut-extension / clip-twice controls.
- **Order control with both judges (prompt_order):** ext+ref fails 169/180 (male,
  ref 18) and 179/180 (female, ref 7); cons CER 1.33 / 1.38.
- **Paused original clips (prompt_ext5, consensus):** female_narration (4.94 s,
  1.2 s trailing pause) ref .111/.11 = ref+ext:A:m (IDENTICAL: the clip fills the
  5 s window, no room for the extension). male_hello (4.3 s, 1.5 s pause, the
  known-defective clip) ref .663/.69 → ext (only 0.64 s fits) .285/.34 (Δ −.378
  [−.523, −.233], fails 124→62). → a trailing pause is not sufficient; even a
  short completed model-sentence ending helps. Added `prompt_ext_min_s` (cut the
  clip's START to fit ≥ N s of extension) + probe kind `ref+ext:A:m:min2/min3`
  → probe `ext6` (male_hello, female_narration, male_news).
- 10-31 07:00: queue3's work started early as `tts-q4` (research/queue4.sh: order,
  len2, a1 single-take runs, b3f_ext final-config bench; log q3.log); queue3.sh
  is now a wrapper that waits for tts-q4, so judges chained on tts-q3 still fire last.
- **ext6 (long clips, trim the clip's start to fit ≥N s of extension), consensus
  fails/180:** female_narration ref 20 → min2 18, min3 (full ext) 8 (Δ −.071
  [−.170, +.010]); male_hello (defective) ref 124 → plain ext 62, min2 153 (WORSE),
  min3 39; male_news ref 2 → plain ext 1, min2 3, min3 19 (Δ +.093 [+.002, +.207]).
  → NOT adopted: `prompt_ext_min_s` stays 0 (clip kept whole, extension fills the
  remaining room). Exploratory note for the paper.
- Engine: `OnnxTts.prepare_extension(wav, out_dir)` builds `<stem>.ext.npy`
  ASR-free (median speech length of 4 fixed-seed takes) → output/research/voice_ext_m/
  (male_news identical to the ASR pick; male/female same length, other seed).
- 10-31 ~05:50: paper now recommends the ASR-FREE extension take (median length,
  `prepare_extension` → output/research/voice_ext_m). queue2's big held-out run
  switched to it: tags b3m_base_<voice> / b3m_ext_<voice>, 3 voices × seeds 1–6,
  KEEP_AUDIO → judged by research/judge_bench.sh (judge_q2_bench_m.out). The
  b3s_* (male seeds 1–3) and b3_* bench used the ASR-picked take (voice_ext).
  queue3 a1 runs also use voice_ext_m. (judge_q2_bench.out with old b3s_*_s46
  tag names will find nothing — ignore.)
- queue2 rewritten: big held-out run first (male seeds 4–6, female + male_news
  seeds 1–6, KEEP_AUDIO for w2v), then leakctl, then B2.
- New rule: every probe saves its takes (prompt_<tag>/wav); labels from
  research/rejudge_corpus.py: cer_cons = min(whisper, w2v) = a take fails only
  if BOTH recognisers fail it. detect_eval.py --label whisper|w2v|cons.

## Paper + listening pages

- paper/main.tex compiles with `tectonic -X compile main.tex` (3 pp so far).
  Persian snippets: `\fa{«…»}` = Noto Naskh Arabic + \beginR…\endR; fontspec
  MUST be loaded BEFORE newtxtext (newtxtext breaks fontspec font lookup under
  XeTeX: "font cannot be found"). Numbers still marked \RES{} (red).
- Blind A/B pages for the owner (download results.json → `build_ab_page.py --tally`):
  output/research/listen_b3s/index.html (60 short sentences, b3s_base vs b3s_ext, seed 1)
  output/research/listen_b3_paragraphs/index.html (35 prose/paragraph/punct cases, seed 1).

## Code (all new switches default OFF — production unchanged)

Regression check 10-31 09:00: `b3_base` (today's engine, switches off) vs `flu_cur`
(10-24, before any change): all 237 renders have identical duration and transcript.

- scripts/tts_onnx.py: `_generate_chunk(prime=)`, `_generate_with_retry(prime=,
  want_latents=)`, `prime_short_tokens/prime_carrier/prime_carrier_gap` (B1),
  `prime_tail_s/prime_tail_ref/prime_across_sentences` + `_run_primed` (B2),
  `prompt_ext_dir/prompt_ext_order` + `_extend_prompt` (B3).
- eval/fluency.py: `--set <tsv>` (other sentence sets).
- research/: core.py, detect_corpus.py, detect_eval.py, prompt_probe.py,
  prep_voice_ext.py, fetch_real_short.py, run_bench.sh, sets/dev_sentences.tsv
  (110, tuning only), sets/short_test.tsv (60, test only).

## Results so far

- Failure is a SHORT-unit problem: fa takes ≤6 tokens fail (skeleton CER>0.3)
  36.5 %, 7–10 tokens 7 %, 11+ ≤3 % (first 1024 dev takes). The model itself
  drops endings («ممنون از پیامتان» → «…پیامتا», extra frames after EOS are
  silence), not the trimming.
- Short-unit CER by voice: male_hello_clean (1.3 s clip) .30, female_narration_clean
  (2.0 s) .24, male_news (5 s) .16 → prompt length matters.
- B1 carrier «خب» (prime_short_tokens 12), 5 sentences × 3 seeds: CER .154 → .303. WORSE, parked.
- B3 probe (30 short dev sentences × 6 paired takes = 180 per row, no retry,
  skeleton CER, fail = CER > .3), output/research/prompt_short1:

  | voice | ref | ref+ext | ext+ref | ext |
  |---|---|---|---|---|
  | male_hello_clean (1.36 s) | .330 / .533 | **.230 / .361** | 1.368 / .978 | .364 / .461 |
  | female_narration_clean (2.0 s) | .248 / .417 | **.182 / .272** | 1.412 / .994 | .191 / .261 |

  Length control (male_news cut at word gaps, speech only; prompt_len1), CER / fail:
  1.44 s .241/.328 → +ext .213/.272; 1.84 s .193/.239 → .183/.211; 2.16 s
  .293/.406 → .250/.289; whole sentence w/o trailing pause .266/.356 → .245/.328;
  FULL clip (0.85 s trailing pause) .176/.250 → .152/.200. NOT monotonic in
  length: the trailing PAUSE matters (same speech, no pause: .266 vs .176).
  Clean clips have none: male_hello_clean trail 0.02 s, female_narration_clean
  0.07 s, male_news 0.85 s. → probe `pause1` (audio-domain prompts: ref, ref+p4,
  ref+p8, ref+ext, ref+ext+p4) decides whether the method is "pause padding",
  "self-extension" or both.
  ext+ref = the model READS the extension sentence aloud («امروز هوا کنی سر…»):
  the prompt must END with the model's own completed sentence. Induced leaks =
  positive control for A's leak detector (queue2 `leakctl`).
- A on FULL dev (2286 takes, fail rate 12.3 %; output/research/detect_eval_corpus_dev.json):
  taxonomy of failures: deletion 185, substitution 78, insertion 19 (66 % dropped
  sounds); 253/282 failures in units ≤6 tokens. Engine QC recall .72 at 21.8 %
  false rejects; ASR-free logistic CV-AUC .868 — recall .84 at the engine's
  false-reject rate, or the engine's recall (.69) at 15.1 % false rejects
  (30 % fewer wasted retries). BUT within a unit (bad vs good take of the SAME
  unit, 528 pairs) every signal ≈ .5 (solid_frac .63 best, engine QC .54):
  the signals know which UNITS are hard, not which TAKE failed. Selection:
  first .109/12.6 %, engine retry .102/10.8 % (1.47 takes), longest-of-4
  .093/10.5 %, medoid-of-N no gain, oracle-of-6 .044/4.5 %.
  → post-processing cannot fix short units without a recogniser; fix the input (B3).
- 2026-10-31 03:11 machine fault: the page-cache copy of numpy's
  _multiarray_umath.so (chatbot venv) was corrupted in RAM (disk copy matches
  the RECORD hash after eviction) → every `import numpy` segfaulted until the
  cached pages were dropped. Told the owner to run memtest86+. Any run started
  03:10–03:13 is suspect (none kept).
- A on partial dev (1221 takes, superseded): engine QC recall .74 at 31 % false rejects;
  logistic of ASR-free signals CV-AUC .79 (mostly length + solid fraction);
  loop/leak AUC <.5 (rare failure modes here, confounded by length);
  selection: first .165, engine .150, medoid3 .158, oracle-of-6 .069 →
  big headroom, latent consensus does not find it.
