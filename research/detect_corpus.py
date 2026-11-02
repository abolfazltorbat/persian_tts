"""Study A, step 1: a labelled corpus of single takes (no retry).

For every generation unit of every sentence, for each voice, `--takes`
independent takes are generated exactly as production does (pristine prompt
state, no retry), and for each take we record
  * the engine's own signal QC (would today's retry rule accept it?),
  * the ASR-free signals of core.signals (latent self-/cross-similarity,
    syllable count vs vowel count, EOS behaviour),
  * the ground truth: Whisper on the trimmed take, consonant-skeleton CER.
Latents are saved (float16) so signals can be recomputed without regenerating.

    python research/detect_corpus.py --set dev --tag dev --takes 6
    python research/detect_corpus.py --set bench --tag test --takes 4

Resumable: rows already in rows.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

SETS = {
    "dev": core.RESEARCH / "sets" / "dev_sentences.tsv",
    "short": core.RESEARCH / "sets" / "short_test.tsv",
}


def cases_for(name: str) -> list[dict]:
    if name == "bench":
        return core.bench_cases()
    return core.load_set(SETS.get(name, name))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--voices", default="male_hello_clean,female_narration_clean,male_news")
    ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cats", default="")
    ap.add_argument("--prompt-kind", default="ref", help="ref | ref+ext | ext+ref (needs --ext-dir)")
    ap.add_argument("--ext-dir", default=str(core.OUT / "voice_ext"))
    a = ap.parse_args()

    out = core.OUT / f"corpus_{a.tag}"
    (out / "lat").mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = {r["rid"] for r in core.read_jsonl(rows_path)}
    lab = core.Lab()
    cases = cases_for(a.set)
    if a.cats:
        keep = set(a.cats.split(","))
        cases = [c for c in cases if c["cat"] in keep]
    voices = [v for v in a.voices.split(",") if v]

    jobs = []
    for c in cases:
        for ui, (ch, gap, lang, si) in enumerate(lab.units(c["text"])):
            for v in voices:
                for k in range(a.takes):
                    rid = f"{c['id']}|u{ui}|{v}|k{k}"
                    if rid not in done:
                        jobs.append((rid, c, ui, si, ch, lang, v, k))
    print(f"{len(cases)} cases, {len(jobs)} takes to do ({len(done)} already done)", flush=True)
    def prompt_for(v):
        ref = lab.prompt_latents(v)
        if a.prompt_kind == "ref":
            return ref
        ext = np.load(Path(a.ext_dir) / f"{Path(lab.voice_path(v)).stem}.ext.npy")
        room = int(5 * 12.5) - len(ref)
        ext = ext[-room:]
        return np.concatenate([ref, ext] if a.prompt_kind == "ref+ext" else [ext, ref]).astype(np.float32)

    prompts = {v: prompt_for(v) for v in voices}

    def work(job):
        rid, c, ui, si, ch, lang, v, k = job
        raw = prompts[v]
        cache, off = lab.prompt_cache(raw)
        t0 = time.perf_counter()
        tk = core.generate(lab, cache, off, ch, np.random.default_rng(core.stable_seed(rid)))
        gen_s = time.perf_counter() - t0
        audio = core.decode(lab, tk)
        speech, qc = core.engine_qc(lab, audio, ch)
        t1 = time.perf_counter()
        sig = core.signals(lab, tk, speech, raw)
        sig_s = time.perf_counter() - t1
        return job, tk, speech, qc, sig, gen_s, sig_s

    t_start = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in as_completed(futs):
            job, tk, speech, qc, sig, gen_s, sig_s = f.result()
            rid, c, ui, si, ch, lang, v, k = job
            t2 = time.perf_counter()
            hyp, cer = core.judge(lab, speech, ch, lang)
            asr_s = time.perf_counter() - t2
            safe = rid.replace("|", "__")
            np.save(out / "lat" / f"{safe}.npy", tk.z.astype(np.float16))
            row = {"rid": rid, "set": a.set, "prompt_kind": a.prompt_kind, "case": c["id"], "cat": c["cat"], "lang": lang,
                   "voice": v, "unit": ui, "sent": si, "take": k, "chunk": ch,
                   "words": len(ch.split()), **qc, **sig, "hyp": hyp, "cer": round(cer, 4),
                   "gen_s": round(gen_s, 3), "sig_s": round(sig_s, 4), "asr_s": round(asr_s, 3)}
            core.append_jsonl(rows_path, row)
            n += 1
            if n % 50 == 0 or n == len(jobs):
                el = time.time() - t_start
                print(f"{n}/{len(jobs)} {el/60:.1f} min, eta {el/n*(len(jobs)-n)/60:.1f} min", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
