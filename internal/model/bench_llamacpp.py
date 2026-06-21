#!/usr/bin/env python
"""llama.cpp CPU latency baseline — the RIGHT single-stream peer for the fak core.

The SOTA-landscape research (experiments/model-baseline/) concluded: vLLM/SGLang are
GPU continuous-batching SERVING engines (a different regime, not fak's claim), but
llama.cpp is the apples-to-apples peer for CPU single-stream latency — same hardware
regime, same batch=1 autoregressive decode. The only axis difference is precision:
llama.cpp runs quantized/F16 GGML kernels; fak runs f32. We measure F16, Q8_0, and
Q4_K_M so the precision axis is explicit, not hidden.

Same protocol as bench_hf.py / cmd/modelbench: prefill over P in {16,64,256} tokens
and D=32 incremental decode steps, wall-clock median, single-thread and all-thread.
We feed token IDS via the low-level eval loop so timing is the forward pass, not
sampling/detokenization overhead.

Usage:
  python internal/model/bench_llamacpp.py --gguf-dir experiments/model-baseline/gguf \
         --out experiments/model-baseline/llamacpp.json
"""
import argparse, json, os, sys, time, statistics, glob
from llama_cpp import Llama

PREFILL_SIZES = [16, 64, 256]
DECODE_PROMPT = 16
DECODE_STEPS = 32


def lcg_ids(n, vocab):
    ids, state = [], 2463534242
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        ids.append(state % vocab)
    return ids


def prefill_ms(llm, ids, reps):
    times = []
    for _ in range(reps):
        llm.reset()
        llm.n_tokens = 0
        t = time.perf_counter()
        llm.eval(ids)
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def decode_per_token_ms(llm, prompt_ids, steps, vocab, reps):
    per_tok = []
    for r in range(reps):
        llm.reset()
        llm.n_tokens = 0
        llm.eval(prompt_ids)
        nid = (r * 131 + 7) % vocab
        t = time.perf_counter()
        for _ in range(steps):
            llm.eval([nid])
            nid = (nid * 48271 + 1) % vocab
        per_tok.append((time.perf_counter() - t) / steps)
    return statistics.median(per_tok) * 1e3


def run_one(path, threads, reps):
    name = os.path.basename(path)
    sys.stderr.write(f"[llama.cpp] loading {name} (threads={threads})...\n"); sys.stderr.flush()
    llm = Llama(model_path=path, n_ctx=512, n_threads=threads, n_batch=512,
                logits_all=False, verbose=False)
    vocab = llm.n_vocab()
    # warm up (faults weights, primes kernels)
    llm.reset(); llm.n_tokens = 0; llm.eval(lcg_ids(8, vocab))
    cfg = {"gguf": name, "threads": threads, "n_vocab": vocab, "prefill": []}
    for p in PREFILL_SIZES:
        ms = prefill_ms(llm, lcg_ids(p, vocab), reps)
        cfg["prefill"].append({"tokens": p, "reps": reps, "median_ms": ms,
                               "tok_per_sec": p / (ms / 1e3)})
        sys.stderr.write(f"[llama.cpp:{name} t{threads}] prefill P={p}: {ms:.1f} ms ({p/(ms/1e3):.1f} tok/s)\n")
        sys.stderr.flush()
    dms = decode_per_token_ms(llm, lcg_ids(DECODE_PROMPT, vocab), DECODE_STEPS, vocab, reps)
    cfg["decode"] = {"prompt_tokens": DECODE_PROMPT, "decode_steps": DECODE_STEPS, "reps": reps,
                     "per_token_median_ms": dms, "tok_per_sec": 1.0 / (dms / 1e3)}
    sys.stderr.write(f"[llama.cpp:{name} t{threads}] decode: {dms:.1f} ms/tok ({1.0/(dms/1e3):.1f} tok/s)\n")
    sys.stderr.flush()
    del llm
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", default="experiments/model-baseline/gguf")
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ncpu = os.cpu_count() or 1

    ggufs = sorted(glob.glob(os.path.join(a.gguf_dir, "*.gguf")))
    if not ggufs:
        sys.stderr.write(f"no GGUF in {a.gguf_dir}\n"); sys.exit(1)

    import llama_cpp
    report = {"engine": "llama.cpp (llama-cpp-python prebuilt CPU wheel)",
              "model": "SmolLM2-135M-Instruct (GGUF)", "version": llama_cpp.__version__,
              "cpu_count": ncpu, "configs": []}
    # Each GGUF: single-thread (algorithm-class peer to fak) + all-thread (realistic).
    for path in ggufs:
        for threads in (1, ncpu):
            try:
                report["configs"].append(run_one(path, threads, a.reps))
            except Exception as e:
                sys.stderr.write(f"FAIL {path} t{threads}: {type(e).__name__} {str(e)[:200]}\n")

    blob = json.dumps(report, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(blob)
        sys.stderr.write(f"wrote {a.out}\n")
    else:
        print(blob)


if __name__ == "__main__":
    main()
