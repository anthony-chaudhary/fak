#!/usr/bin/env python
"""llama.cpp BATCHED (multi-sequence) CPU decode throughput — the honest peer for
fak's in-kernel multi-user batched decode (internal/model.BatchSession, cmd/batchbench).

MODEL-BASELINE-RESULTS.md established that single-stream batch-1 decode is at parity
(fak Q8 ~7.7 ms/tok vs llama.cpp Q8 6.9). MODEL-BATCHING-RESULTS.md then showed fak's
in-kernel multi-user batching reaches ~862 tok/s aggregate at B=512. To compare that to
llama.cpp HONESTLY we must measure llama.cpp's OWN batched throughput — not its single
stream. This benchmark does exactly that, using llama.cpp's low-level multi-sequence
batch API (the same continuous-batching path llama-server's parallel slots drive): B
independent sequences, one decode token each per llama_decode() call, logits computed
for ALL B (every sequence needs its next-token logits each step, exactly as fak's
StepBatch returns per-user logits).

Apples-to-apples with cmd/batchbench:
  - same model (SmolLM2-135M GGUF), same prompt length (16) and decode steps (12),
  - aggregate tok/s = B / per-step-wall-clock, best (min) per-step over reps
    (least-contended sampling — contention only ever slows a step),
  - deterministic LCG token ids (token VALUES never affect matmul/attention cost),
  - all-thread (n_threads = cpu_count), the realistic serving config.

Usage:
  python internal/model/bench_llamacpp_batched.py \
    --gguf experiments/model-baseline/gguf/SmolLM2-135M-Instruct-Q8_0.gguf \
    --out experiments/model-baseline/llamacpp-batched-q8.json
"""
import argparse, json, os, sys, time, ctypes
import llama_cpp as L

PROMPT_LEN = 16
DECODE_STEPS = 12


def lcg_ids(n, vocab, seed=0):
    ids, state = [], (2463534242 + seed) & 0xFFFFFFFF
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        ids.append(state % vocab)
    return ids


def make_ctx(model, B, n_ctx, threads):
    cp = L.llama_context_default_params()
    cp.n_ctx = n_ctx
    cp.n_batch = max(2048, B)     # max tokens submitted per llama_decode
    cp.n_ubatch = max(512, B)     # physical micro-batch; must hold B decode tokens
    cp.n_seq_max = B
    cp.n_threads = threads
    cp.n_threads_batch = threads
    cp.logits_all = False
    return L.llama_new_context_with_model(model, cp)


def fill_decode_batch(batch, tokens, positions, B):
    """One decode token per sequence; logits for ALL B (serving step)."""
    batch.n_tokens = B
    for i in range(B):
        batch.token[i] = tokens[i]
        batch.pos[i] = positions[i]
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = i
        batch.logits[i] = 1


def prefill_all(ctx, model, B, prompt_ids, batch_cap):
    """Prefill each of B sequences with the same `prompt_ids`, packed into chunks
    of <= batch_cap tokens. Not timed; only populates the KV cache."""
    batch = L.llama_batch_init(batch_cap, 0, B)
    P = len(prompt_ids)
    # (seq, pos) work list
    work = [(s, p) for s in range(B) for p in range(P)]
    idx = 0
    while idx < len(work):
        chunk = work[idx:idx + batch_cap]
        batch.n_tokens = len(chunk)
        for j, (s, p) in enumerate(chunk):
            batch.token[j] = prompt_ids[p]
            batch.pos[j] = p
            batch.n_seq_id[j] = 1
            batch.seq_id[j][0] = s
            batch.logits[j] = 0
        rc = L.llama_decode(ctx, batch)
        if rc != 0:
            L.llama_batch_free(batch)
            raise RuntimeError(f"prefill llama_decode rc={rc} (chunk={len(chunk)})")
        idx += len(chunk)
    L.llama_batch_free(batch)


def bench_batch(model, vocab, B, threads, reps):
    n_ctx = B * (PROMPT_LEN + DECODE_STEPS) + 8
    ctx = make_ctx(model, B, n_ctx, threads)
    if not ctx:
        raise RuntimeError("llama_new_context_with_model failed")
    try:
        prompt_ids = lcg_ids(PROMPT_LEN, vocab)
        decode_seed_ids = lcg_ids(B, vocab, seed=991)  # one starting next-token per seq
        best = None
        for r in range(reps):
            # fresh KV each rep so positions line up
            if hasattr(L, "llama_memory_clear"):
                L.llama_memory_clear(L.llama_get_memory(ctx), True)
            elif hasattr(L, "llama_kv_cache_clear"):
                L.llama_kv_cache_clear(ctx)
            prefill_all(ctx, model, B, prompt_ids, max(2048, B))
            batch = L.llama_batch_init(B, 0, B)
            tokens = list(decode_seed_ids)
            positions = [PROMPT_LEN] * B
            t0 = time.perf_counter()
            for step in range(DECODE_STEPS):
                fill_decode_batch(batch, tokens, positions, B)
                rc = L.llama_decode(ctx, batch)
                if rc != 0:
                    L.llama_batch_free(batch)
                    raise RuntimeError(f"decode llama_decode rc={rc} B={B} step={step}")
                # advance: deterministic next token + position (values don't affect cost)
                for i in range(B):
                    tokens[i] = (tokens[i] * 48271 + 1) % vocab
                    positions[i] += 1
            elapsed = time.perf_counter() - t0
            L.llama_batch_free(batch)
            step_ms = (elapsed / DECODE_STEPS) * 1e3
            if best is None or step_ms < best:
                best = step_ms
            sys.stderr.write(f"[llamacpp-batched B={B} t{threads} rep{r}] step={step_ms:.3f} ms "
                             f"agg={B/(step_ms/1e3):.1f} tok/s\n")
            sys.stderr.flush()
        return best
    finally:
        L.llama_free(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    a = ap.parse_args()
    batches = [int(x) for x in a.batches.split(",") if x.strip()]

    L.llama_backend_init()
    mp = L.llama_model_default_params()
    mp.n_gpu_layers = 0
    model = L.llama_model_load_from_file(a.gguf.encode(), mp)
    if not model:
        sys.stderr.write(f"failed to load {a.gguf}\n"); sys.exit(1)
    vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(model))

    naive_serial_ms = 52.1
    points, peak = [], None
    b1_tok_s = None
    for B in batches:
        try:
            step_ms = bench_batch(model, vocab, B, a.threads, a.reps)
        except Exception as e:
            sys.stderr.write(f"B={B} FAILED: {type(e).__name__} {str(e)[:200]}\n")
            continue
        agg = B / (step_ms / 1e3)
        if B == 1:
            b1_tok_s = agg
        pt = {"batch": B, "decode_steps": DECODE_STEPS, "reps": a.reps,
              "step_ms": step_ms, "per_user_ms_per_tok": step_ms / B,
              "agg_tok_per_sec": agg,
              "speedup_vs_b1": (agg / b1_tok_s) if b1_tok_s else None,
              "speedup_vs_naive_serial": agg / (1000.0 / naive_serial_ms)}
        points.append(pt)
        if peak is None or agg > peak["agg_tok_per_sec"]:
            peak = {"batch": B, "agg_tok_per_sec": agg,
                    "speedup_vs_b1": pt["speedup_vs_b1"],
                    "speedup_vs_naive_serial": pt["speedup_vs_naive_serial"]}

    report = {"engine": f"llama.cpp batched multi-sequence decode ({os.path.basename(a.gguf)})",
              "model": "SmolLM2-135M (GGUF)", "version": L.__version__,
              "threads": a.threads, "n_vocab": vocab,
              "prompt_len": PROMPT_LEN, "decode_steps": DECODE_STEPS,
              "naive_serial_ms_per_tok": naive_serial_ms,
              "baseline_b1_tok_per_sec": b1_tok_s,
              "peak": peak, "points": points}
    blob = json.dumps(report, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(blob)
        sys.stderr.write(f"wrote {a.out}\n")
    else:
        print(blob)
    L.llama_model_free(model)


if __name__ == "__main__":
    main()
