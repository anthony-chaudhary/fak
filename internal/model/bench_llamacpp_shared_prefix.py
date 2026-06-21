#!/usr/bin/env python
"""llama.cpp cross-agent SHARED-PREFIX fleet decode — the FAIR peer for fak fleetserve.

This is the honest llama.cpp number for the agentic-fleet workload (C concurrent agents
sharing one P-token prefix, each generating D tokens). It uses llama.cpp's OWN cross-sequence
KV sharing — the `kv_unified=true` context + `llama_memory_seq_cp` — to prefill the shared
prefix ONCE and splice it into all C sequences, then batched-decode them. (Verified: with the
default per-sequence (non-unified) cache this either needs C independent prefix prefills or the
partial seq_cp asserts `is_full`; the unified cache is the config that gives prefill-once +
share + batched decode. We measure llama.cpp at its BEST so the head-to-head is not rigged.)

Timed end-to-end per C: prefill(P) once + seq_cp(prefix → C-1 sequences) + D batched decode
steps. agents/sec = C / total_wall_clock, best (min) over reps. Mirrors cmd/fleetserve's
reuse path exactly, so the two JSONs compare directly.

Usage:
  python internal/model/bench_llamacpp_shared_prefix.py \
    --gguf experiments/model-baseline/gguf/SmolLM2-135M-Instruct-Q8_0.gguf \
    --prefix 1024 --decode 32 --concurrency 1,8,16,32,64 \
    --out experiments/model-baseline/llamacpp-shared-prefix-q8.json
"""
import argparse, json, os, sys, time
import llama_cpp as L


def lcg(n, vocab, seed=0):
    s = (2463534242 + seed) & 0xFFFFFFFF
    out = []
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(s % vocab)
    return out


def bench_one(model, vocab, P, D, C, threads, reps):
    n_ctx = P + C * D + 64
    cp = L.llama_context_default_params()
    cp.n_ctx = n_ctx
    cp.n_batch = max(2048, C)
    cp.n_ubatch = max(512, C)
    cp.n_seq_max = C
    cp.n_threads = threads
    cp.n_threads_batch = threads
    if hasattr(cp, "kv_unified"):
        cp.kv_unified = True   # the config that allows partial seq_cp prefix sharing
    ctx = L.llama_new_context_with_model(model, cp)
    if not ctx:
        raise RuntimeError(f"ctx alloc failed (C={C}, n_ctx={n_ctx})")
    try:
        prefix = lcg(P, vocab)
        seed_ids = lcg(C, vocab, seed=991)
        best = None
        best_parts = None
        for r in range(reps):
            mem = L.llama_get_memory(ctx)
            L.llama_memory_clear(mem, True)
            # prefill the shared prefix ONCE into seq 0
            b = L.llama_batch_init(P, 0, 1)
            b.n_tokens = P
            for i in range(P):
                b.token[i] = prefix[i]; b.pos[i] = i
                b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0
            t0 = time.perf_counter()
            rc = L.llama_decode(ctx, b)
            pre = time.perf_counter() - t0
            L.llama_batch_free(b)
            if rc != 0:
                raise RuntimeError(f"prefill rc={rc} (C={C})")
            # share prefix [0,P) into seqs 1..C-1 (cells shared, not duplicated)
            t0 = time.perf_counter()
            for s in range(1, C):
                L.llama_memory_seq_cp(mem, 0, s, 0, P)
            cpt = time.perf_counter() - t0
            # batched decode all C sequences for D steps
            bb = L.llama_batch_init(C, 0, C)
            toks = list(seed_ids); pos = [P] * C
            t0 = time.perf_counter()
            for step in range(D):
                bb.n_tokens = C
                for i in range(C):
                    bb.token[i] = toks[i]; bb.pos[i] = pos[i]
                    bb.n_seq_id[i] = 1; bb.seq_id[i][0] = i; bb.logits[i] = 1
                rc = L.llama_decode(ctx, bb)
                if rc != 0:
                    L.llama_batch_free(bb)
                    raise RuntimeError(f"decode rc={rc} (C={C} step={step})")
                for i in range(C):
                    toks[i] = (toks[i] * 48271 + 1) % vocab; pos[i] += 1
            dec = time.perf_counter() - t0
            L.llama_batch_free(bb)
            tot = pre + cpt + dec
            if best is None or tot < best:
                best = tot
                best_parts = (pre, cpt, dec)
            sys.stderr.write(f"[llamacpp-shared C={C} t{threads} rep{r}] pre={pre*1e3:.0f}ms "
                             f"cp={cpt*1e3:.2f}ms dec={dec*1e3:.0f}ms agents/s={C/tot:.1f}\n")
            sys.stderr.flush()
        pre, cpt, dec = best_parts
        return {"concurrency": C, "prefix_len": P, "decode_steps": D, "reps": reps,
                "prefill_ms": pre * 1e3, "seqcp_ms": cpt * 1e3, "decode_ms": dec * 1e3,
                "total_ms": best * 1e3, "agents_per_sec": C / best}
    finally:
        L.llama_free(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--concurrency", default="1,8,16,32,64")
    a = ap.parse_args()
    concs = [int(x) for x in a.concurrency.split(",") if x.strip()]

    L.llama_backend_init()
    mp = L.llama_model_default_params(); mp.n_gpu_layers = 0
    model = L.llama_model_load_from_file(a.gguf.encode(), mp)
    if not model:
        sys.stderr.write(f"load failed {a.gguf}\n"); sys.exit(1)
    vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(model))

    points = []
    for C in concs:
        try:
            points.append(bench_one(model, vocab, a.prefix, a.decode, C, a.threads, a.reps))
        except Exception as e:
            sys.stderr.write(f"C={C} FAILED: {type(e).__name__} {str(e)[:200]}\n")
    report = {"engine": f"llama.cpp shared-prefix (unified KV + seq_cp, {os.path.basename(a.gguf)})",
              "model": "SmolLM2-135M (GGUF)", "version": L.__version__,
              "threads": a.threads, "prefix_len": a.prefix, "decode_steps": a.decode,
              "points": points}
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
