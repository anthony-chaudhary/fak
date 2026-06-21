#!/usr/bin/env python
"""llama.cpp direct T x A agent-serving benchmark.

This is the low-level llama.cpp peer for:

  go run ./cmd/fleetserve -quant -turns 1,2,4 -concurrency 1,4,8 ...

Workload shape:
  - prefill one shared P-token system/tool prefix into sequence 0;
  - copy that prefix into A agent sequences with llama_memory_seq_cp;
  - for each of T turns, decode D assistant tokens for all A agents in one batch;
  - between turns, ingest R private tool/result tokens per agent.

The result is a direct turns x agents surface against llama.cpp's own multi-sequence
KV machinery, not an inference from fak's no-reuse ablation.
"""
import argparse
import json
import os
import sys
import time

import llama_cpp as L


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def lcg(n, vocab, seed=0):
    state = (2463534242 + seed) & 0xFFFFFFFF
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(state % vocab)
    return out


def clear_memory(ctx):
    if hasattr(L, "llama_memory_clear"):
        L.llama_memory_clear(L.llama_get_memory(ctx), True)
    elif hasattr(L, "llama_kv_cache_clear"):
        L.llama_kv_cache_clear(ctx)
    else:
        raise RuntimeError("llama.cpp binding has no memory-clear API")


def decode_batch(ctx, work, batch_cap, n_seq_max):
    """Submit work entries: (token, pos, seq_id, logits_flag)."""
    batch = L.llama_batch_init(batch_cap, 0, n_seq_max)
    try:
        off = 0
        while off < len(work):
            chunk = work[off : off + batch_cap]
            batch.n_tokens = len(chunk)
            for i, (tok, pos, seq, logits) in enumerate(chunk):
                batch.token[i] = tok
                batch.pos[i] = pos
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq
                batch.logits[i] = 1 if logits else 0
            rc = L.llama_decode(ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode rc={rc} chunk={len(chunk)}")
            off += len(chunk)
    finally:
        L.llama_batch_free(batch)


def prefill_prefix(ctx, prefix, batch_cap, agents):
    work = []
    last = len(prefix) - 1
    for i, tok in enumerate(prefix):
        work.append((tok, i, 0, i == last))
    t0 = time.perf_counter()
    decode_batch(ctx, work, batch_cap, agents)
    return time.perf_counter() - t0


def copy_prefix(ctx, prefix_len, agents):
    mem = L.llama_get_memory(ctx)
    t0 = time.perf_counter()
    for seq in range(1, agents):
        L.llama_memory_seq_cp(mem, 0, seq, 0, prefix_len)
    return time.perf_counter() - t0


def result_ids(result_tokens, vocab, turn, agent, rep):
    return lcg(result_tokens, vocab, seed=10_000 + rep * 1_000_000 + turn * 10_000 + agent * 97)


def run_turns(ctx, vocab, prefix_len, turns, agents, decode_steps, result_tokens, batch_cap, rep):
    ids = lcg(agents, vocab, seed=991)
    pos = [prefix_len] * agents
    decode_total = 0.0
    result_total = 0.0

    for turn in range(turns):
        batch = L.llama_batch_init(max(batch_cap, agents), 0, agents)
        try:
            t0 = time.perf_counter()
            for _ in range(decode_steps):
                batch.n_tokens = agents
                for i in range(agents):
                    batch.token[i] = ids[i]
                    batch.pos[i] = pos[i]
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = i
                    batch.logits[i] = 1
                rc = L.llama_decode(ctx, batch)
                if rc != 0:
                    raise RuntimeError(f"decode rc={rc} turn={turn}")
                for i in range(agents):
                    ids[i] = (ids[i] * 48271 + 1) % vocab
                    pos[i] += 1
            decode_total += time.perf_counter() - t0
        finally:
            L.llama_batch_free(batch)

        if turn + 1 < turns and result_tokens > 0:
            work = []
            for agent in range(agents):
                toks = result_ids(result_tokens, vocab, turn, agent, rep)
                for r, tok in enumerate(toks):
                    # Result-ingest is teacher-forced context growth; the next-token
                    # distribution after the result is not consumed by this benchmark.
                    work.append((tok, pos[agent] + r, agent, False))
            t0 = time.perf_counter()
            decode_batch(ctx, work, batch_cap, agents)
            result_total += time.perf_counter() - t0
            for agent in range(agents):
                pos[agent] += result_tokens

    return decode_total, result_total


def make_ctx(model, prefix_len, turns, agents, decode_steps, result_tokens, threads, batch_cap):
    per_agent_tail = turns * decode_steps + max(0, turns - 1) * result_tokens
    n_ctx = prefix_len + agents * per_agent_tail + 64
    cp = L.llama_context_default_params()
    cp.n_ctx = n_ctx
    cp.n_batch = batch_cap
    cp.n_ubatch = max(512, agents)
    cp.n_seq_max = agents
    cp.n_threads = threads
    cp.n_threads_batch = threads
    if hasattr(cp, "kv_unified"):
        cp.kv_unified = True
    ctx = L.llama_new_context_with_model(model, cp)
    if not ctx:
        raise RuntimeError(f"context allocation failed (T={turns}, A={agents}, n_ctx={n_ctx})")
    return ctx, n_ctx


def bench_one(model, vocab, prefix_len, turns, agents, decode_steps, result_tokens, threads, reps, batch_cap):
    ctx, n_ctx = make_ctx(model, prefix_len, turns, agents, decode_steps, result_tokens, threads, batch_cap)
    prefix = lcg(prefix_len, vocab, seed=1)
    best = None
    best_parts = None
    try:
        for rep in range(reps):
            clear_memory(ctx)
            pre = prefill_prefix(ctx, prefix, batch_cap, agents)
            cp = copy_prefix(ctx, prefix_len, agents)
            dec, res = run_turns(
                ctx, vocab, prefix_len, turns, agents, decode_steps, result_tokens, batch_cap, rep
            )
            total = pre + cp + dec + res
            if best is None or total < best:
                best = total
                best_parts = (pre, cp, dec, res)
            sys.stderr.write(
                f"[llamacpp-turn-agent T={turns} A={agents} rep={rep}] "
                f"pre={pre*1e3:.0f}ms cp={cp*1e3:.2f}ms dec={dec*1e3:.0f}ms "
                f"res={res*1e3:.0f}ms turns/s={(turns*agents)/total:.2f}\n"
            )
            sys.stderr.flush()
    finally:
        L.llama_free(ctx)

    pre, cp, dec, res = best_parts
    total_ms = best * 1e3
    return {
        "turns": turns,
        "agents": agents,
        "prefix_len": prefix_len,
        "decode_steps": decode_steps,
        "result_tokens_between_turns": result_tokens,
        "reps": reps,
        "n_ctx": n_ctx,
        "prefill_ms": pre * 1e3,
        "seqcp_ms": cp * 1e3,
        "decode_ms": dec * 1e3,
        "result_prefill_ms": res * 1e3,
        "total_ms": total_ms,
        "agents_per_sec": agents / best,
        "agent_turns_per_sec": (agents * turns) / best,
    }


def warm_model(model, vocab, threads, batch_cap):
    ctx, _ = make_ctx(model, 16, 1, 1, 1, 0, threads, max(batch_cap, 16))
    try:
        clear_memory(ctx)
        prefill_prefix(ctx, lcg(16, vocab, seed=77), max(batch_cap, 16), 1)
        run_turns(ctx, vocab, 16, 1, 1, 1, 0, max(batch_cap, 16), 0)
    finally:
        L.llama_free(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--prefix", type=int, default=1024)
    ap.add_argument("--turns", default="1,2,4")
    ap.add_argument("--agents", default="1,4,8,16")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--result", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--batch-cap", type=int, default=2048)
    args = ap.parse_args()

    if not hasattr(L, "llama_memory_seq_cp"):
        raise SystemExit("llama.cpp binding lacks llama_memory_seq_cp; cannot run direct shared-prefix peer")

    turns_grid = parse_ints(args.turns)
    agent_grid = parse_ints(args.agents)
    batch_cap = max(args.batch_cap, args.prefix, max(agent_grid or [1]))

    L.llama_backend_init()
    mp = L.llama_model_default_params()
    mp.n_gpu_layers = 0
    model = L.llama_model_load_from_file(args.gguf.encode(), mp)
    if not model:
        raise SystemExit(f"failed to load {args.gguf}")
    vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(model))
    warm_model(model, vocab, args.threads, batch_cap)

    points = []
    try:
        for turns in turns_grid:
            if turns < 1:
                continue
            for agents in agent_grid:
                if agents < 1:
                    continue
                try:
                    points.append(
                        bench_one(
                            model,
                            vocab,
                            args.prefix,
                            turns,
                            agents,
                            args.decode,
                            args.result,
                            args.threads,
                            args.reps,
                            batch_cap,
                        )
                    )
                except Exception as exc:
                    sys.stderr.write(f"T={turns} A={agents} FAILED: {type(exc).__name__} {str(exc)[:200]}\n")
    finally:
        L.llama_model_free(model)

    report = {
        "engine": f"llama.cpp turn-agent shared-prefix ({os.path.basename(args.gguf)})",
        "model": "SmolLM2-135M (GGUF)",
        "version": getattr(L, "__version__", "unknown"),
        "threads": args.threads,
        "prefix_len": args.prefix,
        "turn_grid": turns_grid,
        "agent_grid": agent_grid,
        "decode_steps_per_turn": args.decode,
        "result_tokens_between_turns": args.result,
        "points": points,
    }
    blob = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(blob + "\n")
        sys.stderr.write(f"wrote {args.out}\n")
    else:
        print(blob)


if __name__ == "__main__":
    main()
