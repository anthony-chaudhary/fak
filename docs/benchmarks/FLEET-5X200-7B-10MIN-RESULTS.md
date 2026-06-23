---
title: "Verify it yourself: 5 agents × 200 turns of 7B work in under 10 minutes on a MacBook"
description: "A standalone, reproducible page. Measured on an Apple M3 Pro with Qwen2.5-7B Q8 (llama.cpp Metal): the 5-agent × 200-turn fleet lands at ~8.2 min — under 10 — when you batch the agents' decode and reuse the shared prefix. Run the two commands below and check the numbers."
---

# Five agents, two hundred turns each, one 7B — under ten minutes (verify it yourself)

> **This page is standalone and falsifiable.** Every number below comes from two commands you can
> run on any Apple-Silicon Mac with a 7B GGUF (see **Verify it yourself**). Nothing here is a
> private harness or a number you have to take on faith. Measured **2026-06-22** on an Apple **M3
> Pro** (the `node-macos-a` bench host), **Qwen2.5-7B-Instruct Q8_0**, llama.cpp Metal (`-ngl 99`),
> flash-attention on. Raw logs: `experiments/session/macbook-m3pro-7b-batched-{bench,ctx}.log`.

## The claim

Five agents doing **related work** (one shared 2,048-token preamble: system prompt + tool schemas
+ a shared task brief), each running **200 turns** (decode 20 tokens, ingest a 12-token tool result,
repeat), on a **7B** model on a **MacBook Pro**, finish in **under 10 minutes** — *if* the agents'
decode is **batched** into one weight stream and the shared prefix is **prefilled once**. Run them
the naive way and the same workload is **hours**.

| how you run 5×200 turns of 7B | wall-clock (M3 Pro, measured throughput) | under 10 min? |
|---|---:|:---:|
| **batched + shared prefix** (the fak-fused / continuous-batching pattern) | **~8.2 min** | ✅ |
| 5 independent single-stream sessions (warm KV, **no** cross-agent batching) | ~20.1 min | ✗ |
| naive stateless — re-prefill the whole context every turn | **≥ 4.0 hours** | ✗ |

The gap between row 1 and row 2 is **decode batching** (2.5×). The gap to row 3 is **not
re-prefilling** (≥30×). Both are what the fak kernel does; the absolute minutes are what an M3 Pro's
Metal 7B forward actually delivers.

> ⚠️ **Which forward — read this before reading "under 10 minutes" as a fak claim.** The 8.2 min
> above is the **host's Metal forward** (llama.cpp `-ngl 99`) running fak's *batched-and-shared
> pattern*. It is **not** the *pure* fak kernel doing the matmuls. fak's **own** forward on a Mac is
> pure-Go **CPU** (8.7 t/s decode / 16 t/s prefill on this M3 Pro — its Metal *decode* lane is still
> open, `internal/metalgemm`), so the identical 5×200 fleet **on fak's own forward is ~22–51 min,
> well over the bar**. The pure fak kernel reaches sub-10-min only where its forward is GPU-class —
> its **CUDA** path (A100 `sm_80`, or the committed RTX-4070 decode-parity result) — **not** on a
> CPU-only Mac. So: *the fleet fits in 10 min on a MacBook's Metal forward with fak's reuse pattern;
> the **pure-fak-kernel-on-a-Mac** version does not, until the Metal decode lane lands.* fak's
> contribution is the reuse + batching + per-agent KV ownership + safety floor — not the raw t/s.

## What we measured (the raw M3 Pro 7B numbers)

`llama-batched-bench` on the 7B Q8, prefix length 2,048, generating at parallelism 1 / 2 / 5:

| parallel agents (B) | prefill t/s | decode t/s (aggregate) | decode per agent |
|---:|---:|---:|---:|
| 1 | 391.9 | 17.41 | 17.41 |
| 2 | 392.1 | 33.92 | 16.96 |
| **5** | 392.1 | **47.2** | 9.4 |

Decode batching is real but **sub-linear**: 5 agents share one weight-read per step, so aggregate
throughput rises 17.4 → 47.2 t/s (**2.71×**, not the ideal 5×) — the weights are re-read once for
five tokens until attention/compute catch up. Across the fleet's growing context the 5-way rate
eases from **47.2 t/s @ 2k → 44.4 @ 5k → 41.9 @ 8k** (context sweep, same harness), so we use the
context-averaged **~44 t/s** for the fleet decode floor below. (Measuring beat guessing here: a
naive "batch = 5×" assumption would have under-counted the fleet by ~40%.)

## The fleet wall-clock (arithmetic over the measured rates)

A 5-agent × 200-turn session has exactly (pure arithmetic — `tools/fleet_10min_projection.py
--selftest`):

- **decode:** 5 × 200 × 20 = **20,000** token-decodes.
- **prefill, batched + shared (arm C):** prefix once + incremental results = `P + C·(T−1)·R` =
  2,048 + 5·199·12 = **13,988** tokens.
- **prefill, naive (arm A):** re-prefill the whole context every turn = `C·Σₜ(P+t·(D+R))` =
  **5,232,000** tokens (191× more than batched+shared once you fold in cross-agent sharing; **374×**
  vs arm C exactly).

Apply the measured M3 Pro rates:

| arm | prefill | decode | **total** | verdict |
|---|---:|---:|---:|---|
| **C — batched + shared prefix** | 13,988 / 392 = 36 s | 20,000 / 44 = **7.6 min** | **8.2 min** | ✅ under 10 |
| B — 5 single-stream sessions | 22,180 / 392 = 57 s | 20,000 / 17.41 = 19.1 min | 20.1 min | ✗ |
| A — naive re-prefill | ≥ 5,232,000 / 392 = 3.7 h | 19.1 min | ≥ 4.0 h | ✗ |

(arm A prefill is a **lower bound** — flat t/s ignores the O(L²) growth of prefill self-attention,
which only makes it worse.) **arm C = 8.2 min, decode-floor-bound:** 7.6 of the 8.2 minutes is the
irreducible decode; reuse has shrunk prefill to a 36-second sliver.

## Verify it yourself

On any Apple-Silicon Mac with the 7B GGUF (`brew install llama.cpp`; grab `qwen2.5-7b-instruct-q8_0`):

```sh
# 1) the load-bearing measurement: 5-way batched decode throughput on the 7B
llama-batched-bench -m qwen2.5-7b-instruct-q8_0.gguf -ngl 99 -npp 2048 -ntg 512 -npl 1,2,5 -c 16384
#    -> read the S_TG (t/s) column for B=5 ; that is your fleet decode rate

# 2) turn the measured rates into the fleet wall-clock (exact token arithmetic + your rate card)
python3 tools/fleet_10min_projection.py --prefix 2048 --turns 200 --agents 5 --decode 20 --result 12
#    -> arm C (batched+shared) total ; compare to the 10-min bar
```

Want the *exact* per-turn fleet (shared prefix → copy to 5 seqs → per-turn batched decode → ingest
result), not just throughput? `pip install llama-cpp-python` (Metal) and run the committed peer
harness `internal/model/bench_llamacpp_turn_agents.py --gguf <7b> --prefix 2048 --turns 200
--agents 5 --decode 20 --result 12`. It does the literal 5×200 schedule.

## What the fak kernel actually contributes (honest scoping — read before quoting)

- **The 8.2 minutes is a *batching* result, not a fak-throughput miracle.** A tuned multi-tenant
  server (`llama-server --parallel 5`, vLLM, SGLang) batches too and would hit the same ~8 min.
  **fak is not faster than llama.cpp on raw t/s** (it is ~0.5× decode single-stream;
  `QWEN25-7B-RESULTS.md`). What fak adds is delivering the batched-fleet pattern **in one kernel,
  correctly, while preserving per-agent KV ownership** — every agent can `Evict`/`Clone` its own
  span, which a shared-slot serving engine structurally cannot offer — under a **default-deny
  safety floor** the model can't talk past. The speed is the host's forward; the *correctness +
  ownership + safety* of running five agents over one weight stream is the kernel.
- **Baseline-scoped multiples.** "≥30× vs naive" is vs re-prefilling every turn (a real hand-rolled
  `llama-cli -p <full prompt>` loop — but a worst case, not a serving baseline). "2.5× vs
  single-stream" is vs running 5 independent sessions with no cross-agent batching. Against a
  *batched* server, fak is ~parity on throughput — the win is ownership + safety + integration.
- **Measured vs arithmetic.** The throughputs (392 / 17.41 / 44 t/s) are **measured**. The fleet
  minutes are those rates × the **exact** token counts. Reproduce both with the commands above.

## The kernel really schedules the fleet (supporting, weightless, runs anywhere)

The pure fak kernel runs the full 5-agent × 200-turn loop — prefix-once, clone into 5, batched
decode, per-agent KV growth — with **no model weights at all** (`model.NewSynthetic`, whose
throughput is weight-value-independent; only the logits are meaningless):

```sh
go run ./cmd/sessionbench -synthetic smollm2-135m -turns 200 -agents 5 -prefix 2048 -decode 20 -result 12
```

On this run the same harness measured the prefill quadratic that makes the naive arm so expensive
(throughput **falls** with context, because prefill self-attention is O(L²)):

| prefill length L | 256 | 2,048 | 4,352 | 6,400 | 8,448 |
|---|---:|---:|---:|---:|---:|
| tok/s (SmolLM2-135M) | 676 | 751 | 475 | 338 | 266 |

A linear cost model would under-price the naive arm by ~2.5× at the session's tail; the harness sums
the *measured* curve over the exact per-turn contexts instead.

## Witnesses

| claim | witness | rules out |
|---|---|---|
| batched 7B decode = 44–47 t/s; single = 17.4 t/s | `experiments/session/macbook-m3pro-7b-batched-{bench,ctx}.log` (llama-batched-bench, M3 Pro) | "the throughput is modeled" |
| the fleet token counts are exact (374× / 1.59×) | `python3 tools/fleet_10min_projection.py --selftest` (cross-checked vs `headline-qwen-50x5.json`) | "the counts are fitted" |
| the kernel schedules 5×200 live with KV growth | run `go run ./cmd/sessionbench -synthetic smollm2-135m -turns 200 -agents 5 -prefix 2048 -decode 20 -result 12` (weightless, any box) | "the fleet shape was modeled, not run" |
| the prefill quadratic is measured (the curve above) | `experiments/session/fleet-5x200-run.log` (sessionbench -synthetic, this box) | "the naive arm's cost is extrapolated from one rate" |
| batched ≡ serial decode bit-for-bit (the win is reuse, not a numerics shortcut) | `go test ./internal/model -run 'TestBatchedDecodeMatchesSerial\|TestBatchFromPrefixMatchesIndependentPrefill'` | "fak computes something cheaper/wrong" |
| synthetic-model throughput is faithful | `internal/model/synthetic_perf_test.go` | "random weights give bogus tok/s" |

## DGX / A100 companion

The same fleet on the lab's **8× A100-SXM4-40GB** DGX (256 cores) is the bigger-iron companion —
tracked separately; the A100's batched 7B throughput is far higher than the M3 Pro's, so the
headroom under the 10-minute bar widens. (Status: dispatched via the control bridge; numbers land in
a follow-up once the A100 7B batched-bench completes.)

## Bottom line

**Measured on an M3 Pro:** five agents × two hundred turns each on a 7B finish in **~8.2 minutes** —
under the ten-minute bar — *because* the five agents' decode is batched into one weight stream
(measured 44 t/s vs 17.4 single) and the 2,048-token preamble is prefilled once instead of 374× over.
Run it the naive way and it is **≥ 4 hours**. The fak kernel is what delivers that batched-and-shared
pattern in one binary with per-agent KV ownership and a default-deny safety floor — verify every
number with the two commands above.
