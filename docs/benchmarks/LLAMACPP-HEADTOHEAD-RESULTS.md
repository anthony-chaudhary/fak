---
title: "fak vs llama.cpp head-to-head: CPU performance axes"
description: "Measures fak against llama.cpp on CPU across single-stream decode, prefill, batched throughput, and shared-prefix, with an honest parity-vs-gap verdict per axis."
---

# LLAMACPP-HEADTOHEAD-RESULTS — fak vs llama.cpp across every performance axis, measured

> **The honest verdict up front: fak CAN be a throughput peer of llama.cpp — and on the GPU
> it IS one.** With a reusable CUDA graph, fak reaches **decode parity with llama.cpp Q8_0
> (≈120 tok/s) on the RTX 4070** — a model that fits the GPU, at higher precision (fak f32) —
> measured head-to-head (see `GPU.md` §3b). So the idea that a pure-Go kernel "can't match a
> hand-tuned engine" is simply false; it's a tuning problem, not an architecture ceiling.
>
> This document measures the **CPU** axis on a no-GPU box, and there the picture is honest and
> different: **single-stream decode is parity (1.12×)**, **batched throughput is behind
> (2.26×)** — fak is not 2× *faster* than llama.cpp on this CPU, and the residual is a
> **hand-tuned-assembly / GEMM-tiling boundary, not architecture** (llama.cpp's GGML int8
> micro-kernel is hand-written assembly; fak's forward pass is pure Go + one hand-written int8
> kernel). It records the full CPU head-to-head — including two axes the prior model docs never
> measured against llama.cpp: **batched throughput** (solidly measured — fak behind) and
> **cross-agent shared-prefix** (a preliminary, settings-dependent probe — flagged below as an
> OPEN work item, not a settled conclusion). A direct **turns × agents** harness now measures
> the multi-turn shape against llama.cpp directly instead of inferring it from fak's no-reuse
> ablation; the realistic sweep targets 10-80 turns and 8-20 agents (the first 10-turn/8-agent
> probe has fak behind on CPU). It also records the one bit-identical optimization this
> investigation shipped (per-step allocation elimination in the Q8 batched decode).
>
> Model: SmolLM2-135M. Box: 16-core/32-thread Zen5, no GPU. Precision: Q8_0 (apples-to-apples;
> Q4_K_M where noted). llama.cpp = `llama-cpp-python` 0.3.30 CPU wheel (what `pip install
> llama-cpp-python` gives). All fak benchmarks run via WSL on Windows (native exec is WDAC-blocked here).
>
> **Cross-arch companion:** `M3-LLAMACPP-RESULTS.md` reruns this comparison on Apple M3 Pro
> (arm64) — it ships the NEON Q8 lane this doc's Zen5 amd64 lane already had, runs Qwen2.5-1.5B,
> and reaches the same honest verdict (no 2× axis; the residual is a hand-tuned-kernel /
> GEMM-tiling / Metal boundary, not architecture).
> Reproduce: `internal/model/bench_llamacpp_batched.py`, `bench_llamacpp_shared_prefix.py`,
> `internal/model/bench_llamacpp_turn_agents.py`, `cmd/batchbench`, `cmd/fleetserve`,
> `BenchmarkStepBatchQ`. Raw JSON under
> `experiments/model-baseline/`.

## The measured axes

| axis | llama.cpp | fak | verdict |
|---|---|---|---|
| **single-stream decode** (1-thread, Q8) | 6.9 ms/tok | 7.7 ms/tok | parity — fak **1.12× behind** (memory-bandwidth-bound; residual = an instruction Go can't emit + shared-box bandwidth) |
| **single-stream prefill** (per-token slope) | 0.337 ms/tok | 0.346 ms/tok | **parity** (1.03×) |
| **batched decode throughput** (peak, 32-thread) | **~2816 tok/s** (B=256) | 1132 → 1247 → **2916 tok/s** (B=960, after the fixes below) | **parity / slight fak lead** (1.04×) |
| **cross-agent shared-prefix** (peak agents/s, P=1024 D=32) | **17.2** (C=32) — *preliminary, settings-dependent* | 5.2 (C=32) / 6.0 (C=64) | **OPEN — needs verification** (see Axis 4) |
| **turns × agents bounded-long baseline** (P=1024 T=40/80 A=8/20 D=32 R=48, reps=1) | **1.59-2.71 agent-turns/s** | 0.58-0.87 agent-turns/s | fak **0.322-0.405× llama.cpp** on the long endpoint grid |

The single-stream rows are the prior `MODEL-BASELINE-RESULTS.md` numbers (re-confirmed). The
batched and shared-prefix rows are **new** — and they are where the prior narrative was wrong.

## Axis 3 — batched throughput: llama.cpp dominates (the narrative correction)

`MODEL-BATCHING-RESULTS.md` framed fak's multi-user batched decode (862 tok/s aggregate at
B=512, "44.9× the unbatched f32-serial baseline") as the throughput win for "the vLLM regime llama.cpp
doesn't target," and **never measured llama.cpp's own batched decode.** It does batch — very
well. `bench_llamacpp_batched.py` drives llama.cpp's low-level multi-sequence batch API (the
same continuous-batching path `llama-server`'s parallel slots use):

| B | llama.cpp Q8 agg tok/s | llama.cpp Q4 agg tok/s |
|---:|---:|---:|
| 1 | 170 | 233 |
| 8 | 1164 | 1273 |
| 16 | 1754 | — |
| 32 | 2263 | — |
| 256 | **2816** | **2828** (B=128) |

llama.cpp's batched CPU decode peaks at **~2816 tok/s**. The current fak Q8 batched lane now
reaches **2916 tok/s** at B=960 under `cmd/batchbench -quant -reps 4 -decode-steps 16`
(32 workers), after the allocation, attention, SwiGLU, and Q8 tile dispatch fixes below. That
is parity/slight lead on this axis, not a 2× win: fak's batching is now competitive with
llama.cpp's batched ceiling, while the remaining speed story still hinges on kernel tuning and
the other axes below.

## Axis 4 — cross-agent shared-prefix: OPEN WORK ITEM (do not treat as settled)

> **Status: preliminary, settings-dependent — a work item to verify, NOT a settled
> conclusion.** The numbers below come from a *single-config* probe and hinge on
> *non-default* llama.cpp settings with known edge-case constraints. Do not cite this as a
> definitive refutation of the cross-agent-KV-reuse thesis until it is verified robustly
> (multiple configs, larger scale, a real `llama-server` parallel-slot run — not just the
> `llama-cpp-python` low-level wheel — and a check that it is not fragile/version-specific).

The repo's cross-agent-KV-reuse thesis (and `docs/explainers/agentic-serving-related-art.md`)
positions "compute a shared prefix once, reuse its KV across agents" as something fak's
kernel-owned KV does that "a per-slot serving engine structurally cannot." I built fak's
explicit path for this — `NewBatchFromPrefix` (prefill once → `KVCache.Clone` into all C agents
→ batched decode, proven bit-identical by `TestBatchFromPrefixMatchesIndependentPrefill`) —
expecting a structural win. A probe then **suggested** llama.cpp may do it too, but only under
specific settings:

With `kv_unified=true` (**off by default** — the default partitions `n_ctx` per sequence and
needs C separate prefix prefills) + `llama_memory_seq_cp` (which **asserts `is_full`** on a
partial buffer in this build), one probe had llama.cpp prefill the shared prefix once (~420 ms
for P=1024) and share it into all C sequences for ~0.4–5 ms (`bench_llamacpp_shared_prefix.py`,
`rc=0`, prefill cost flat in C), then batched-decode:

| C (agents) | llama.cpp agents/s *(preliminary)* | fak agents/s | |
|---:|---:|---:|---:|
| 8 | 8.1 | 2.7 | |
| 16 | 12.3 | 3.3 | |
| 32 | **17.2** | 5.2 | |
| 64 | 11.9 | 6.0 | |

**If** this holds up under the verification above, llama.cpp would get cross-agent prefix
sharing and batched decode at once. Either way, the part that **does** stand: fak's `Clone`
deep-copies, so each agent can independently `Evict` a poisoned span from its own KV, whereas
`seq_cp` shares cells (no copy-on-write here) — a **security/correctness** capability that
*costs* fak performance, it does not add any.

## Axis 5 — direct turns × agents realistic sweep

The missing direct comparison is now executable. `cmd/fleetserve` accepts `-turns` and
`-result`, so fak runs the same shape the llama peer does: shared prefix once, T assistant
decode bursts, and per-agent result-token ingestion between turns. The llama side is
`internal/model/bench_llamacpp_turn_agents.py`, using `kv_unified=true` +
`llama_memory_seq_cp` and the low-level multi-sequence batch API. `tools/fak_llama_turn_agent_compare.py`
folds the two raw JSONs.

The old P=256/T=1-2 smoke grid is now treated as wiring-only evidence. A realistic agent run
is more like 10-80 turns across 8-20 live agents, but an 80-turn run cannot keep fat
per-turn result blobs in an 8k semantic sequence. The benchmark therefore uses two canonical
profiles and one cheap probe:

| profile | P | T grid | agents | D | R | reps | max per-agent ctx | purpose |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `probe` | 1024 | 10 | 8 | 32 | 64 | 1 | 1920 | quick validation cell, not canonical |
| `interactive` | 2048 | 10,20 | 8,12,16,20 | 64 | 256 | 3 | 8192 | active agents with fat tool/result context |
| `long` | 1024 | 40,80 | 8,12,16,20 | 32 | 48 | 3 | 7376 | long agents with compacted/rolled context |
| `bounded-long` | 1024 | 40,80 | 8,20 | 32 | 48 | 1 | 7376 | endpoint baseline when the full long grid is too slow |
| `mac-longer` | 2048 | 80,120 | 8,12,16,20 | 32 | 16 | 1 | 7792 | longer native-node probe for the Mac payload |

The per-agent context column is the semantic sequence length. The llama peer uses one unified
KV arena, so the total KV allocation also scales with agent count; that is storage geometry,
not a claim that a single agent sees more than the listed context.

Reproduce from repo root:

```bash
python tools/run_turn_agent_realistic_sweep.py --profile probe
python tools/run_turn_agent_realistic_sweep.py --profile bounded-long
python tools/run_turn_agent_realistic_sweep.py --profile interactive
python tools/run_turn_agent_realistic_sweep.py --profile long
```

`--profile all` runs both canonical grids. Use `--dry-run` first to print the exact commands
and context estimates without spending the run time.

`bounded-long` skips the no-reuse ablation (`fleetserve -ablation=false`) because the endpoint
baseline only needs fak reuse vs llama.cpp, and the no-reuse path makes the long/high-agent
cells several-hour runs on this host. FAK was run under WSL for this baseline; Windows-native
long cells repeatedly terminated without JSON after several minutes, matching the repo's
standing guidance to use WSL for fak model benchmarks on this workstation.

The compare artifact treats llama.cpp as the tuned reference, not as a naive/no-reuse arm.
`tools/fak_llama_turn_agent_compare.py` keeps `fak_reuse_speedup_vs_noreuse` only as an
internal ablation field and writes `fak_current_vs_tuned_reference` as the headline ratio.
If we want to discuss our own future tuning headroom, pass an explicit
`--candidate-tuning-mult <x>`; the generated `fak_projected_vs_tuned_reference` field is a
projection until a measured tuned FAK artifact replaces it.

Bounded long-context endpoint result:

| T | agents | fak agent-turns/s | llama.cpp agent-turns/s | fak / llama |
|---:|---:|---:|---:|---:|
| 40 | 8 | 0.663 | 1.920 | 0.345× |
| 40 | 20 | 0.874 | 2.713 | 0.322× |
| 80 | 8 | 0.581 | 1.650 | 0.352× |
| 80 | 20 | 0.643 | 1.587 | 0.405× |

Earlier realistic probe result:

| T | agents | fak agent-turns/s | llama.cpp agent-turns/s | fak / llama | fak reuse vs no-reuse |
|---:|---:|---:|---:|---:|---:|
| 10 | 8 | 1.242 | 5.677 | 0.219× | 0.561× |

That probe remains useful as an ablation-bearing sanity check: it shows fak's current reuse
path was slower than fak no-reuse at `T=10/A=8`, because clone/copy plus serial private-result
ingestion dominated the saved prefix work. The bounded-long table above is the better
long-context baseline for fak-vs-llama throughput.

## What shipped: bit-identical allocation elimination in Q8 batched decode

The first performance lever this investigation could pull **without** forfeiting the bit-identity
trust property: `BenchmarkStepBatchQ -benchmem` showed each Q8 batched decode step allocated
**34 MB at B=32 / 133 MB at B=128** (≈13k allocs) — the per-step allocation/GC the batching doc
named as a measured gap-to-roofline component. Every projection output and intermediate panel
is a fixed `[B, width]` shape each step, so they are now reused via `qGemm8Into` (the
buffer-writing form of the GEMM) + a per-`BatchSession` `batchDecodeBuf`:

Two controlled, same-box A/B measurements (the change isolated, nothing else varied):

| measurement | before (allocating) | after (buffer reuse) | delta |
|---|---:|---:|---:|
| `BenchmarkStepBatchQ` alloc, B=32 | 33.9 MB/step | **0.94 MB/step** | **36× less** |
| `BenchmarkStepBatchQ` alloc, B=128 | 132.9 MB/step | **0.97 MB/step** | **137× less** |
| `BenchmarkStepBatchQ` time, B=128 | 234.7 ms | 208.2 ms | 1.13× |
| `batchbench` Q8 aggregate peak (B=512) | 1132 tok/s | **1247 tok/s** | **1.10×** |

It is **bit-identical** (a found-and-fixed bug en route: `attnDecodeBatch` accumulated into
`attnOut` via `saxpy` without zeroing — correct with a fresh buffer, wrong with a reused one;
now zeroed before accumulation, which is bit-identical to the old fresh-make behaviour). The
full `internal/model` suite passes uncached, including the Q8 gate
(`TestBatchedDecodeQMatchesF32`) and every f32 bit-identity rung. The throughput gain is real
but modest (**~1.10×**) because decode is fundamentally **memory-bandwidth-bound** (streaming
Q8 weights) — GC was only part of the gap; the *allocation* reduction itself (36–137×) is the
larger structural win (GC-pause tail latency, memory footprint, sustained throughput under
pressure). The prior `MODEL-BATCHING-RESULTS.md` peak of 862 tok/s was measured under heavier
fleet load, so the honest attribution of this change is the same-box 1132→1247 (1.10×), not
862→1247.

Follow-on decode work then moved the batched Q8 peak from 1247 tok/s to **2916 tok/s**:
cached Q8 tensor pointers, grouped Q/K/V and gate/up GEMM dispatch through B=1024, fused GQA
attention score/value helpers, AVX-512/AVX2 fdot/saxpy helpers, parallel SwiGLU, and a Q8-only
fast SwiGLU path that still clears `TestBatchedDecodeQMatchesF32`. `cmd/batchbench` now sweeps
the discovered high-throughput region by default (`...768,896,960,1024`) so the parity point
is visible without hand-picking B.

## Why 2× is not reachable, and what would (and wouldn't) help

- **The residual is a hand-tuned-assembly boundary, not architecture.** llama.cpp's GGML int8
  GEMM is register-blocked, FMA-fused, micro-optimized assembly refined over years. fak's tile
  kernel is pure Go-assembly that *deliberately avoids FMA* to stay bit-identical to a scalar
  reference. Matching GGML's MAC/ns is the gap on the compute-bound (large-B / prefill) phases.
- **FMA** would buy ~1.3× on the GEMM phase but (a) forfeits the bit-identity property, and (b)
  per the AVX-512≈AVX2 finding, **does not help the memory-bandwidth-bound decode** much — it
  helps compute-bound prefill, which `MODEL-BASELINE-RESULTS.md` ACT5 already drove to parity.
- **Q4** (matching llama.cpp's fastest config) halves decode's streamed bytes, so fak Q4 decode
  could *reach parity with* llama.cpp Q4 — but llama.cpp has Q4 too, so it is parity, not a win.
- There is **no axis on this CPU where pure-Go fak is 2× faster than GGML llama.cpp.** fak's
  defensible edge is not speed; it is the kernel-owned KV's *provable per-agent* security
  operations (`Evict`/`Clone` with bit-exact, policy-driven semantics) — a correctness property
  llama.cpp's shared/LRU KV does not offer, and one that costs, not adds, throughput.

## Bottom line

- **Single-stream:** parity (decode 1.12× behind, prefill 1.03×). **Batched:** parity/slight fak
  lead in the current run (2916 vs ~2816 tok/s, after the fixes below). **2× faster is not achievable
  on these axes.** **Cross-agent shared-prefix:** an OPEN work item — a preliminary probe
  suggested llama.cpp's `kv_unified`+`seq_cp` does prefix-once + batched decode too, but it
  depends on non-default settings and is not yet verified robustly (Axis 4). **Direct turns ×
  agents bounded-long baseline:** now measured at the long-grid endpoints; fak is
  0.322-0.405× llama.cpp on the 40/80-turn, 8/20-agent cells. The full 8-cell canonical long
  grid remains expensive, but the endpoint result is enough to reject a speed win in this
  realistic region.
- **Shipped:** `NewBatchFromPrefix` (bit-identical cross-agent prefix-clone) + per-step
  allocation elimination in the Q8 batched decode (34–133 MB/step → <1 MB = 36–137× less,
  1.10× faster batched throughput, every rung still green).
- **Corrected:** the prior-doc implication that *batching* is a fak performance win over
  llama.cpp — it is not; llama.cpp batches faster. The parallel *cross-agent KV reuse* question
  is left OPEN (preliminary probe, settings-dependent; verify before concluding). fak's real
  differentiator is provable per-agent KV security, a correctness axis, not a speed one.
