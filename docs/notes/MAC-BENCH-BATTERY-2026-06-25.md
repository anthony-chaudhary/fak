---
title: "fak Mac bench battery — M3 Pro own-forward ladder + radix regression + agent-live (2026-06-25)"
description: "A 2026-06-25 sleep-resilient benchmark battery on node-macos-a (Apple M3 Pro): fak's pure-Go CPU own-forward decode/prefill up the model ladder, a model-ladder residency sweep to 27B, a clean RadixAttention regression pass, and the agent-live coverage gap. Every number labeled forward=pure-go-cpu with full lineage."
date: 2026-06-25
---

# Mac bench battery — own-forward ladder + radix regression + agent-live (2026-06-25)

Run on `node-macos-a` (Apple M3 Pro / Mac15,7, 12 core = 6P+6E, 36 GB unified, macOS 26.5
Darwin 25.5), fresh shallow public clone at HEAD `e27511f`, Go via `GOTOOLCHAIN=auto`. Driven
over Tailscale SSH from a Windows box, detached + sleep-resilient (the MacBook is a remote
power-nap-prone node; the battery ran under `caffeinate -dimsu` and survived SSH drops via
`nohup`). **Every forward number is fak's OWN pure-Go CPU forward** (`engine: fak-in-kernel
Q8_0`, backend `cpu-ref`, 12 workers) — no Metal, no llama.cpp, no gateway. The box had just
woken (load avg ~2.2), so single-stream numbers carry slightly more variance than a cold-idle arm.

## 1. Own-forward decode/prefill ladder (the two honest throughput rungs)

| model | source | decode tok/s | prefill@256 tok/s | load | note |
|---|---|---:|---:|---:|---|
| Qwen2.5-1.5B Q8 | GGUF (q8_0) | **36.31** | 234.1 | 3.1 s | regression witness vs 38.07 canonical — holds within variance |
| Qwen2.5-7B Q8 | HF safetensors (quant-at-load) | **10.43** | 59.2 | 37.3 s | faster than the 8.7 fence — but HF-sourced (not the fence's GGUF) |
| Qwen2.5-7B Q8 | **GGUF (3-shard, same as fence)** | **10.92** | 59.5 | 12.7 s | apples-to-apples vs the 8.7 fence — and the GGUF loads 3× faster than HF |

Honesty notes:
- The 1.5B (36.31 tok/s) reproduces the 2026-06-23 canonical (38.07) within normal just-woke
  variance — a clean drift check, the baseline holds.
- **The 7B result is the headline new datum**: measured from TWO independent sources that agree
  within 5% — HF-snapshot quant-at-load (10.43 tok/s) and the **3-shard Q8 GGUF, the same source
  the 8.7 fence used** (10.92 tok/s). Both are **consistently faster than the recorded 8.7
  tok/s baseline**, so the 7B own-forward CPU decode on this M3 Pro is honestly ~10.4–10.9 tok/s,
  not 8.7. Cite the GGUF number (10.92) — it's apples-to-apples with the fence and the GGUF loads
  3× faster than HF (12.7 s vs 37.3 s). Decode tok/s held stable across both sources and despite
  box contention (the pure-Go forward saturates its 12 workers regardless), which makes the
  upward revision robust, not a one-off lucky arm.

## 2. Model-ladder residency witness (load-only — NOT a throughput claim)

| model | source | peak RSS | fits in 36 GB? | claim |
|---|---|---:|---|---|
| Qwen2.5-14B Q4_K_M | 3-shard GGUF (pass shard 00001) | **22.5 GB** | yes (~13 GB headroom) | sharded-GGUF auto-discovery + lean load works |
| Qwen3.6-27B Q4_K_M | GGUF | **26.3 GB** | yes (~10 GB headroom) | the 27B architecture loads + fits in fak's own kernel |

These are **load-only residency witnesses**, run because a full decode sweep at the 27B's ~0.9
tok/s would consume the window. The 27B residency does NOT re-establish the committed 0.9 tok/s
throughput witness (a separate run) — it proves only that the weights load and fit. Dropped by
design (per adversarial review against shipped HEAD): Qwen3-30B-A3B (qwen3moe has no wired
expert-splitter in the pure-Go loader — only `glm_moe_dsa`), 32B-Q8-lean and 72B-Q2 (would
exceed 36 GB unified).

## 3. RadixAttention regression witness — clean pass

Re-ran `radixbench` with the real Qwen2.5-1.5B-Q8 (live arm, 2 reps, uncontended). The
model-independent reuse algorithm reproduces the committed 2026-06-23 numbers to the decimal:

| workload | hit rate (this run) | committed | live ratio (this run) | committed |
|---|---:|---:|---:|---:|
| few-shot | 88.24% | 88.2% | 7.05× | 7.01× |
| multi-turn-chat | 79.55% | 79.5% | 4.11× | 4.15× |
| tree-of-thought | 77.25% | 77.2% | 3.61× | 3.69× |
| agents | 86.67% | 86.7% | 5.85× | 5.84× |

The scheduling theorem reproduces (agents FCFS 62.1% → cache-aware 86.7% = 100% of optimal) and
the policy-eviction witness passed (freed 8 tokens on a verdict, benign sibling kept). Hit rates
are hardware/model-independent (the same radix-tree + longest-prefix + LRU-leaf algorithm SGLang
headlines); the live wall-clock ratios are the new datum and sit in the predicted 3.6–7.0× band.

## 4. agent-live — the coverage gap, filled (and an honest negative)

The bench planner's top `node-macos-a` gap (`agent-live`, 0 prior runs) now has a real on-device
A/B: `fak serve --gguf` (1.5B-Q8 in-kernel) + `fak agent` over `/v1`, both arms on the canned
airline-support task, 100% on-device CPU forward. The serve came up `engine=inkernel
model=qwen2.5-1.5b vdso=true auth=true`.

**What's real:** the adjudication kernel fires in the loop — `get_user_details` and
`search_direct_flight` → `ALLOW` (read-only), `book_flight` → **`DENY (DEFAULT_DENY)`** (the
capability floor refused the booking the policy doesn't allow). 2 turns, 2 tool calls, 0 errors,
1255 prompt + 290 completion tokens, no injection, no destructive exec, transcript_sha recorded.

**The honest negative:** `task_completed=false` for BOTH arms, `turns_saved=0`, `tokens_saved=0`.
The 1.5B **hallucinated success** — its final answer narrates "booking is successful, flight
UA123" even though the `book` tool was DENIED and never returned a confirmation. The success bit
(book must return "confirmation" w/o "error") correctly reads false. This reproduces the known
dogfood pattern: at 1.5B the in-kernel CPU path proves the **wire + adjudication**, not model
usefulness — the model confabulates a completion past a correct policy DENY. fak and baseline arms
are identical here (a single 2-turn run has no reuse delta to measure). Honest, and more
informative than a green check: **the harness + kernel work; the 1.5B is too weak to actually
finish, and the kernel caught it trying to book anyway.**

## 5. session-benchmark

Re-run with the corrected `-hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -lean` (first pass
defaulted to a nonexistent smollm2 cache — my bug). The full agent-count sweep did **not** finish
within its 1200s wall — a single live cell (50 turns × 5 agents × best-of-2 decode on the 1.5B CPU
forward) is itself multi-minute, so it wall-killed mid-cell. That's an honest finding:
`sessionbench` with default params (50 turns, prefix ladder to 6848) is too heavy for a CPU
own-forward to complete in a sane wall — the right shape for this box would be fewer turns /
smaller prefixes.

The salvaged datum is real and useful — **own-forward prefill scaling vs context length** (the
load/warm phase completed before the cell sweep stalled):

| prefill context (tokens) | tok/s |
|---:|---:|
| 256 | 238.8 |
| 1904 | 137.1 |
| 2048 | 132.7 |
| 3552 | 95.5 |
| 5200 | 73.5 |
| 6848 | 60.0 |

Prefill throughput decays from ~239 tok/s at 256 tokens to ~60 tok/s at 6848 — the expected
super-linear attention cost with context length, measured on fak's own pure-Go CPU forward. This
is directly relevant to agentic context budgets: a long-context turn on this box prefills at ~60
tok/s, so a 6.8k-token agent prompt is ~1.9 s of prefill before the first decoded token.

### 5b. Bounded session-value stack (wave 2 — the sweep that completed)

Re-run bounded (`-turns 8 -agents 3 -prefix 512 -reps 1`) so it finishes. First live measurement
of the multi-agent session-value stack on the M3 Pro (1.5B-Q8, 8 turns × 3 agents, 512-tok shared
prefix):

| arm | total ms | prefill tokens | net value-add |
|---|---:|---:|---|
| A — cold, no cache (re-prefill every turn) | 126,777 | 19,008 | worst-case reference |
| B — warm per-agent KV (honest serving baseline) | 43,721 | 2,880 | — |
| C — **fak fused (cross-agent prefix share)** | 38,685 | 1,856 | **3.28× vs A, 1.13× vs B** |

fak's cross-agent KV clone costs **3.76 ms** (vs re-prefilling the 512-tok shared prefix), and it
prefills **10.24× fewer tokens than cold** / 1.55× fewer than the warm per-agent cache. The honest
serving number is **1.13× vs a warm cache** (fak's marginal win = cross-agent prefix sharing on top
of an already-warm cache — fak's actual story); the 3.28× cold number is a worst-case reference, as
the artifact's own headline states. Live-validation cross-checks the computed arm-A cost against a
fully-live arm to within 0.4% (`raw_computed_over_live = 0.996`).

### 5c. Agent-live on 7B (wave 2) — the stronger model is the honest one, and exposes a benchmark limit

Same A/B on the 7B-Q8 GGUF. The 7B does NOT confabulate like the 1.5B did — its final answer
**accurately reports the refusal**: "All proposed tool calls were refused by the fak kernel:
convert_currency: DENY; book_flight: DENY (DEFAULT_DENY/TERMINAL)." This surfaces a real limit of
the agent-live benchmark as configured: **the default capability floor DENIES both `book_flight`
and `convert_currency`, which the task requires — so `task_completed` can NEVER be true under this
policy**, regardless of model. The benchmark correctly measures *adjudication* (reads ALLOW, writes
DENY) but cannot measure task *success* without a policy that permits booking. To get a real
fak-vs-baseline completion delta, the agent-live lane needs a policy whose allow-list includes the
task's required tools (or a task within the read-only allow-list). A genuine, actionable finding
for whoever next runs this lane.

## Reproduce

The battery script (`mac_battery.sh`) + driver supervisor live in the session scratchpad. Core
per-model commands (run from the clone root, `GOTOOLCHAIN=auto`, `PATH` incl. `/usr/local/go/bin`):

```bash
G=~/.cache/fak-models/gguf
# 1.5B-Q8 decode/prefill
go run ./cmd/modelbench -gguf $G/qwen2.5-1.5b-instruct-q8_0.gguf -lean \
   -decode-reps 10 -decode-steps 32 -prefill-reps 5 -prefill-sizes 16,64,256
# 7B (HF snapshot, quant-at-load)
go run ./cmd/modelbench -hf ~/.cache/fak-models/qwen2.5-7b-instruct -lean \
   -decode-reps 3 -decode-steps 24 -prefill-reps 3 -prefill-sizes 16,64,256
# 14B / 27B residency (load-only)
go run ./cmd/modelbench -gguf $G/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf -lean -load-only
go run ./cmd/modelbench -gguf $G/Qwen3.6-27B.q4_k_m.gguf -lean -load-only
# radix regression
go run ./cmd/radixbench -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -lean -quant -reps 2
```
