---
title: "fak Mac Bench Refresh: M3 Pro CPU Q8 Parity + RadixAttention"
description: "A 2026-06-23 M3 Pro refresh re-measuring fak's single-stream CPU Q8 parity vs llama.cpp on Qwen2.5-1.5B and the first uncontended live RadixAttention 1.5B run."
---

# Mac bench refresh — CPU Q8 parity + uncontended radix on the M3 Pro (2026-06-23)

Run on `node-macos-a` (Apple M3 Pro, 12 core = 6P+6E, 36 GB unified, macOS 26.5),
fresh clone of the public repo at HEAD `374776a` (fak `0.31.0`), Go 1.26.0 self-fetched.
Box was idle for every timed arm. Two deliverables, plus a reconciliation that fixes a
real traceability bug in the authority doc.

## 1. Canonical CPU Q8 parity, refreshed — and reconciled

Re-measured the canonical single-stream CPU Q8 baseline (`fak modelbench -hf -lean` vs
`llama-bench -ngl 0`, Qwen2.5-1.5B-Instruct Q8_0, same `541bf3762`/8200 llama build as the
committed artifact). The same-12-thread arm reproduces the 2026-06-21 baseline within noise:

| arm (equal 12-thread budget) | fak | llama.cpp | fak/llama |
|---|---:|---:|---:|
| decode | 38.07 tok/s | 52.44 tok/s | **0.73×** |
| prefill@256 | 240.45 tok/s | 412.51 tok/s | **0.58×** |

(2026-06-21 committed: 0.72× / 0.58× — held across the 0.30.0 → 0.31.0 bump. A clean
regression-prevention witness.)

### The reconciliation (why a headline number moved)

`BENCHMARK-AUTHORITY.md` inline read **decode 0.58× (41.9 vs 71.9) · prefill@256 0.45×
(247.2 vs 547)**, citing commit `3448d7b` — which **does not exist in the public repo**
(a stale private-tracker SHA). Its own cited artifact said 0.72× / 0.58×. The doc and its
"single source of truth" disagreed.

A thread-count sweep explains the whole gap. llama.cpp single-stream **decode is
memory-bandwidth-bound and runs faster on P-cores only**, while fak's pure-Go decode is
compute-parallel and wants all 12:

| metric | fak @6 | fak @12 | llama @6 | llama @12 |
|---|---:|---:|---:|---:|
| decode tok/s | 21.81 | **38.07** | **68.70** (±0.19) | 52.44 (±3.92) |
| prefill@256 tok/s | 35.15 | **240.45** | 388.02 | **412.51** |

So the stale `71.9` decode was a llama **−t 6** run (68.7 here, and notably the most stable
config), and the stale `547` prefill was an **older llama build** (the current 8200 build
prefills 388–412, not 547). Both were real once; neither matches the current artifact.

The refresh resolves it honestly by reporting both views and citing a real commit:

- **Equal 12-thread budget** (apples-to-apples thread count): decode **0.73×**, prefill **0.58×**.
- **Each engine at its own best thread config** (the conservative fence): decode **0.55×**
  (fak 38.07 @12 vs llama 68.70 @6), prefill **0.58×** (both peak at 12).

The conservative fence (decode **0.55×**) is the number to quote when one figure is wanted —
it is the least fak-flattering, so no reader is misled into thinking fak is closer than it is.
fak still trails llama.cpp single-stream by design; the win is the cross-agent reuse layer on
top, measured elsewhere.

Artifact: [`experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json`](../../experiments/model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json)
(schema bumped to `fak-cpu-parity/2`, both thread arms embedded).

## 2. RadixAttention live — first clean uncontended 1.5B run on the M3 Pro

The committed `RADIXATTENTION-RESULTS.md` real-model live column was SmolLM2-135M on a
heavily contended box (its absolute ms were ~140× inflated, kept only for the within-run
ratio). This is the first all-four-workloads, real-1.5B-Q8, **uncontended** live arm:

| workload | hit rate | live ratio (uncontended) |
|---|---:|---:|
| few-shot | 88.2% | **7.01×** |
| multi-turn-chat | 79.5% | **4.15×** |
| tree-of-thought | 77.2% | **3.69×** |
| agents | 86.7% | **5.84×** |

The hit rates, token-speedups, declare-one deltas, and the scheduling theorem (agents FCFS
62.1% → cache-aware 86.7% = 100% of optimal) are model-independent and **reproduce the
committed numbers exactly** — only the live wall-clocks are the new datum. The policy-eviction
witness also passed (freed 8 tokens on a verdict, benign sibling kept). The live ratios sit in
the 3.7–7.0× band the model-ladder thesis predicts.

Artifact: [`experiments/radixattention/radixbench-qwen2.5-1.5b-q8-m3pro-uncontended-20260623.json`](../../experiments/radixattention/radixbench-qwen2.5-1.5b-q8-m3pro-uncontended-20260623.json).

## Reproduce

```bash
GOTOOLCHAIN=auto go build -o /tmp/modelbench ./cmd/modelbench
GOTOOLCHAIN=auto go build -o /tmp/radixbench ./cmd/radixbench
HF=~/.cache/fak-models/qwen2.5-1.5b-instruct
GGUF=~/.cache/fak-models/gguf/qwen2.5-1.5b-instruct-q8_0.gguf

# parity, both thread arms
/tmp/modelbench -hf $HF -lean -decode-reps 10 -decode-steps 32 -prefill-reps 5 -prefill-sizes 256
/tmp/modelbench -hf $HF -lean -budget 6 -decode-reps 10 -decode-steps 32 -prefill-reps 5 -prefill-sizes 256
llama-bench -m $GGUF -ngl 0 -t 12 -p 256 -n 32 -r 10 -o json
llama-bench -m $GGUF -ngl 0 -t 6  -p 256 -n 32 -r 10 -o json

# radix live, real model
/tmp/radixbench -hf $HF -lean -quant -reps 2
```
