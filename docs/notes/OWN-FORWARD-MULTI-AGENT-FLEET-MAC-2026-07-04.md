---
title: "Own-forward multi-agent fleet on a Mac — measured, not estimated (2026-07-04)"
description: "Real, live sessionbench runs of the 3-arm multi-agent session-value stack on fak's OWN pure-Go CPU forward (no Metal, no llama.cpp) on the M3 Pro bench node — filling the gap BENCHMARK-AUTHORITY.md and FLEET-5X200-7B-10MIN-RESULTS.md left as an arithmetic estimate (~22-51 min, never run). Answers: does the multi-agent value-add survive when per-token decode is slow? Yes, but through a different mechanism than the Metal case — and the exact commit this ran on is not cleanly citable, disclosed plainly."
date: 2026-07-04
---

# Own-forward multi-agent fleet on a Mac — measured, not estimated

> **Honesty header (`docs/proofs/00-METHOD.md`).** Run over SSH against `node-macos-a`
> (Apple M3 Pro, 12 core, 36 GiB unified, macOS 26.5.2). **Read this before citing a tok/s
> number from this page**: the binary was built from the box's existing checkout at
> `~/work/fleet/fak` (a private mirror, remote `anthony-chaudhary/fleet`), pinned at commit
> `381c330` — **not** the public `fak` `origin/main` HEAD this doc-set otherwise cites — and
> that checkout had a live, **uncommitted** 566-insertion/396-deletion diff in progress across
> `internal/model/{parallel,quant,quant_arm64,quant_forward,quant_quantize_noasm}.go` (someone
> else's concurrent matmul/quant-kernel refactor; not touched, not reverted, not committed by
> this run). Those files are exactly the hot path that sets decode/prefill tok/s. So: the
> **relative comparison between arms A/B/C is self-consistent** (same binary ran all three), but
> the **absolute tok/s figures below are directional, not a citable fak-commit number** — a
> clean re-run against a `git archive origin/main` snapshot (the method
> [`ULTRACODE-WORKFLOW-PROOF-HARNESS-2026-07-01.md`](ULTRACODE-WORKFLOW-PROOF-HARNESS-2026-07-01.md)
> used) is the next checkable step, not done here. Every arm is still a **real, live, end-to-end
> kernel call** — nothing here is modeled or synthetic.

## The question this answers

[`FLEET-5X200-7B-10MIN-RESULTS.md`](../benchmarks/FLEET-5X200-7B-10MIN-RESULTS.md) proves "5
agents × 200 turns of 7B in 8.2 min" only for **llama.cpp's Metal forward**, and states plainly
that fak's **own** pure-Go CPU forward would be "~22-51 min, well over the bar" — but that number
was **computed from single-stream rate cards, never actually run**. `BENCHMARK-AUTHORITY.md` row
40 repeats the same caveat with the same ⚠️. So the open question sitting in two authoritative
docs was: *if you actually run the fak-fused multi-agent pattern on fak's own (slow) CPU forward,
does it still deliver a multi-agent value-add, or does the slow per-token rate wash it out?*

This page runs it — live, real weights, real kernel calls — at a **reduced but honest scale**
(fewer turns than the full 5×200 shape; see "Scope" below) and answers: **it still works** — the
fleet completes correctly and the fak-fused arm still beats a warm per-agent cache — but the
*mechanism* is not what the Metal number suggested.

## Scope (what changed from the 5×200 shape, and why)

The literal 5-agent × 200-turn shape was attempted first, on the 1.5B model, and killed after 59
minutes with no completion in sight and no partial artifact (`sessionbench` has no
`--checkpoint`/`--resume`, unlike `cmd/modelbench`/`fanrun` — see "Next steps"). `node-macos-a` is
a shared box (5 logged-in users, load average ~4.3 at the time), so an open-ended multi-hour job
is not a considerate use of it. The runs below use the same prefix/decode/result shape (P=2048,
D=20, R=12) and the full **5-agent** width, at a **reduced turn count** (T=50 for 1.5B, T=20 for
7B) chosen to finish in a bounded window. This is a smaller shape than the flagged 5×200 gap, not
the same one filled exactly — stated plainly rather than left for a reader to assume otherwise.

## Results — real, live, `-reps 1` (single rep; see caveat below)

Methodology is `cmd/sessionbench`'s 3-arm stack (full description in the command's doc comment):
**A** = naive stateless re-prefill every turn (computed from measured prefill-cost samples,
anchored to a live small-scale validation run); **B** = per-agent persistent KV, serial decode,
prefix prefilled once per agent (the honest single-tenant serving baseline); **C** = fak fused —
shared prefix prefilled **once** total and cloned into 5 sessions, decode **batched** across all 5
agents each step (`BatchSession.StepBatch`), incremental tool-result ingestion per agent.

| | 1.5B Q8 (T=50, C=5) | 7B Q8 (T=20, C=5) |
|---|---:|---:|
| **A — naive re-prefill** (computed+anchored) | 7198.9 s (~120 min) | 7212.9 s (~120 min) |
| **B — per-agent warm KV** (live) | 920.7 s (~15.3 min) | 1002.1 s (~16.7 min) |
| **C — fak fused** (live) | **865.4 s (~14.4 min)** | **798.2 s (~13.3 min)** |
| **NetVsTuned (B/C) — HEADLINE** | **1.06×** | **1.26×** |
| NetVsNaive (A/C) — worst-case reference | 8.32× | 9.04× |
| Turn-tax (A/B) | 7.82× | 7.20× |
| B decode time | 795.2 s | 689.2 s |
| C decode time | 799.8 s | 694.6 s |
| B prefill time | 125.5 s | 313.0 s |
| C prefill time | 65.5 s | 103.6 s |
| Exact prefill-token ratio B/C (contention-free floor) | 2.64× | 3.57× |
| Live-validate: computed-vs-live arm-A prefill | 1.09-1.12× | 1.05× |

Raw artifacts: `experiments/session/own-forward-fleet-mac-1.5b-5x50-20260704.{json,log}` and
`experiments/session/own-forward-fleet-mac-7b-5x20-20260704.{json,log}`.

## The finding worth reporting honestly: the win is prefill sharing, not decode batching

Look at the **decode** row: B and C are within 0.6-0.8% of each other **at both model sizes** — C
is not faster, and at 7B it is (very slightly) slower. On the llama.cpp-Metal path,
`FLEET-5X200-7B-10MIN-RESULTS.md` measured a real 2.71× aggregate decode speedup from batching 5
agents (17.4 → 47.2 t/s). Here, on fak's own pure-Go CPU forward, batching 5 agents into one
`StepBatch` call delivers **no measurable decode throughput gain** at `-reps 1`.

The entire 1.06×-1.26× NetVsTuned win instead comes from **prefill**: B pays for the shared
2,048-token prefix **five times** (once per agent, serially); C pays for it **once**, then clones.
The wall-clock prefill ratio (1.92× at 1.5B, 3.02× at 7B) tracks the exact token-count ratio
(2.64×, 3.57×) reasonably closely — the gap is the fixed per-call overhead that a single big
prefill amortizes better than a small one.

So the honest mechanism statement is: **on fak's own CPU forward, the multi-agent value-add is
100% prefix-sharing, 0% decode-batching** — the inverse emphasis from the GPU/Metal case, where
decode batching was the dominant, measured lever. This is a real, falsifiable, and slightly
surprising result — worth a follow-up to understand *why* `StepBatch` doesn't amortize weight-read
cost across agents on the Go/NEON CPU path the way it does on Metal (whether it's a scheduling
issue, a lack of SIMD lane-sharing across the batch dimension, or single-core saturation already
maxing out per-agent decode before the batch has anything to share). Filed as a checkable next
step, not claimed as understood here.

## Reading this against the separation law

[`ultracode-multi-agent-dogfood.md`](../explainers/ultracode-multi-agent-dogfood.md) draws a hard
line between fak's **inference** axis (tokens/sec, cache reuse — a serving-economics number) and
**ultracode's** concurrency-factor axis (independent reviewed deliverables per orchestration
window — an agent-orchestration number). This page is squarely the **inference** axis: it answers
"if an ultracode-style fleet of coding agents pointed its tool-calling model at `fak serve --gguf`
on a Mac with no Metal, no llama.cpp — what serving economics would they get?" It does **not**
measure or claim anything about orchestration concurrency factor, and the two must not be blended
(same rule that doc states).

The honest answer to that serving-economics question: **slow, but it works** — the fleet
completes correctly, a real (if model-size-dependent) 1.06-1.26× win over a well-tuned per-agent
cache survives the CPU-slow per-token rate, and the ≥8× win over the no-cache pattern is
overwhelming. What it is *not*, on this hardware/build, is fast in absolute terms — 13-15 minutes
for a 20-50 turn / 5-agent session on fak's own CPU forward is real latency an agent fleet would
feel, and the reduced-scope run here does not resolve whether the same ratios hold at the full
200-turn shape (plausible, given the mechanism is prefill-driven and prefill work per turn is flat
regardless of total turn count, but not measured).

## Next steps this opens

- **Re-run on a clean `git archive origin/main` snapshot** (not the dirty private-mirror checkout
  used here) so the absolute tok/s numbers are citable against a real fak commit — the rigor
  [`ULTRACODE-WORKFLOW-PROOF-HARNESS-2026-07-01.md`](ULTRACODE-WORKFLOW-PROOF-HARNESS-2026-07-01.md)
  already established as the pattern for this exact node.
- **Investigate the CPU decode-batching null result** — confirm with `-reps` > 1 (this page used
  best-of-1) whether B≈C decode time is real or noise, and if real, whether
  `internal/modelengine.NativeScheduler` / `BatchSession.StepBatch` actually amortizes anything on
  the pure-Go NEON path, ties into epic [#1911](https://github.com/anthony-chaudhary/fak/issues/1911)
  (agentic-first scheduling, orthogonal to caching) child C
  ([#1914](https://github.com/anthony-chaudhary/fak/issues/1914), cross-agent prefill fusion).
- **`sessionbench` has no checkpoint/resume.** The killed 59-minute 5×200 run left zero artifact.
  `internal/benchckpt` (shipped for `cmd/modelbench`/`fanrun`, tracked under
  [#2382](https://github.com/anthony-chaudhary/fak/issues/2382)) is the existing pattern; wiring
  it into `sessionbench` would let a full 200-turn own-forward run survive a node reboot/SSH drop
  and resume instead of restarting cold — the concrete blocker to filling the exact 5×200 gap.
- **Update `BENCHMARK-AUTHORITY.md` row 40** with a pointer to this page (done in this commit) so
  the ⚠️ estimate now links to a real, if reduced-scope, measurement instead of standing alone.

## Reproduce

```bash
# On node-macos-a (or any Apple-Silicon Mac with the HF snapshots cached):
GOTOOLCHAIN=auto go build -o /tmp/sessionbench ./cmd/sessionbench

# 1.5B, 5 agents x 50 turns
/tmp/sessionbench -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -lean \
  -prefix 2048 -turns 50 -agents 5 -decode 20 -result 12 -reps 1 \
  -out session-1.5b-5x50.json

# 7B, 5 agents x 20 turns
/tmp/sessionbench -hf ~/.cache/fak-models/qwen2.5-7b-instruct -lean \
  -prefix 2048 -turns 20 -agents 5 -decode 20 -result 12 -reps 1 \
  -out session-7b-5x20.json
```

## Witnesses

| claim | witness |
|---|---|
| 1.5B live 3-arm run (A computed+anchored, B/C live) | `experiments/session/own-forward-fleet-mac-1.5b-5x50-20260704.json` + `.log` |
| 7B live 3-arm run (A computed+anchored, B/C live) | `experiments/session/own-forward-fleet-mac-7b-5x20-20260704.json` + `.log` |
| decode B≈C at both sizes (the CPU batching-null finding) | `decode_ms` fields in both JSONs above (795150 vs 799816 ms; 689166 vs 694564 ms) |
| the dirty-checkout caveat is real, not hedging | `git diff --stat` on `~/work/fleet/fak` at run time: `parallel.go` 586 lines changed, `quant_arm64.go` 352 changed, `quant_forward.go`/`quant_quantize_noasm.go` smaller — captured in this page's honesty header |
| the original estimate this page checks | [`FLEET-5X200-7B-10MIN-RESULTS.md`](../benchmarks/FLEET-5X200-7B-10MIN-RESULTS.md), `BENCHMARK-AUTHORITY.md` row 40 |
