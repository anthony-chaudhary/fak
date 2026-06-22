---
title: "Ultra-Long-Context Sessions: the exact work floor at >100k tokens"
description: "The fused agent kernel's reread-elimination win at the >100k-token regime, proven as an exact contention-free work floor: ~10x vs naive on a single >100k session, ~40x+ on a 5-agent fleet each >100k — token floor cross-validated against sessionbench, FLOP floor O(L^2)-aware."
---

# ULTRA-LONG-CONTEXT-RESULTS — the exact work floor at >100k tokens (2026-06-22)

> **📊 AUTHORITY:** This document's ultra-long-context floor numbers are centrally indexed in
> **[BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md)**, the single source of truth. The
> floor is regenerable arithmetic — `go run ./cmd/longctxbench -ladder` — not a hand-entered number.

> **What this answers.** The session value-stack (`SESSION-VALUE-STACK-RESULTS.md`) and the fleet
> sweeps prove the reuse win *up to ~7k-token contexts* — because the naive arm's re-prefill is
> O(T²) and intractable to run live much further (it is exactly why `sessionbench` already
> *computes* its arm A). But the regime the kernel exists for — the "long agent session" and
> "agent city" of the [scaling-laws thesis](../SCALING-LAWS-OF-AGENTS-2026-06-19.md), where each
> agent's context crosses **100k tokens** — was never measured. This document closes that gap the
> honest way: not by faking a 100k wall-clock, but by proving the **work the fused kernel
> eliminates** as a closed-form, contention-free **floor** derived from the session shape and the
> model geometry. The ratios are arithmetic facts a `go test` re-derives byte-for-byte.

## The regime (what ">100k per agent" means)

`C` agents share a `P`-token prefix (a large working set: a long document, a repo snapshot, a
RAG context, fat tool schemas). Each runs `T` turns; a turn decodes `D` assistant tokens then
ingests `R` tool-result tokens. A single agent's context grows `P → P + (T-1)(D+R) + D`. The
**ultra-long** regime is any cell whose peak per-agent context is **≥ 100,000 tokens** — reached
either by a large shared prefix or by a deep session, or both.

## The three arms (identical to the session value-stack — the delta is PURE reread-elimination)

| arm | what it models | prefix prefill | per-turn prefill | decode |
|---|---|---|---|---|
| **A — naive-stateless** | the common local pattern: a stateless API / `llama-cli -p <full prompt>` each turn | re-prefills the **whole context** every turn (O(T²); the prefill's own attention is O(L²)) | — | serial |
| **B — per-agent-KV** | a warm single-tenant cache: persistent KV per agent, no cross-agent sharing | once **per agent** (C×P) | incremental (R only) | serial |
| **C — fak fused** | the kernel: prefix prefilled **once** + cloned into C agents | **once** total | incremental (R only) | batched* |

`*` Decode batching is a **bandwidth** win (one weight stream serves C lanes), not a FLOP win —
proven in [MODEL-BATCHING-RESULTS.md](MODEL-BATCHING-RESULTS.md). It is **deliberately excluded
from this work floor** so the floor isolates the *reread-elimination* win alone and never
double-counts the bandwidth win. The live wall-clock (`sessionbench`) adds it back on top.

**A/C** is vs the **naive** pattern (a worst-case **REFERENCE**, *not* a serving baseline anyone
ships). **B/C** is vs a **warm per-agent KV cache** (the honest serving baseline). They are
reported side by side and never conflated — the standing BENCHMARK-AUTHORITY law.

## Two floors, both exact and contention-free

- **Token floor** — the exact prefill-token count each arm processes. Byte-identical to
  `cmd/sessionbench`'s `prefillTokens` (`A = C·Σ(P+t(D+R))`; `B = C·(P+(T-1)R)`; `C = P+C(T-1)R`),
  so it **cross-validates against the live bench's own floor**. Linear in tokens → the
  **conservative** lower bound (it ignores that re-prefilling 100k tokens costs O(L²), not O(L)).
- **FLOP floor** — the **O(L²)-aware** work floor. Prefill attention over an `L`-token context is
  `≈ L²/2` query-key pairs; the naive arm pays that quadratic *every turn over the whole growing
  context*, while the fused kernel pays the prefix once and only incrementally ingests new spans.
  This is where the ultra-long-context win actually lives. Exact arithmetic from the model shape.

## Results — the canonical regime ladder (Qwen2.5-7B geometry)

`go run ./cmd/longctxbench -ladder` (artifact:
`experiments/session/ultra-long-context-floor-20260622.json`):

| P | T | C | max ctx | **tok A/C** | tok B/C | flop A/C | flop B/C | flop A/B | regime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2,048 | 50 | 5 | 6,784 | **62.0×** | 1.5× | 41.1× | 1.3× | 31.8× | anchor (not ultra) |
| 100,000 | 10 | 1 | 106,500 | **9.9×** | 1.0× | 9.5× | 1.0× | 9.5× | **ULTRA single** |
| 100,000 | 10 | 5 | 106,500 | **42.1×** | 4.3× | 34.4× | 3.6× | 9.5× | **ULTRA multi** |
| 100,000 | 50 | 5 | 134,500 | **131.6×** | 2.8× | 78.9× | 2.0× | 40.1× | ULTRA multi (deep) |
| 100,000 | 40 | 40 | 106,500 | **147.4×** | 14.9× | 79.9× | 8.4× | 9.5× | ULTRA agent-city |

### The two headlines the regime names

- **Single session > 100k: ~10× vs naive** (9.9× token / 9.5× FLOP). With one agent there is no
  cross-agent prefix to share, so **B/C is exactly 1.0** — the entire win is the *turn-tax*: not
  re-prefilling a 100k context on every one of the 10 turns. This is an A/C (vs-naive) number.
- **5-agent fleet, each > 100k: ~40×+ vs naive** (42.1× token / 34.4× FLOP), and **~4.3× vs a warm
  per-agent KV cache** (B/C). The vs-naive figure is turn-tax × cross-agent prefix-share; the
  vs-tuned figure is the genuine marginal cross-agent reuse.

## The honest reconciliation (read before quoting B/C)

The repo's standing bound is **"B/C ≈ 2–4× vs a tuned cache"**, measured at a *small* prefix
(P≈2k). This floor shows **why**, exactly, and that the bound is regime-specific, not a ceiling:

```
B/C = [C·prefixWork + sharedWork] / [prefixWork + sharedWork]
```

rises monotonically from **1** (P→0, no prefix to share) toward the **agent count C** (P→∞, prefix
dominates). At P=2k it is small (1.3–1.5× *reread-only* on this floor); at a 100k *shared* prefix
with 40 agents it climbs to **8.4× (FLOP) / 14.9× (token)**. Same formula, different prefix
fraction — proven monotone by `TestLongContextBOverCMonotoneInPrefix`.

Two further honesty notes so nothing reads out of scope:

- **This floor's B/C is *lower* than the live `sessionbench` B/C** at the same shape (anchor: 1.5×
  token-floor here vs **4.1×** live in the committed headline). That is **by design**: this floor
  counts *reread-elimination only*; the live bench's larger number adds the **decode-batching
  bandwidth** win this floor deliberately omits. The floor is the conservative reread-only term.
- **The anchor row reproduces the committed number.** Its token A/C of **62.0×** is byte-identical
  to the `62.0× token floor` already recorded for the 50×5 P=2048 session (SCALING-LAWS §3, line
  145) — independent evidence the floor arithmetic is correct.

## Methodology — why a floor, not a wall-clock

Running a 100k-token naive arm live is intractable (the same O(T²) re-prefill that makes the win
large). But you do not need to: the **work each arm performs is a closed-form function of
(P, T, C, D, R) and the model geometry** — pure arithmetic, no weights, no decode, no box. The
attention term is counted exactly: appending `n` tokens to a context of `prior` forms
`n·prior + n²/2` query-key pairs, each one MAC per head-dim across all heads, counted twice (scores
+ the softmax-weighted V sum), per layer. From-scratch prefill of `L` is the `prior=0` case
(`L²/2` pairs). This is contention-free and deterministic; the **live wall-clock validation at
>100k is a separate, bench-node-gated measurement** (it needs a model resident — see the live-anchor
issue).

This floor makes **no model-quality or resolve-rate claim**: it depends only on the session
*structure*, not on what the model decodes. A real session with variable per-turn decode/result
sizes is bounded between the floors for its min/max turn sizes.

## Witnesses (the DOS discipline — every claim closed by a mechanical witness the author did not write)

| claim | witness (re-runnable) | what it rules out |
|---|---|---|
| the token floor equals `sessionbench`'s exact floor | `go test ./internal/turnbench -run TestLongContextTokenFloorMatchesSessionbench` | "the floor uses a different (favorable) count" |
| a single agent claims **no** cross-agent win | `go test ./internal/turnbench -run TestLongContextAntiInflation` (C=1 ⇒ B/C ≡ 1.0) | "the multi-agent win is inflated onto solo sessions" |
| single ~10× / multi ~40×+ are the real ladder values | `go test ./internal/turnbench -run TestLongContextHeadlineRegimes` | "the headline numbers are hand-picked" |
| B/C is bounded by C and monotone in prefix fraction | `go test ./internal/turnbench -run TestLongContextBOverCMonotoneInPrefix` | "B/C can be quoted at any size out of regime" |
| the attention quadratic is real in the floor | `go test ./internal/turnbench -run TestPrefillWorkQuadraticDominates` | "the long-context win is linear hand-waving" |
| decode work is identical across arms (no double-count) | `go test ./internal/turnbench -run TestLongContextDecodeFLOPsInvariant` | "the floor smuggles in the decode-batching win" |
| the artifact is deterministic and regenerable | `go test ./internal/turnbench -run TestRunLongContextLadderDeterministicAndPicksRegimes` | "the number drifts run to run" |

The literal `dos verify` truth-syscall over the ship commit runs on the DOS host; here the
discipline is the witness set above plus the committed JSON artifact.

## Reproduce

```sh
# the canonical regime ladder (single 100k → agent-city), writes the committed artifact
go run ./cmd/longctxbench -ladder -out experiments/session/ultra-long-context-floor-20260622.json

# a custom sweep on any of the three model geometries
go run ./cmd/longctxbench -model qwen25-7b -prefix 100000 -turns 10 -agents 1,5,40 -decode 200 -result 500

# the witnesses
go test ./internal/turnbench -run TestLongContext -count=1
```

## The honest scoping (one paragraph)

At the >100k regime the fused kernel eliminates **~10× the work of the naive re-prefill loop on a
single session and ~40×+ on a 5-agent fleet** (A/C, vs a worst-case reference) — a floor that grows
with both session depth and agent count and is dominated by the O(L²) prefill quadratic the naive
arm re-pays every turn. Against an *already-warm* per-agent KV cache the genuine marginal win is
**~1× solo and ~4× at five agents**, rising toward the agent count as the shared prefix dominates
(B/C, vs the serving baseline). These are exact, contention-free *work* numbers — not a wall-clock
and not a model-quality claim; the live wall-clock at this regime is the next, separately-gated step.
