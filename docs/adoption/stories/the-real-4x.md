---
title: "The 4x that's real: the tuned warm-cache result, told honestly"
description: "fak's honest performance headline is ~4.1x less work than a tuned, warm-cache single-tenant stack on a 50-turn, 5-agent session (Qwen2.5-1.5B, Q8_0, Apple M3 Pro) — not the flattering 60.3x naive number. This story tells what the 4x is (prefix reuse + decode batching over an already-warm KV cache), what it is not (a throughput win over llama.cpp; a hosted-API win), and the exact conditions it holds under. Every figure is witnessed against SESSION-VALUE-STACK-RESULTS.md."
slug: the-real-4x
keywords:
  - honest benchmark
  - agent serving
  - KV cache reuse
  - prefix reuse
  - decode batching
  - multi-agent session
  - tuned baseline
  - net value-add
date: 2026-07-02
---

# The 4x that's real

**Short answer.** On a realistic 50-turn, 5-agent session, fak's fused kernel does
**~4.1x less work than a *tuned* single-tenant stack** that already has a warm per-agent KV
cache. There is a bigger number in the same benchmark — **60.3x** — and it is also real, but
it is measured against the *naive* re-prefill-every-turn pattern, and leading with it would be
the exact move that makes a perf-literate reader stop trusting you. This page tells the 4.1x
honestly: what it is, what it is not, and the conditions it holds under.

*For anyone weighing whether fak's reuse story is worth it on a self-hosted fleet. By the end
you will know which baseline the honest number is measured against, why the tuned number is the
one to quote, and where the flattering number comes from. Every figure here is witnessed against
[`SESSION-VALUE-STACK-RESULTS.md`](../../benchmarks/SESSION-VALUE-STACK-RESULTS.md) and indexed
in [`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md).*

## The setup (what was measured, on what box)

The benchmark (`cmd/sessionbench`) runs a fixed multi-agent workload on **Apple M3 Pro**
(6P+6E, 36 GB unified), model **Qwen2.5-1.5B-Instruct** at **Q8_0**, pure-Go forward pass,
native `go run`. The session shape is the "50+ turns, 5+ agents" regime the agent-serving
claim actually needs:

- **T = 50 turns · C = 5 agents · P = 2048-token shared prefix · D = 32 decode · R = 64 tool-result tokens.**
- `C` agents share one long prefix (system prompt + tool schemas). Each runs `T` turns; a turn
  decodes `D` assistant tokens, then ingests `R` private tool-result tokens. Each agent's
  context grows `P → P + T·(D+R)`: a big shared preamble, short answers, a per-agent context
  that grows every turn.

Three arms run the **same bit-identical Q8 forward pass** — the only difference is which work
is reused versus redone:

| arm | what it models | the work it pays |
|---|---|---|
| **A — naive-stateless** | a stateless API / `llama-cli -p <full prompt>` each turn | re-prefills the whole context every turn (O(T²)) |
| **B — per-agent-KV (tuned)** | a careful single-tenant stack: prompt-cache / persistent KV per agent | prefix once per agent, incremental ingestion, serial decode |
| **C — fak fused** | prefix prefilled **once** + cloned into C agents, **batched** decode, incremental ingestion | prefix once total; one weight stream serves all C |

## The result

| metric | value |
|---|---|
| arm A (naive-stateless) | 68,726 s (computed from measured prefill curve) |
| arm B (per-agent-KV, tuned single-tenant) | 4,697 s |
| arm C (fak fused) | 1,139 s |
| **net value-add vs tuned single-tenant (B/C)** | **4.1x** |
| net value-add vs naive (A/C) | 60.3x |
| turn-tax (A/B) | 14.6x |

**The 4.1x is B/C** — fak versus a stack that already has a warm KV cache. It comes from two
levers stacked together: **prefix reuse** (the 2048-token preamble is prefilled once, not five
times) and **decode batching** (one weight stream serves all five agents instead of five serial
streams), while each agent keeps ownership of its own KV span.

## What the 4.1x is *not*

- **It is not a throughput win over llama.cpp.** A tuned `llama-server` with parallel slots
  batches and reuses KV too, and on raw single-stream tok/s it is *ahead* of fak's pure-Go
  forward pass (decode ≈0.46x, prefill ≈0.15x on M3 Q8_0, per `M3-LLAMACPP-RESULTS.md`). The
  4.1x is a *reuse-vs-redo* ratio inside one fixed engine, not an engine-vs-engine race.
- **It is not a hosted-API win.** If you just call a hosted endpoint you get fak's safety floor
  and none of these savings — the reuse win is a self-hosted, read-heavy-fleet property.
- **It is not "less work" across the whole scaling sweep at 4x.** On the smaller SmolLM2-135M
  sweep the vs-tuned (B/C) number sits at **~2.4–2.7x**. The headline reaches 4.1x because it
  uses a 4x-larger prefix (P=2048), which raises the prefix-reuse component. The repo's standing
  honesty bound for the vs-tuned win is **~2–3x** (consistent with ~2–2.5x versus a tuned SGLang
  in `TURN-TAX-RESULTS.md`). Quote 4.1x as *this session's* number, not a universal constant.
- **No single lever is novel.** Continuous batching, prefix/KV reuse, and prompt caching are all
  established. The contribution is the integrated kernel that delivers them correctly in one
  binary while preserving per-agent KV ownership (each agent can `Evict`/`Clone` its own span).

## Why not lead with the 60.3x

The 60.3x (A/C) is a faithful measurement against the naive pattern — one process per agent,
re-prefilling the full context every turn, no batching. That pattern is real; it is exactly what
a hand-rolled `llama-cli -p <full prompt>` loop does. But a reader who knows serving will
immediately divide the naive re-prefill away as a strawman, and if that was your headline you
have lost them. The 4.1x survives that scrutiny because its denominator is a stack that already
did the obvious thing. **Telling the tuned number first is the credibility move** — it is what
makes every other number on the page believable.

## How honest is the arm-A number?

Arm A's O(T²) re-prefill is intractable to run fully at 50×5 (~hours on a 1.5B CPU forward
pass), so it is *computed* from `prefillCost(L)` sampled **live** at lengths across the session
— which captures prefill attention's real O(L²) growth (at L=2816 the measured cost is **2.9x**
a linear extrapolation from L=256, so a linear model would badly under-price it). A `-validate`
pass runs arm A fully live at small scale and confirms the projection:
`anchored_computed_over_live = 1.004`, `decode_A_over_B = 1.001`. Arms **B and C are fully
live**. So the flattering direction (arm A) is the one that is checked hardest against a live
run — the opposite of inflation.

## The floor that does not depend on any of this

The reuse stack is the *ceiling*. The *floor* is the engine-agnostic trust property: on the same
box, `fak turntax --suite turntax-airline` records **injections admitted to context 1 → 0** and
**destructive ops executed 1 → 0**, verdicts the model did not author, reproducible on any
backend. The happy-path control (`turntax-happy`) saves exactly **0** — the anti-inflation gate.
A 50-turn session derailed by one poisoned tool result is total waste; the floor is what keeps
the saved work real instead of confidently wrong. That floor holds even when you call a hosted
API and get none of the 4.1x.

## Reproduce it

```sh
# headline — Qwen2.5-1.5B, 50 turns × 5 agents (realistic model)
FAK_WORKERS=6 go run ./cmd/sessionbench \
  -hf <qwen2.5-1.5b-instruct-snapshot> -lean \
  -turns 50 -agents 5 -prefix 2048 -decode 32 -result 64 \
  -out experiments/session/headline-qwen-50x5.json

# the bit-identity gates the whole value-stack rests on
go test ./internal/model/ -run 'TestBatchedDecodeMatchesSerial|TestBatchFromPrefixMatchesIndependentPrefill' -count=1
```

## Honest fences

- The workload is **synthetic-but-fixed** (`cmd/sessionbench` shapes), not a captured
  production trace. The numbers are exact for that workload; a different shape shifts them.
- Arm A is **computed-from-measurement** (validated live at small scale); arms B and C are live.
- The 4.1x is *this* session (P=2048); the durable vs-tuned bound is ~2–3x. Do not carry 4.1x as
  a constant.
- fak is **not faster than llama.cpp** on raw throughput. This is a reuse axis, not a race.

---

**Related:** [`SESSION-VALUE-STACK-RESULTS.md`](../../benchmarks/SESSION-VALUE-STACK-RESULTS.md)
(the full benchmark, arms, and witnesses) ·
[`the cache cliff`](cache-cliff.md) (why the prefix-reuse win exists at all) ·
[`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md) (the central claim ledger).

_Dimension H (Benchmark-as-story) of the
[concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)._
