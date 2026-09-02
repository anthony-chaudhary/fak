---
title: "fak FAQ — Performance and the numbers"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# Performance and the numbers

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

It is durable, not zero-copy: `cas.json` holds a real copy of every byte the page table references, including the sealed poison. The sealed bytes are never paged into a context because the gate stands between them and any new window, but they are physically present on the swap device. This is a deliberate tradeoff that buys durability and a re-screenable seal across the process boundary; the zero-copy `Ref` and region-backend seam is frozen in the ABI but left unbuilt for now. A reload pages in only the working set a query touches, so resolving a follow-up does not materialize the whole image.

## Performance and the numbers

Every headline number with the baseline it is measured against — the apples-to-apples win versus a tuned warm-cache stack, never blurred with the larger figure against a naive re-send baseline.

## What is the real headline serving number, 4x or 60x?

The apples-to-apples serving number is about **4x** (4.1x) fewer tokens than a tuned warm-cache stack; the 60x figure is only against a naive re-send-everything baseline and must never be quoted as the serving win. Both come from the same 50-turn x 5-agent fleet run (Qwen2.5-1.5B Q8, M3 Pro): `net_value_add_vs_tuned = 4.12` against arm B (tuned per-agent warm KV), and `net_value_add_vs_naive = 60.3` against arm A (naive stateless). Arm A is modeled from a prefill cost function and validated live within ~0.4%; arms B and C are live. Bit-identity gates confirm the arms emit identical tokens, so the win is reuse, not a numerics shortcut.

## Why does fak report both a vs-tuned number and a vs-naive number?

Because they answer two different questions, and collapsing them into one would overclaim. The vs-tuned number (~4x on the 50x5 fleet) compares fak against a stack that already keeps a warm per-agent KV cache, so it isolates the marginal value fak adds on top of best practice. The vs-naive number (~60x) compares against re-sending the whole context every turn, which measures the total turn-tax a stateless setup pays. The benchmark authority pins every figure to a baseline letter (A = naive, B = tuned, C = fak) precisely so the two never blur.

## What does the 8.8-9.7x WebVoyager number actually measure?

It is a modeled prefill work-elimination floor over the real 643-task WebVoyager dataset, swept across 1 to 8 workers, against a naive per-turn re-prefill baseline. At 1 worker it is 8.8x (170.9M vs 19.4M prefill tokens); at 8 workers it is 9.7x (1.37G vs 141.3M). The number is deterministic prefill-token arithmetic over the real task geometry (8,745 navigation turns, median 12 per task) — not a wall-clock measurement. Against a tuned per-agent-KV stack (not the naive floor) the cross-worker reuse is only 1.0x to 1.1x. Live model runs are a separate pending phase.

## Is the WebVoyager win still 9.7x against a tuned warm-cache stack?

No. The 9.7x is purely against the naive re-prefill baseline; against a tuned per-agent KV cache the marginal WebVoyager win is only about **1.0x to 1.10x** (1 to 8 workers). This is the most important stratification caveat to keep straight: the turn-tax axis (vs naive) and the cross-worker reuse axis (vs tuned) are different measurements. WebVoyager turns are short, so once each agent already has a warm cache there is little additional shared prefix to reuse across workers.

## What is the 20-24x SWE-bench number, and against what baseline?

It is a prefill/KV work-elimination floor of **17.9x to 23.4x** (workers 1 to 16) on the 500-instance SWE-bench Verified set, measured against a naive re-prefill baseline. The per-worker rows are 17.9x at 1 worker, 22.1x at 4, 22.9x at 8, and 23.4x at 16; cross-worker reuse against a tuned cache is only 1.00x to 1.31x. This is a deterministic token floor computed from difficulty-bucket turn estimates, runs on a Mac with no GPU, and is not a head-to-head wall-clock against a tuned SGLang server. The actual code resolve-rate is a separate GPU-server run still pending.

## Where does the speedup actually come from if fak is not a faster GPU engine?

The win comes from reread-rate, not GPU speed: fak does shared prefill work once and reuses it instead of re-processing the same context every turn. A multi-agent fleet that re-sends overlapping context pays a per-turn prefill tax; fak owns the KV cache as a kernel object, so a computed prefix is cloned and reused and a tool-result span can be evicted from the middle without recomputing the tail. Raw token throughput is still won by vLLM, SGLang, and llama.cpp; fak measures itself against those honestly and does not claim to beat them on tokens per second.

## Why is the reuse win self-host only?

Because the savings come from owning the KV cache, which a frontier API does not expose. An app that merely calls a hosted provider gets fak's safety floor (the capability lock and result quarantine) but none of the prefill-reuse savings, since the KV state lives inside the provider's serving process. The frontier-scale agent-city numbers are explicitly design targets, not measurements. To get the reuse wins you run fak in front of a self-hosted model where the cache is a kernel-owned object.

## How fast is fak's policy adjudication?

The decision itself is sub-millisecond: a captured access-log line shows a policy `DENY` adjudication at `duration_ms = 0.511`. The fold runs in-process with no hook spawn, no IPC, and no engine call on the decide path, which is why the per-call cost is below typical OS clock granularity; benchmarks use an inner calibration loop to time it. On a pure-kernel decide path the allow-verdict cost has been measured as low as ~362 ns, with the in-process boundary roughly 2,400x to 2,849x cheaper than spawning a `fak hook` process per call.

## Is the sub-millisecond adjudication number the same as the fleet speedup?

No, they are unrelated measurements and should not be conflated. The ~0.5 ms adjudication is the cost of a single policy decision (a captured `DENY` log line); the in-process-vs-spawn ratio (~2,400x) is a subsystem regression sentinel for the decide path, not a serving-throughput headline. The fleet speedups (~4x vs tuned, 8.8-9.7x WebVoyager vs naive) are about prefill reuse across many turns. One is per-decision latency, the other is per-fleet token elimination.

## What does max|delta| = 0 mean for the benchmark numbers?

It is the honesty gate proving the speedup is reuse, not a numerical shortcut: reused KV state is bit-for-bit identical to a full recompute, with maximum absolute logit difference of exactly zero. Witnesses cover causal invalidation (a sibling read stays byte-identical across an external write), RadixAttention split-reuse equaling recompute, and cached-decode equaling full prefill. Because the arms emit identical tokens, the token savings cannot be explained away as a cheaper-but-different computation; the answer is the same, computed once instead of every turn.

## Is the SWE-bench code resolve-rate measured yet?

No, the resolve-rate is not yet measured; only the cost and cache-elimination arithmetic is shipped. The prefill/KV work-elimination floor (17.9x to 23.4x vs naive) runs deterministically on a Mac with no GPU, but the actual fraction of SWE-bench Verified instances that fak's agent resolves is a GPU-server run that is still pending. A local 135M model produces a resolve-rate near zero; the real number requires a larger model on the GPU server. Treat the 20-24x as a token floor, never as a claim about how many bugs get fixed.

## How big can the fleet win get on ultra-long contexts above 100k tokens?

On contexts above 100k tokens the apples-to-apples fleet floor is about **4.3x versus a warm per-agent KV cache**; against a naive re-prefill baseline the same work floor is roughly 10x for a single session and 40x+ for the fleet, though that easy baseline is never the serving win. The single-session win (9.9x token, 9.5x FLOP) is entirely the turn-tax, since one session has no cross-agent prefix to share. These are exact contention-free work floors from token and O(L^2) FLOP arithmetic, computed with the `longctxbench -ladder` command; a live wall-clock measurement above 100k is separately gated and still simulated.

## What is the right serving baseline if I already run a tuned SGLang server?

Against a tuned SGLang server the realistic serving win is roughly **2x to 2.5x**, not the 5x to 15x figures, which apply only versus naive single-tenant serving or the cache-favorable vDSO subset. The vDSO fast-path numbers in particular use a deliberately cache-favorable demo slice; on a real tau2-airline workload the addressable-vDSO purity is about 0.7%, so the vDSO is an upside secondary, never the headline. When you already have a warm-cache engine, the marginal value fak adds is the bounded 2x to 2.5x band plus the safety floor.

## Does fak's turn-tax saving claim a general speedup?

No. The turn-tax demo that deletes 9 extra model turns runs on a deliberately cache-favorable 14-call airline slice (about 64% addressable) and is not a general speedup. On a real tau2-airline workload the addressable vDSO purity is about 0.7%, which works out to roughly 0.33 turns saved per session, so a self-host build does not amortize on efficiency alone. The durable, engine-agnostic part of that benchmark is the safety floor: injections admitted to context go 1 to 0 and destructive ops executed go 1 to 0, reproducible on any backend.

## What proves the modeled naive baseline is not inflated?

The naive arm is validated live to within about 0.4%: the ratio of anchored-computed to live cost is 1.0039, so the README's "within ~1%" framing is conservative. The naive total of roughly 19.1 hours is modeled from a prefill cost function because running it live really does take about that long, while fak's fused arm at ~19.0 minutes is live. There is also an anti-inflation control: a clean 3-call happy-path workload saves exactly zero by construction and by test, so the harness cannot manufacture a win where none exists.

## Has the +1 retry-turn cost of an injection been seen live, or only modeled?

