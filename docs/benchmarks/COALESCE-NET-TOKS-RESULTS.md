---
title: "fak coalescebench: PROJECTED net-tok/s(B) for SSD-offloaded MoE expert coalescing"
description: "The deterministic replay bench (cmd/coalescebench) drives the cross-agent expert-cache coalescing simulator over a synthetic GLM-5.2-shaped router and projects the net-tok/s(B) roofline curve plus the coalescing ratio C(B). Every cell is PROJECTED — a roofline over labelled inputs, not a served measurement; the real-trace witness is #5251."
---

# COALESCE-NET-TOKS-RESULTS — the projected net-tok/s(B) curve (roofline, NOT measured)

> **Read this first.** Every number in this document is **PROJECTED**: a roofline computed
> from a synthetic, deterministically seeded router and labelled bandwidth/byte inputs. **No
> cell is a served measurement.** The bench proves exactly one thing at witness grade — the
> *coalescing arithmetic itself* (how the distinct-expert union `U(B)` and the SSD page-in
> stream behave as agent count B grows, replayed through the deterministic
> `deepseekv4moe.SimulateExpertCacheBatch` LRU of #5244). Everything downstream of that
> arithmetic (the tok/s columns) divides labelled placeholder bandwidths into simulated byte
> counts. **This projection stands as a roofline pending the real-trace witness (#5251)** —
> captured GLM-5.2 top-K routes replacing the synthetic router — and, beyond that, an actual
> served run. The single-agent reality on this class of box is WITNESSED elsewhere and low:
> 0.243 tok/s decode, GLM-5.2 CPU-offload (`glm52-lab-benchmark-results`).
>
> Reproduce: `go run ./cmd/coalescebench` (defaults below; same flags → byte-identical
> table, enforced by `TestRunBenchDeterministic`). Design: `docs/notes/MOE-SSD-MULTI-AGENT-NET-TOKS-2026-07-18.md`
> §2 (roofline) and §2.1 (the two baselines). Epic: #5243; bench ticket: #5245.

## What is being projected (the §2 roofline)

B concurrent agents advance one decode step together on an SSD-offloaded MoE. Each layer,
each agent picks top-K experts; the batch streams each **distinct** (layer, expert) group
once. Per-agent seconds-per-token is the max of three rooflines:

```
t_agent(B)  = max( SSD_term(B), RAM_term(B), FLOP_term )
SSD_term(B) = PageIns_per_step · e / (B · BW_ssd)     # the coalesced expert stream (§2's Σℓ Mℓ —
                                                      # simulator Misses: groups not already resident)
RAM_term(B) = ( NR + U_step(B) · e ) / BW_ram         # the dense/resident roofline
FLOP_term   = active_flops_per_token / FLOPS          # the compute roofline
net_toks(B) = B / t_agent(B)
```

The page-in counts come from the **deterministic simulator**, not a closed-form guess: the
synthetic router (uniform→Zipf skew knob) generates every agent's top-K per (layer, step),
and `SimulateExpertCacheBatch` replays them against a bounded LRU. One API note, stated so
the columns bind to the right field: the landed #5244 `DistinctStreamed` counts *all*
distinct groups per step (Hits+Misses); the SSD stream in the §2 sense — groups **actually
paged in** — is the trace's `Misses`, which is what `SSD_term` uses here.

## Inputs (flags), with provenance labels

| Input | Default | Provenance |
|---|---|---|
| L (MoE layers) | 76 | **HEADER-DERIVED** — GLM-5.2 GGUF header: 79 layers = 3 dense + 76 MoE (`GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md` §2) |
| N (experts/layer) | 256 | **HEADER-DERIVED** (same doc) |
| K (experts/token) | 8 | **HEADER-DERIVED** (same doc) |
| e (bytes per (layer,expert) group) | 21.81 MiB | **HEADER-DERIVED** — 1.619 GiB/expert across 76 MoE layers = 1.619·1024/76 (same doc §2) |
| NR (non-routed active bytes/token) | 19.31 GiB | **HEADER-DERIVED** (same doc §2) |
| BW_ssd | 7 GiB/s | **PLACEHOLDER** — PCIe-Gen4 NVMe class; the striped-read hardware witness (#4298 stripeload) is operator-gated |
| BW_ram | 200 GiB/s | **PLACEHOLDER** — server-class host memory bandwidth |
| active FLOP/token | 64 GFLOP | derived ~2×32 B active params (same doc §3.2) |
| FLOPS | 1000 GFLOP/s | **PLACEHOLDER** — host-GEMM ~1 TFLOP/s class |
| expert-cache | 128 GiB (6009 groups) | **PLACEHOLDER** — the epic's premise is a RAM tier smaller than the 414.5 GiB routed band |
| router skew | Zipf s=1.0 | **PLACEHOLDER** — real GLM-5.2 router-load skew is exactly what #5251 captures |
| steps / seed | 32 / 1 | replay length and PRNG seed (splitmix64; no time, no global PRNG) |

## The projected curve (default flags; every cell PROJECTED)

`go run ./cmd/coalescebench` — output verbatim, 2026-07-18:

| B | U(B) | C(B) | SSD_term | net_toks | ×vs-uncoalesced | ×vs-1agent | binding-roofline | label |
|--:|-----:|-----:|---------:|---------:|----------------:|-----------:|:-----------------|:------|
| 1 | 8.0 | 1.00 | 0.7996 s | 1.25 | 1.00× | 1.00× | SSD | PROJECTED |
| 4 | 25.1 | 1.28 | 0.6563 s | 6.09 | 1.22× | 4.87× | SSD | PROJECTED |
| 16 | 68.4 | 1.87 | 0.5527 s | 24.61 | 1.23× | 19.68× | RAM | PROJECTED |
| 64 | 152.6 | 3.36 | 0.5512 s | 48.07 | 0.60× | 38.44× | RAM | PROJECTED |
| 128 | 201.3 | 5.09 | 0.3636 s | 74.18 | 0.46× | 59.31× | RAM | PROJECTED |

**Regime transition (PROJECTED): B\* = 16** — the first swept B where `SSD_term` (0.5527 s)
drops below `RAM_term` (0.6502 s). Past B\*, the coalesced expert stream no longer binds:
the "slow SSD box" is riding the resident RAM roofline, which is the dense
[`MODEL-BATCHING-RESULTS.md`] regime.

- `U(B)` = mean distinct experts per (layer, decode-step): 8.0 → 201.3 while `B·K` goes
  8 → 1024. That sublinearity IS lever L2.
- **`C(B) = B·K/U(B)` reaches 5.09× at B=128** — inside the 4–10×+ band §2.1 projects the
  coalescing win to live in, before the resident transition. This column and B\* are the
  defensible headline.
- Aggregate `net_toks` climbs 1.25 → 74.18 tok/s (×59.3 the single-agent rate) — an
  **aggregate, latency-tolerant** number only (see baselines below).

## The two baselines, side by side (§2.1) — state the denominator or say nothing

- **×vs-uncoalesced** `= net_toks(B) ÷ (B · net_toks(1))`: against B independent
  un-coalesced streams *each granted the full SSD bandwidth* — i.e. effectively **B separate
  boxes**, an **OPTIMISTIC upper-baseline** (the landing-review note on #5245). While the
  batch is SSD-bound this shows the per-stream coalescing win (1.22–1.23× at B=4–16 under
  uniform-ish skew s=1). **Past B\* it falls below 1 (0.46× at B=128)** — one box riding its
  RAM roofline loses to B whole boxes, as it must. That is not a defect of coalescing; it is
  why §2.1 says the defensible headline is **C(B) and the regime transition**, not any bare
  multiple.
- **The same-box physics baseline — the un-coalesced shared-SSD floor** (printed under the
  table): B un-coalesced, un-cached streams time-sharing ONE SSD each stream the full
  `L·K·e` expert bytes per token, so their aggregate is `BW_ssd/(L·K·e)` = **0.54 tok/s at
  ANY B** (defaults). The coalesced curve's rise above this constant floor (1.25 → 74.18
  PROJECTED) is the modeled fak-kernel coalescing+cache win on the same box.
- **×vs-1agent** `= net_toks(B) ÷ net_toks(1)`: 59.31× at B=128. This is the
  aggregate-over-latency number the affordable-fleet note warns against citing raw —
  report it only as "aggregate, latency-tolerant"; no single user's token gets faster.

These are **inference-throughput** numbers. They must never be blended with the
agent-orchestration concurrency metric (`ultracode-multi-agent-dogfood.md`): net tok/s on
one box says nothing about issues closed per hour.

## What would make this a measurement (the honest gap)

1. **#5251 — real routing traces.** Capture per-token top-K from a real GLM-5.2 forward
   and replay them through this same bench (`U(B)` and skew stop being synthetic). Until
   then the skew knob is a placeholder and every row here is a roofline.
2. **Measured bandwidths.** `BW_ssd`/`BW_ram` from the stripeload hardware witness
   (operator-gated) replace the placeholder denominators.
3. **A served run.** The roofline is an upper-bound *shape*; only a served multi-agent
   decode on the target box turns any cell WITNESSED. The B=1 witnessed reality (0.243
   tok/s) sits well below this B=1 projection (1.25 tok/s), which is exactly what a
   roofline-vs-served gap looks like.

## Determinism (the one thing this bench does witness)

Same flags → byte-identical table: the router is a seeded splitmix64 PRNG (no
`time`, no global `math/rand`), the simulator is the deterministic #5244 LRU, and
`cmd/coalescebench`'s `TestRunBenchDeterministic` enforces it (plus: seed reaches the
router, skew concentrates `U(B)`, the §2 arithmetic and the shared-SSD floor are pinned exactly, and the bench's
per-step unions cross-check the simulator's `DistinctStreamed` count, failing closed on
disagreement).
