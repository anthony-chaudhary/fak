---
title: "Net tokens/sec: why a slow SSD-offloaded MoE box scales with agent count — the expert-coalescing lever (2026-07-18)"
description: "A roofline model for the multi-agent inference-throughput advantage on frontier MoE served from SSD. Composes three levers — shared weight-stream amortization, cross-agent expert-working-set coalescing, and cross-agent prefix KV reuse — into a net-tok/s(B) curve, and shows why the batch-1 'slow' rate (GLM-5.2 sub-1 tok/s) is the wrong meter: with B concurrent agents the routed-expert working set coalesces, the SSD stream per agent-token falls, and aggregate net tok/s rides up onto the resident roofline. Every cell labelled measured vs projected. Kept in the INFERENCE-throughput claim family and explicitly UNBLENDED from the agent-orchestration concurrency metric."
---

# Net tokens/sec on a slow SSD-offloaded MoE — the expert-coalescing lever

Date: 2026-07-18

> **The claim, in one line.** On a frontier MoE (GLM-5.2, 753 B, 256 experts, K=8) whose
> routed experts stream from SSD, the *single-agent* decode rate is bandwidth-bound and
> "slow" (sub-1 tok/s CPU-offload; 0.243 tok/s WITNESSED on GPU server 2 CPU-only, see
> [`glm52-lab-benchmark-results`]). That number is the **wrong meter** for a fleet. Run
> **B concurrent agents** and their per-step top-K expert selections **coalesce**: the union
> of experts a batch touches is far smaller than `B·K`, so each SSD-streamed expert serves
> many agents at once. The SSD bytes *per agent-token* fall as B grows, per-agent tok/s
> holds (or rises), and **aggregate net tok/s scales with B** — 4–10×+ over B independent
> un-coalesced streams — until the working set becomes fully resident and the box rides its
> ordinary RAM/compute roofline. The slow box was never slow; it was *under-batched*.

> **What this is.** A roofline model + lever decomposition that seeds a fleet of tickets
> (epic below). **What this is NOT.** A measurement. Only cells tagged **WITNESSED** are
> served numbers; everything else is **PROJECTED** from a labelled input. The load-bearing
> new lever (cross-agent expert coalescing) has **no runtime witness yet** — the first two
> child tickets build the deterministic simulator and the replay bench that produce one.

## 0. Scope guard — which claim family this is (read first)

This is an **inference-throughput** claim: aggregate decode tokens/sec under multi-agent
batching, the same axis as [`docs/benchmarks/MODEL-BATCHING-RESULTS.md`] (dense CPU batched
decode, WITNESSED 41× aggregate at B=960). It is **NOT** the agent-orchestration concurrency
metric defined in [`docs/explainers/ultracode-multi-agent-dogfood.md`] (a wall-clock
concurrency factor N with an Amdahl ceiling over *deliverables*). Per that doc's one
load-bearing rule, **the two multiples must never be blended.** "100 agents × 2 tok/s =
200 net tok/s" is an *inference* statement about token production on one box; it says nothing
about how many issues close per hour. Keep them in separate sentences and separate columns.

## 1. The three levers (what compounds)

A slow SSD-offloaded MoE box, served to a fleet of agents, gets faster *per agent-token* for
three independent reasons that multiply:

| # | Lever | Axis it amortizes | Status in fak |
|---|---|---|---|
| **L1** | **Weight-stream amortization** (dense) | The RAM-resident non-routed band + shared expert: one weight stream, B rows, stacked into one GEMM. | **WITNESSED** for dense — `internal/model/batch.go`, [`MODEL-BATCHING-RESULTS.md`]. Applies to MoE's ~60% non-routed band unchanged. |
| **L2** | **Cross-agent expert coalescing** (MoE-SSD, **the new lever**) | The SSD-streamed routed experts: B agents' top-K selections union to `U(B) ≪ B·K` distinct experts; each streamed expert serves every agent in the batch that routed to it. | **NOT BUILT.** The intra-agent analogue (draft/verify window union) is #4355; the single-stream residency trace is `deepseekv4moe.SimulateExpertCache`. Cross-**agent** coalescing is the epic below. |
| **L3** | **Cross-agent prefix KV reuse** | Prefill / first-token: the shared system prompt + tool schemas + repo snapshot every agent carries — computed once for all B, not B times. | **WITNESSED** mechanism — `internal/radixkv` longest-prefix reuse; public fan-out clone #1535. Quantified per [`SCALING-LAWS-OF-AGENTS`] `agents × turns × working-set × reread-rate`. |

L1 and L3 already ship and are witnessed. **L2 is the missing piece and the reason a *MoE*
box scales differently from a dense box** — a dense batch amortizes one fixed weight stream;
an MoE batch additionally *shrinks its own SSD working set* as it grows.

## 2. The roofline — net-tok/s(B)

Let, per decode step:

- `BW_ssd` — SSD/host read bandwidth (bytes/s), the binding resource for offloaded experts.
- `BW_ram` — resident memory bandwidth (bytes/s), the binding resource once experts are cached.
- `e` — bytes per routed expert group at the served quant (GLM-5.2 UD-Q4_K_M: **~1.619 GiB/expert** across all MoE layers per token's route is the wrong grain — use the per-(layer,expert) group byte size `RoutedExpertGroupBytes`; see [`GLM52-DGX-THEORETICAL-CEILING`] for the header-derived active-bytes split).
- `NR` — non-routed active bytes/token (RAM-resident): dense layers + shared expert + attention. GLM-5.2 header-derived **~19.31 GiB** (≈60% of active).
- `K` — experts/token (8). `N` — experts/layer (256). `L` — MoE layers (76).
- `U_ℓ(B)` — distinct experts selected at layer ℓ by the B agents this step, `≤ min(B·K, N)`.
- `M_ℓ(B)` — of those, the ones **not already resident** (the SSD misses).

Per-agent seconds-per-token is the max of three rooflines (the batch runs at the slowest):

```
t_agent(B) = max(
    SSD_term  = ( Σ_ℓ M_ℓ(B) · e )              / (B · BW_ssd),   # coalesced expert stream
    RAM_term  = ( NR + Σ_ℓ U_ℓ(B)·e_resident )  /       BW_ram,   # dense/resident roofline (L1)
    FLOP_term = active_flops_per_token           /       FLOPS     # compute roofline
)
net_toks(B) = B / t_agent(B)
```

The whole thesis lives in `Σ_ℓ M_ℓ(B) / B` — **SSD misses per agent-token**:

- **B = 1:** `U_ℓ = K`, few cache hits → `SSD_term ≈ Σ_ℓ K·e / BW_ssd` — huge; this is the 0.243 tok/s wall.
- **B growing:** `U_ℓ(B)` grows *sublinearly* (bounded by `N`, concentrated further by routing skew — real router load is not uniform). So `M_ℓ(B)/B` **falls**: the same experts get reused across more agents. `SSD_term` per agent drops.
- **B large enough that `U_ℓ(B) → N`:** every expert is touched every step → after warmup the *entire* expert set is resident → **`SSD_term → 0`** and the batch rides `RAM_term`/`FLOP_term` — i.e. the dense [`MODEL-BATCHING-RESULTS`] curve (up to 41× aggregate). **The SSD-streaming box has become a resident box, for free, by batching.**

### 2.1 Where the 4–10×+ comes from (two honest baselines)

The multiple depends entirely on what you compare against — state the baseline or the number
is meaningless:

- **vs B independent un-coalesced streams** (`net = B · tok/s₁`, linear but each stream pays full SSD): the advantage is the **coalescing ratio** `C(B) = (B·K) / U(B)` on the SSD term, plus the regime transition. `C(B)` is where 4–10×+ is projected to live before the resident transition; **the first two tickets measure it.** This is the honest "our caching makes the same slow box do more."
- **vs a single agent's latency** (`net_toks(B) / tok/s₁`): this is large (tens×) but is the *aggregate/latency conflation* the affordable-fleet note ([§0, `AFFORDABLE-HARDWARE-AGENT-FLEETS`]) warns against citing raw. Report it only as "aggregate, latency-tolerant," never as a speedup a single user feels.

The defensible headline is **the coalescing ratio and the regime transition**, not a bare
aggregate-over-latency number.

## 3. The "per user, then hyperscaler" composition

The B agents above are **one user's fan-out** (an ultracode session, a coding-agent swarm, an
overnight backlog drive). The coalescing is *within* that user's batch and needs no
cross-tenant trust. In a hyperscaler, ordinary **cross-user continuous batching** (PagedAttention-
class) composes *on top*: the per-user coalesced batch is itself one contributor to the
machine-wide batch. So the machine-level net-tok/s is `(per-user coalesced) ⊗ (cross-user
batch)` up to the SSD/compute roofline — the per-user lever does not compete with datacenter
batching, it **feeds** it. fak owns the per-user, in-kernel, trust-scoped half (the `Evict`/
`Clone` per-user KV survives batching, [`MODEL-BATCHING-RESULTS`]); the cross-user half is the
standard serving stack's job. Ticket G states this composition precisely and its non-overlap.

## 4. Ticket map (the epic this note seeds)

Epic: **Net tokens/sec — multi-agent throughput on slow SSD-offloaded MoE via expert coalescing.**

| Ticket | Lever | Gate | Dispatchable now? |
|---|---|---|---|
| **A** Cross-agent expert-coalescing simulator (`SimulateExpertCacheBatch`) | L2 | none — pure Go, deterministic | **yes (top)** |
| **B** Net-tok/s replay bench (`cmd/`, B=1…128 curve + coalescing ratio) | L2 | none — CPU-only | **yes (top)** |
| **C** Coalescing ratio + effective-bytes/token as an admission/telemetry metric; wire to `fak serve --plan-json` (#4361) | L2 | none | yes |
| **D** Fused grouped-GEMM over the coalesced resident expert union + serial-parity test | L1×L2 fusion | CPU path yes; GPU kernel gated | partial |
| **E** L2×L3 compounding proof (shared prefix computed once AND experts coalesced) | L2×L3 | none — replay | yes |
| **F** Routing-skew model: real GLM-5.2 top-K load is non-uniform → `U(B)` grows slower than uniform | L2 | none (uses captured trace) | yes |
| **G** Per-user ⊗ cross-user batching composition note + non-overlap statement | L2/L3 | none — doc | yes |
| **H** Witness the coalescing ratio on **real** GLM-5.2 routing traces (capture top-K per token, measure `U(B)`) | L2 | operator/hardware (lab GLM-5.2 forward) | gated |
| **I** SSD-offload net-tok/s demo cell for the #4762 browser demo (aggregate climbs with B, per-agent flat) | L1/L2/L3 | ties to #4762 | after A/B |

Tickets A and B are the spine: they turn the projected coalescing curve into a **witnessed,
deterministic** one with zero hardware. Everything downstream (the demo, the metric, the real-
trace witness) rests on the number they produce.

## 5. Prior art in-repo (do not re-derive)

- Dense batched-decode throughput axis, WITNESSED: [`docs/benchmarks/MODEL-BATCHING-RESULTS.md`], `internal/model/batch.go` (L1).
- Single-stream expert residency trace: `internal/deepseekv4moe/cache_trace.go` `SimulateExpertCache` — A extends this to the batch/multi-agent axis.
- Intra-agent (draft/verify) expert batch-union: **#4355** (colibri epic #4352) — A is the cross-**agent** sibling.
- SSD expert streaming read path: `internal/stripeload`, #4298/#2722; readahead #4359; residency policy #4357.
- Prefix reuse (L3): `internal/radixkv`, fan-out clone #1535, [`SCALING-LAWS-OF-AGENTS`].
- Affordable-fleet framing + the "aggregate is not latency" honesty rule: [`AFFORDABLE-HARDWARE-AGENT-FLEETS`] §0.
- Orchestration-concurrency metric (kept UNBLENDED from this): [`ultracode-multi-agent-dogfood.md`].
- GLM-5.2 header-derived active-bytes split + resident roofline: [`GLM52-DGX-THEORETICAL-CEILING`].
