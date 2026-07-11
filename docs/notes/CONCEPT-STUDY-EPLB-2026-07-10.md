# CONCEPT-STUDY: deepseek-ai/EPLB — load-aware expert placement, through the expert-residency/caching lens

> Borrow-hunt pass, 2026-07-10. Uncommitted study note (scout-loop record). Source repo
> pinned at `deepseek-ai/EPLB` `eplb.py` @ `d52c72d5b2f2fb4c41afbf8eb21366820239913d`
> (full history: 9 commits, single 165-line file, no tests, no CI — read in full).

## What EPLB is

The Expert-Parallel Load Balancer: given per-expert load, compute a physical placement of
MoE experts across GPUs/nodes that minimizes the **max per-rank load**. Four functions:

- `balanced_packing(weight, num_packs)` (`eplb.py:5-41`) — LPT greedy multiway partition with
  an **equal-cardinality** constraint: sort items heaviest-first, drop each into the
  least-loaded pack that still has room. Minimizes max pack *weight*, N/M items per pack.
- `replicate_experts(weight, num_phy)` (`eplb.py:44-71`) — **water-filling replication**: to
  spend a fixed budget of physical replicas, repeatedly add one replica to the logical expert
  with the highest current per-replica load `weight/replicas` (line 67
  `redundant_indices = (weight / logcnt).max(dim=-1)`). Hot experts earn more replicas.
- `rebalance_experts_hierarchical(...)` (`eplb.py:74-128`) — group→node→GPU 3-step
  locality-aware placement: pack whole routing groups onto nodes first (co-residency cuts
  inter-node all-reduce traffic), replicate within node, then pack replicas to GPUs.
- `rebalance_experts(...)` (`eplb.py:130-165`) — entry point; picks hierarchical (locality,
  small EP / prefill) vs global (max replication, large EP / decode) by `num_groups % num_nodes`.

**Design lesson from history** (`e1100fe` "add gpu-level load balance for global policy"): the
global policy originally called `replicate_experts` alone and left GPUs unbalanced; the fix
routes it through the hierarchical packer (`num_groups=1, num_nodes=1`). Replication **must**
be followed by a packing step — `replicate → pack`, never replicate alone.

## Candidates and witness verdicts (against C:/work/fak)

| # | Borrow | Caching lens | Verdict vs fak | Disposition |
|---|---|---|---|---|
| A+B | Load-aware expert placement: pack ranks by measured load + replicate hot experts | expert-residency | **PARTIAL/ABSENT** | **Filed #3886** |
| C | Prefix-cache affinity routing (route request to the instance holding its prefix KV) | prefix-cache | **PRESENT** | drop |
| D | Phase/regime-aware policy split (prefill=locality, decode=global) | tiering policy | PRESENT-ish | note only |
| E | Hot-*prefix* replica-count planner (water-filling applied to KV-prefix fleet replicas) | prefix-cache/tiering | **UNPROVEN** | deferred (see below) |

### A+B — FILED as #3886 (the one real gap)

fak's native EP places experts with `ExpertParallelPlan` (`internal/model/expert_parallel.go:68`)
→ `NewTPPlan` (`internal/model/tensor_parallel.go:70-92`), whose only inputs are `numExperts`
and `ranks`: contiguous **equal-count** bands (`base=dim/ranks`, `extra=dim%ranks`),
**load-oblivious**. The EP note even ships a test that tolerates the symptom without fixing it —
*"load-imbalanced (all picks on one rank) ranks=2 == ranks=1, max|Δ|=0"*
(`docs/notes/GLM52-EXPERT-PARALLEL-MULTIGPU-2026-06-29.md:38`). fak proves it *reduces*
correctly under imbalance; it never *rebalances*. EPLB's `balanced_packing` + `replicate_experts`
+ group locality are exactly the load-aware layer fak lacks. fak only *benchmarks* vLLM's EPLB as
an external floor (`docs/benchmarks/VLLM-EP-EPLB-MOE-BASELINE-RUNBOOK.md`, #1733 closed).

### C — PRESENT (dropped)

fak already does prefix-cache-aware fleet routing: `internal/gateway/kv_fleet_routing.go`
routes a request across the fleet to the instance holding its prefix KV ("overlap-minus-load
placement"), `internal/gateway/replica_router.go` has a `CacheAwarePolicy` scoring candidates by
"prefix residency × inverse load" (#41), and `internal/session/cache_affinity.go` preserves the
affinity key across continuations. EPLB adds nothing here.

### E — deferred, NOT filed (honesty)

EPLB's water-filling (`replicate_experts`) could in principle apply to KV-prefix caching:
replicate a *hot prefix* onto more fleet instances proportional to its request load to cut max
per-instance load. fak's `kv_fleet_routing.go` routes to holders and balances a *cold* prefix to
the least-loaded instance, but I did not find (in a 70-line read) a *load-proportional
replica-count planner* for hot prefixes. This is a plausible second borrow, but claiming it ABSENT
would need a deeper read of `residency_router.go` / `replica_router.go` membership / cachemeta.
Left as a thread rather than a possibly-duplicate issue — a false-ABSENT is worse than none.

## Dedup performed

- No `CONCEPT-STUDY-EPLB` note existed before this pass.
- Neighbors confirmed distinct: **#3174 / #3212** (expert-weight *residency/offload* tier —
  VRAM vs host RAM, shared=always-hot) are a different axis from cross-rank placement;
  **#3771 ELDR** is decode *routing* locality (triage-only); **#1733 closed** is the external
  vLLM EPLB *benchmark floor*, whose title explicitly defers "native expert placement."

## Filed

- **#3886** — feat(moe): load-aware native EP expert placement (EPLB balanced-packing +
  hot-expert replication) — replace load-oblivious `NewTPPlan` bands. Phased: P1 load-weighted
  band sizing (no new residency, reduce path unchanged), P2 hot-expert replication (gated on
  #3174/#3212 residency work).
