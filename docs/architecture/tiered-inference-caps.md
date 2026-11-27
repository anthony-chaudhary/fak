---
title: "Tiered inference caps: L1/L2/L3 KV-cache slots, elastic borrowing, and agentic capacity awareness"
description: "How fak replaces binary max_model_len vs max_num_seqs trade-offs with multi-level L1/L2/L3 KV-cache tiers, elastic idle-slot borrowing, deterministic autotuning, and end-to-end capacity communication channels."
---

# Tiered inference caps: L1/L2/L3 KV-cache slots, elastic borrowing, and agentic capacity awareness

[← Architecture](../architecture.md) · Implementation: [`internal/kvbudget/tiers.go`](../../internal/kvbudget/tiers.go)

## Why binary `max_model_len` vs `max_num_seqs` breaks agentic hierarchies

Traditional serving engines force a binary trade-off:
- High `max_model_len` (e.g., 128k) starves concurrency (`max_num_seqs` collapses to 1–3).
- High `max_num_seqs` caps every request to a tiny context (e.g., 4k–8k), breaking long-context orchestrators.

Agentic workloads (like Ultracode) are **hierarchical**, not uniform:
- **1 master orchestrator** holds deep repository/plan context (up to 128k).
- **Several workers** hold bounded task slices (up to 32k).
- **Many micro-agents** run short classification, lint, or tool-verification loops (up to 4k).

## L1 / L2 / L3 cache analogy

`fak` models the KV-cache pool like a CPU cache hierarchy:

| Tier | Constant | Role | Default MaxContext | Preemptible |
|---|---|---|---|---|
| **L1** | `TierMaster` (`"master"`) | Master orchestrator / deep planner | `131072` (128k) | No |
| **L2** | `TierWorker` (`"worker"`) | Bounded coding / research worker | `32768` (32k) | No |
| **L3** | `TierMicro` (`"micro"`) | Fast micro-agent / classifier / verifier | `4096` (4k) | Yes |

`TierPool.Classify(tokens)` routes any request to the smallest tier whose `MaxContext >= tokens`, avoiding head-of-line blocking where short L3 bursts wait behind L1 slots.

## Slab split/merge and elastic borrowing

`AutotuneTiersFromTokenBudget` and `AutotuneTiers` enforce the **conservation invariant**:

```
sum(MinSlots_i * MaxContext_i) <= TotalTokenCapacity
```

At runtime, `TierPool` acts as a slab allocator with elastic borrowing:
1. **Guaranteed floor first**: `Acquire(tier, tokens)` allocates from the tier's own `MinSlots` floor (`Borrowed = false`).
2. **Elastic borrow from idle capacity**: When a tier exceeds `MinSlots`, it borrows unallocated tokens from idle tiers that set `AllowBorrowFromIdle = true` (`Borrowed = true`). For example, an idle 128k L1 slot can lend four 32k chunks to L2 or thirty-two 4k chunks to L3, while respecting active allocations and non-borrowable floors.
3. **Hysteresis-guarded rebalance**: `Rebalance(demand)` shifts `MinSlots` toward high-demand tiers only when excess demand crosses a hysteresis threshold (`max(2, MinSlots/4)`), preventing thrashing under transient bursts.
4. **Idempotent release**: `Release(lease)` tracks active lease IDs internally so double-releases never corrupt `ActiveSlots` or `FreeTokens`.

## 16GB Aegis Inkling vs. Datacenter mapping

### 16GB Aegis Inkling edge envelope (384k tokens = 393,216 tokens)

With `ProfileUltracode` at `384 * 1024 = 393,216` tokens:
- **L1 (`master`)**: `1 × 131,072` (128k) = `131,072` tokens (33.3%)
- **L2 (`worker`)**: `4 × 32,768` (32k) = `131,072` tokens (33.3%)
- **L3 (`micro`)**: `32 × 4,096` (4k) = `131,072` tokens (33.3%)
- **Total**: `393,216` tokens — exact 100% fit with simultaneous co-admission of all 37 slots.

### Datacenter mapping (206 GiB raw / 165 GiB usable `GLM52DSA` @ F16)

`AutotuneTiers(usableBytes, GLM52DSA, F16, ProfileDatacenterMultiTenant)` derives `>1.36M` tokens at 165 GiB usable VRAM and scales `MinSlots` proportionally across L1 (`128k`), L2 (`32k`), and L3 (`4k`) while preserving the exact same `TierPool` admission and elastic borrowing semantics. Across multi-node clusters, L1 maps to high-VRAM MLA nodes, L2 to mid-tier decode nodes, and L3 to high-batch FP8/preemptible spot nodes.

## The 5 end-to-end communication channels

To prevent agents from blindly over-allocating or hitting opaque 429s, tier capacity is exposed across 5 coordinated channels:

1. **Capacity discovery endpoint (`GET /v1/fak/capacity/tiers`)**: Returns `TierPoolSnapshot` (`total_tokens`, `free_tokens`, per-tier `available_now` / `elastic_available`, and `recommended_fanout`).
2. **Client SDK discovery (`pkg/fakclient/tiers.go`)**: Provides `CapacityTiers(ctx)` and `CapacityTiersSnapshot` over `/v1/fak/capacity/tiers` (with future `/v1/models` metadata planned).
3. **Per-turn HTTP response headers (`X-Fak-*`)**: `WriteTierHeaders` sets `X-Fak-Context-Tier`, `X-Fak-Tier-Cap-Tokens`, and `X-Fak-Available-Tiers` on HTTP responses.
4. **Structured 429 downgrade advice**: When a requested tier is exhausted (`AdviceCode: "TIER_EXHAUSTED_DOWNGRADE_AVAILABLE"`), `BuildTierAdvice` and `WriteTierShedResponse` emit typed `TierAdvice` (`DOWNGRADE_TO_WORKER`, `SPLIT_TO_MICRO`, `COMPACT_CONTEXT`, `WAIT_FOR_SLOT`) and `X-Fak-Available-Tiers` so callers downgrade immediately instead of backing off blindly.
5. **Agent topology planning (`internal/agent/tier_awareness.go`)**: `PlanAgentTopology`, `FormatTierAwarenessPrompt`, and `SelectTierForSubtask` translate `CapacityTiersSnapshot` into 80% compaction watermarks and bounded worker/micro fan-out counts.
