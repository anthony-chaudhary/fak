---
title: "Serving SOTA decision matrix: P/D disaggregation + KV routing (2026-06-26)"
description: "A one-page ride-vs-own matrix comparing vLLM, SGLang, LMCache, Dynamo, Mooncake, and current fak across prefix cache, P/D split, KV transfer, routing, autoscaling, metrics, and invalidation — plus the source-tagged CacheEvent/ServingEvent vocabulary fak normalizes them into."
---

# Serving SOTA: P/D disaggregation + KV routing — the ride-vs-own readout

> Date: 2026-06-26. Scope: the **learning/readout layer** for issue
> [#903](https://github.com/anthony-chaudhary/fak/issues/903) — normalize the current
> serving SOTA into one local decision matrix **before** adding more cache or serving
> machinery, then feed the implementation epics (§5) concrete, SOTA-derived choices.
> Status: design/positioning memo. fak's own column is tagged `[SHIPPED]` / `[SEAM]` /
> `[GAP]` so a ride-engine capability is never silently claimed as fak-native.

This memo is the **serving control-plane** companion to the **cache-layer** parity map in
[`AGENTIC-CACHING-SOTA-2026-06-19.md`](../notes/AGENTIC-CACHING-SOTA-2026-06-19.md) and the
**throughput-vs-trust** sequencing in
[`THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md`](../notes/THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md).
Where that pair maps *what to cache* and *what to build first*, this one maps *who already
owns the P/D + routing mechanics* so fak rides them instead of re-deriving them.

---

## 0. Verdict — what fak rides vs. owns

vLLM, SGLang, and Dynamo already own the hard systems mechanics — paged KV, prefix-cache
block hashing, prefill/decode disaggregation, KV-aware routing, and (Dynamo) SLA
autoscaling. fak does **not** try to out-serve them and says so in
[`llms.txt`](../../llms.txt) ("it is **not** a faster model server"). The orthogonal band
fak owns is **cache legality, provenance, policy, and agent-visible cache economics**:

> A provider or ride-engine cache hit is a **performance fact, not an authorization fact**.
> fak still owns cache **admission, scope, taint, and invalidation** verdicts over the
> reused object, and keeps the four reuse **sources** separable so a ridden engine's saving
> is never reported as a fak-native win.

| Band | Owner | fak posture |
|---|---|---|
| Paged KV, continuous batching, prefix-block hashing | vLLM / SGLang | **ride** — `fak serve` fronts an OpenAI/Anthropic wire ([#451](https://github.com/anthony-chaudhary/fak/issues/451) proved the tool-call wire GPU-free) |
| Prefill/decode disaggregation, KV-aware routing | SGLang router / Dynamo | **ride** — native P/D is a hardware-gated `[GAP]` (dual-track S6b) |
| Cross-instance KV transport (RDMA/NVMe-oF) | Mooncake / NIXL / LMCache | **ride** the transport; **own** the `[from,len)` span identity + materialization key |
| SLA-based autoscaling / planner | Dynamo planner | **ride** — out of fak's short-term scope |
| **Cache admission / scope / taint** | — | **own** `[SHIPPED]` — `internal/ctxmmu`, `internal/cachemeta.Security` |
| **Bit-exact mid-run causal eviction + deletion certificate** | — | **own** `[SHIPPED]` over fak-owned KV — no shipped engine offers this (`internal/model/kvcache.go`, `internal/deletioncert`) |
| **Source-tagged cache economics** (don't blend savings) | — | **own** `[SHIPPED]` — `internal/cachemeta` planes + the source-tag regression (§4) |

---

## 1. The decision matrix

Rows = systems; columns = the seven axes #903 named. Cells are capability statements at
the level the official docs support; each system's primary sources are in §6. fak's cells
carry the honesty tag.

| System | Prefix cache | P/D split | KV transfer | Routing | Autoscaling | Metrics | Invalidation semantics |
|---|---|---|---|---|---|---|---|
| **vLLM** | Automatic Prefix Caching: hash KV blocks by parent-hash + block tokens + LoRA/MM/cache-salt axes | Disaggregated prefill (1P1D / xPyD) via KV-connector, labeled experimental | KV-connector API (LMCache / NIXL / Mooncake backends) | Single-engine scheduler; multi-replica routing via the separate production-stack router | Via production-stack on k8s, not the core engine | Prometheus `/metrics` (incl. prefix-cache hit rate) | LRU block eviction; cache is **performance-only**, no semantic/authorization invalidation |
| **SGLang** | RadixAttention: reusable prefixes in a radix tree; HiCache tiers GPU↔CPU↔disk | First-class PD disaggregation (prefill + decode servers) | Mooncake / NIXL transfer backends for PD KV | `sgl-router`: cache-aware **and** PD-aware request dispatch | Router + k8s replica scaling | Prometheus | LRU radix eviction; performance-only |
| **LMCache** | External KV layer (CPU/DRAM/disk) + non-prefix (CacheBlend) reuse | Supplies the disaggregated-prefill KV path **for** vLLM | NIXL / Mooncake / native; cross-instance KV sharing | KV-aware, composed with the vLLM router | n/a (a KV layer, not a server) | Own observability counters | TTL / LRU; performance-only |
| **Dynamo** | KVBM tiered KV (GPU/CPU/disk/remote) + KV-aware reuse | Disaggregated-serving control plane (prefill/decode workers) | NIXL transfer via KVBM | KV-aware router (route to the worker holding the prefix) | **Planner**: SLA-driven scale up/down | Prometheus + planner metrics | Performance-only; reuse is a placement decision |
| **Mooncake** | KVCache-centric store (distributed KV pool, Kimi) | Architecture **is** KVCache-centric P/D disaggregation | **Transfer Engine** (RDMA / TCP / NVMe-oF) — its core contribution | Conductor / KVCache-aware scheduling | n/a (transport + store, not a full server) | Transfer-engine metrics | Performance-only |
| **fak (current)** | `radixkv` local **exact** prefix `[SHIPPED]`; provider prompt-cache observed + priced `[SHIPPED]`; does not out-perform engine intra-session (scope) | Rides engines; native P/D is `[GAP]` (dual-track S6b, hardware-gated) | `cachemeta` names WHERE/HOW a span shares (`ShareKind`, `BytesMoved`) `[SEAM]`; the byte mover (`StageTransport`, TCP-first) is `[GAP]` | `ReplicaRouter` static N-upstream `[SHIPPED]`; residency/health-aware placement `[GAP]` | `[GAP]` — not a near-term goal | `fak_engine_cache_*` / `fak_cache_*` / `fak_gateway_provider_cache_*`, **source-tagged** `[SHIPPED]` | **The differentiator** `[SHIPPED]` for fak-owned KV: bit-exact mid-run causal eviction + signed `DeletionCertificate` + scope/taint **admission verdict**; degrades to a whole-prefix flush on a ridden engine |

**The one load-bearing tension** (from the throughput/trust spine note): the fastest path to a
parity *number* is to **ride** an engine that owns the KV and exposes only a coarse
whole-prefix reset — which is exactly the move that degrades fak's bit-exact eviction moat on
that span. The resolution is not "pick one": ride engines for the compute number, govern a
fak-serialized span (an L3 tier or fak-owned KV) for the moat, and **never let a ridden-engine
number be reported as a moat claim**. Keeping those claims separable is the deliverable §4 pins.

---

## 2. The unified `CacheEvent` / `ServingEvent` vocabulary

#903's second ladder step asks for "a local `CacheEvent`/`ServingEvent` design note that maps
provider, ride-engine, and native events into the same vocabulary." That vocabulary already
exists as the `internal/cachemeta` **plane** — every cache/serving event, whatever its source,
lowers onto one `cachemeta.Entry` carrying a typed `LookupVerdict`
(`hit/miss/revalidate/transform/quarantine/fault`). The four #903 **sources** map 1:1 onto four
existing planes:

| #903 source | cachemeta plane | Lowering adapter (real symbol) | Metric family |
|---|---|---|---|
| **provider** (OpenAI/Anthropic/Gemini/Bedrock prompt cache) | `PlaneProvider` / `TierProvider` | `cachemeta.FromProviderCache` → `ProviderCacheVerdict` (Transform, `cost_latency_only`) | `fak_gateway_provider_cache_*`, `metrics.Arm.ProviderCache*` |
| **ride-engine** (vLLM/SGLang/LMCache/Dynamo KV routing/offload) | `PlaneKVTransfer` | `engine.CacheEvent` → `cachemeta.FromKVTransfer` → `KVTransferVerdict` | `fak_engine_cache_*` |
| **native-fak** (radix/KV-prefix reuse fak owns) | `PlaneKVPrefix` | `cachemeta.FromKVPrefix` (radixkv) | `fak_cache_*` (unified stream) |
| **vcache / vDSO** (tier-2/3 tool-result *value* cache) | `PlaneToolResult` | `cachemeta.FromVDSOKey` / `FromStaticTool` | `fak_vdso_*`, `fak_cache_*` |

A **`ServingEvent`** (which worker, prefill vs decode, the P→D KV handoff, a router pin) is the
same envelope with `Direction ∈ {offload, restore, route, migrate}` already in
`cachemeta.KVTransferDirection` — `KVRoute` is the router pin, `KVMigrate` the P→D residency
move. So the serving control plane is **not a new vocabulary**; it is the `kv_transfer` plane
with the routing directions, and `KVTransferVerdict` already makes a failed restore a typed
`MISS`/`FAULT` rather than a silent recompute.

**Named gap (the smallest next adapter).** Separability today is carried by the **plane**, not by
a single explicit `Source ∈ {provider, ride_engine, native_fak, vcache_vdso}` enum on the
envelope. A future increment can add that enum (or a `source` label derived from the plane) so a
scrape filters by source directly instead of mapping plane→source by hand. Until then, §4's
regression pins the plane→source mapping so it cannot silently drift.

---

## 3. Which numbers matter (and which lie if blended)

The metrics #903 names split cleanly into *serving latency/goodput* and *cache economics*:

- **Latency / goodput:** TTFT, TPOT, ITL, goodput, KV-util — measured **only** by the parity
  harness (`cmd/paritybench`, dual-track S2), never asserted from the algebra. Until a
  same-hardware run is committed, every tokens/sec statement stays `UNMEASURED`.
- **Cache economics:** prefix-hit rate, cache-write amortization, KV-transfer bytes, cache-fault
  rate, and **cost saved per admitted reuse** — these are honest only when reported **per source**
  (§4). A single blended hit rate is refusal rule 10 in the caching-SOTA note.

---

## 4. Benchmark honesty — four reuse classes, never mixed

The "Done when" gate that a benchmark must **distinguish warm per-agent KV, provider prompt
cache, ride-engine prefix cache, and fak-governed cross-agent reuse without mixing them** is
enforced in code, not convention:

- `cachemeta.IsProviderResidency` / `LocalReuseTokens` / `ProviderReadTokens` **partition** a
  mixed entry stream — a provider entry contributes 0 to any local-reuse total
  (`internal/cachemeta/provider_guard.go`).
- `cachemeta.SavingsSplit.Add` and `metrics.Arm.FoldCacheEntry` route a provider hit to the
  provider counter only; the local counters (`InTokens` / `VDSOHits`) never see it.
- The gateway exports the invariant live: `fak_gateway_provider_cache_local_trust 0`, derived
  from `ProviderCacheEvidence()` (cost/latency-only, never a trust verdict).

[`internal/gateway/serving_event_source_test.go`](../../internal/gateway/serving_event_source_test.go)
(`TestServingEventSourcesAreTagged`) is the source-tag regression this readout adds: it builds
one event per source through its real adapter, asserts the four land on four **distinct** planes,
folds them through the unified `StreamMetrics` and asserts each stays its own row, then folds
them through `metrics.Arm.FoldCacheEntry` and asserts **only** the provider source touches the
provider counter — a provider saving is not a fak win unless the event source says so. GPU-free,
no live engine.

---

## 5. Cross-links — the SOTA-derived decision per implementation epic

This readout feeds the open implementation issues. The decision below is the ride-vs-own split
applied to each; updating the GitHub bodies/children is the implementation follow-on.

| Issue | What it builds | SOTA-derived decision from this matrix |
|---|---|---|
| [#637](https://github.com/anthony-chaudhary/fak/issues/637) | epic: throughput parity + trust/L3 over one shared spine | **Ride** vLLM/SGLang/Dynamo for the P/D + routing number; the **one KV-byte mover** (S3/S4) is the shared artifact both the native-P/D data plane and the L3-governed-span moat consume — keep the two claims separable per §1's tension. |
| [#751](https://github.com/anthony-chaudhary/fak/issues/751) | epic: inbound prompt-MMU, cache-prefix-safe | Mirror vLLM/Anthropic prefix rules: **static prefix before volatile content**, deterministic serialization; the MMU must splice past the last `cache_control` breakpoint so the provider prefix stays byte-identical (the `promptmmu` seam). |
| [#805](https://github.com/anthony-chaudhary/fak/issues/805) | epic: the intent conduit → scheduler & cache placement | This is fak's **own** band: feed kernel intent (scope/taint/priority) into placement as a `ServingEvent` (`KVRoute`/`KVMigrate` on the `kv_transfer` plane), riding SGLang/Dynamo KV-aware routing rather than re-implementing it. |
| [#809](https://github.com/anthony-chaudhary/fak/issues/809) | epic: speculative agent-loop execution (warm next turn) | Promotion-on-match must re-enter `plancfi`/adjudication before any effect — a warmed turn is a **cache candidate, not an execution permit** (caching-SOTA refusal rule 5). |
| [#819](https://github.com/anthony-chaudhary/fak/issues/819) | callavoid: gate vDSO tier-2 with a per-tool-class ProveMemo | The `vcache/vDSO` source (§2) is exactly the plane this gates; the ProveMemo is the per-tool **witness** the caching-SOTA §2.5 parity requirement names (etag/content-hash/git-SHA/lease-epoch). |
| [#870](https://github.com/anthony-chaudhary/fak/issues/870) | feat: complete the GLM-5.2/vLLM agentic battery | GLM-5.2's Coding-Plan-vs-general endpoint and reasoning toggle are **silent provider-cache breakers** — fold them into the provider entry identity (already in `cachemeta.ProviderCache.Endpoint`/`ReasoningMode`) so a mode switch is a distinct write, not an invisible miss. |

---

## 6. Primary sources

Official / primary docs (read as implementation references):

- vLLM Automatic Prefix Caching: <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- vLLM disaggregated prefill / P-D serving: <https://docs.vllm.ai/en/latest/features/disagg_prefill/>
- SGLang RadixAttention / HiCache design: <https://docs.sglang.ai/advanced_features/hicache_design.html>
- SGLang PD disaggregation + router: <https://docs.sglang.ai/advanced_features/pd_disaggregation.html>
- LMCache docs (external KV reuse/offload across vLLM workers): <https://docs.lmcache.ai/>
- NVIDIA Dynamo (planner, router, KVBM, disaggregated serving): <https://docs.nvidia.com/dynamo/latest/>
- Mooncake transfer engine / disaggregated KV transport: <https://github.com/kvcache-ai/Mooncake>

fak's own column (primary source = the tree):

- Source-tag vocabulary: `internal/cachemeta` planes (`provider.go`, `kvtransfer.go`,
  `cachemeta.go` `FromKVPrefix`/`FromVDSOKey`); ride-engine adapter `internal/engine/cacheevents.go`.
- Double-count guard: `internal/cachemeta/provider_guard.go`, `internal/metrics/provider_cache.go`.
- Bit-exact eviction moat: `internal/model/kvcache.go`, `internal/deletioncert`.
- Routing seam: `internal/gateway/replica_router.go`.
- This readout's regression: `internal/gateway/serving_event_source_test.go`.
