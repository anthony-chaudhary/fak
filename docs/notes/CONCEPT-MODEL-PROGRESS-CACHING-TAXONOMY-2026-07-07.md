---
title: "Model-progress caching taxonomy — what state each family produces"
description: "Classifies what cacheable state each model family produces, naming two rows the M1–M10 KV ladder misses: MoE expert residency (M11) and SSM state reuse (M12)."
---

# Model-progress caching taxonomy — what state each family produces, and what of it is cacheable

**2026-07-07.** Companion to the memory-first superset epic **#2236** and its M1–M10 ranking
matrix (`docs/serving/multilevel-default-cache-epic.md`). This note disambiguates the *types and
sub-stages of model progress* and, for each, **what state is produced and whether it is cacheable /
addressable / evictable / reusable** — across dense-transformer, **MoE**, **SSM/linear-attention**,
and **hybrid** families.

**Why it exists.** The M1–M10 ladder is a strong ranking of **KV-cache** concepts — but every one
of its rows silently assumes the dense-transformer state model: *per-position, addressable K/V that
grows linearly with sequence length*. Two families break that assumption, and the ladder has no row
for either:

- **MoE** moves the hot working set off the KV plane entirely — onto **expert weights**. The bytes
  that decide whether a token is fast or slow are *which experts are resident*, not the KV.
- **SSM / linear-attention** replaces per-position KV with a **fixed-size recurrent state** that is
  *not addressable* and *not mid-span-evictable*. M2's longest-prefix-token-match cannot reuse it.

So this note is the R0-conceptual artifact that names those two missing rows (proposed **M11** and
**M12** below) and grounds them in what fak already has in code.

---

## 1. The two axes of "model progress"

**Axis 1 — phase.** A forward pass is one of two regimes, and they cache differently:

| Phase | Regime | Per token | State it produces | fak path |
|---|---|---|---|---|
| **Prefill** | compute-bound, whole-sequence | O(n²) attn over the prompt | writes the *entire* prompt's KV / advances recurrent state to the prompt boundary | cacheless `attnSeq` / `attnPrefillInto`, `linearAttnSeq` |
| **Decode** | memory-bound, one token at a time | O(n) attn over cached state | *appends* one position of KV / *advances* the recurrent state by one step | cached `attnBody` / `attnDecodeBatch` |

Prefix caching is fundamentally *"skip prefill by replaying someone else's decode-time state."* That
only works if the state is **replayable at a boundary** — true for addressable KV, and the crux of
why SSM state (§4c) is the hard case.

**Axis 2 — layer-type heterogeneity.** In a hybrid model a *single* forward pass produces several
*different kinds* of state at once (full-attention KV on some layers, sliding-window KV on others,
recurrent state on the rest). The cache is not one object; it is a **heterogeneous plane** (§4d).

---

## 2. The state-type taxonomy (the matrix)

Rows are *state objects a forward pass produces*. Columns are the properties that decide how each can
be cached. "Addressable" = can you name and reuse an arbitrary sub-span. "Evictable(mid-span)" = can
you drop a span from the *middle* and keep the rest correct (fak's bit-exact quarantine primitive).

| State object | Family | Produced | Grows with seq-len | Addressable | Evictable (mid-span) | Reuse granularity | fak primitive (file) | SOTA row |
|---|---|---|---|---|---|---|---|---|
| **Per-position KV** | dense / GQA | prefill+decode | linear | ✅ per position | ✅ **bit-exact** (pre-RoPE `Kraw`) | token-prefix | `internal/radixkv`, `internal/model/kvcache.go` | M1/M2 |
| **Compressed latent KV (MLA)** | DeepSeek, GLM-5.2 | prefill+decode | linear (small) | ✅ per position | ✅ | token-prefix | `kvlayout.go`, `glm_dsa.go` | M1/M2 |
| **Sliding-window KV (SWA)** | Gemma, MiniMax, GLM | prefill+decode | **bounded** (last W) | ✅ within window | ✅ (evict falls out of window "for free") | window-suffix | `internal/model/swa.go` | M9 |
| **Attention-sink KV** | StreamingLLM-style | prefill | O(1) (first k) | ✅ (pinned) | ❌ (must stay resident) | always-on prefix | `softmaxAttentionScores` `self_attn.sinks` | (M9 adjacent) |
| **Recurrent state + short-conv window** | Mamba/Mamba-2, **GDN** (Qwen3.5/3.6), RWKV, MiniMax | prefill+decode | **fixed** (state_size) | ❌ **lossy-mixed, not per-position** | ❌ **impossible** — typed refusal | **boundary-snapshot only** | `linearAttnCache` (`kvcache.go:17`); `RecurrentEvictUnsupportedError` (`kvcache.go:44`) | **M12 (missing)** |
| **Expert weights (MoE FFN)** | Mixtral, Qwen-MoE, gpt-oss, GLM, MiniMax, DeepSeek-V4 | *static* (loaded), *streamed* on demand | none (weights, not activations) | ✅ per expert | n/a (residency, not eviction) | **residency / prefetch** | `moe_offload.go` (`splitKernel`), `paging_ring.go` (`pagedRing` 3-tier), `expert_parallel.go` | **M11 (missing)** |
| **Router / gating decisions** | MoE | prefill+decode | linear (tiny) | ✅ per token | ✅ | recompute-cheap | `moe.go` (`moeFFN.apply`, `glmRoute`) | M11 (sub-row) |
| **RoPE tables / compile artifacts** | all | once | none | n/a | n/a | process-lifetime | `cachedInvFreq`, CUDA-graph/compile cache (#3052) | (static) |

---

## 3. Reading the matrix by phase

- **Prefill** produces the bytes that prefix caching wants to *skip next time*: the whole prompt's
  KV (addressable → M2 reuse works) **or** the recurrent state advanced to the prompt boundary
  (fixed-size → needs M12 boundary-snapshot, which does not exist).
- **Decode** produces one KV position (append) **or** one state advance. For MoE, decode is dominated
  by *expert-weight movement* — every token may touch a different top-k of experts, so the cache that
  matters is M11 residency, not the KV plane at all.

---

## 4. Per-family reading — where the ladder holds and where it breaks

### 4a. Dense transformer — the ladder's home ground
KV is the whole story. M1 (paged) · M2 (radix prefix) · M3 (tiering) · M5 (evict) · M8 (non-prefix
reuse) all apply directly. fak's differentiators here are real and already in code: **pre-RoPE `Kraw`
+ absolute positions** make a poisoned span *bit-exact evictable and re-RoPE-able* (M2/M8 lead — no
external engine offers `SupportsExactSpan=true`).

### 4b. MoE — the cache pressure moves KV → expert weights  ⟶ **Gap A / proposed M11**
Attention on a MoE model is still dense/GQA/MLA, so KV caching is unchanged. But the **dominant
resident bytes are the experts**: only top-k of E experts fire per token, so on any host where all
experts don't fit in VRAM, the question "is this token fast?" becomes "were its experts resident?"
That is a **weight cache** with residency/prefetch semantics — a memory concept the M1–M10 ladder
**does not rank at all**.

**What is actually cacheable in a MoE forward (and what isn't):**

**A — Expert weights (the bytes; the real working set).**
- **A1. Routed-expert hot-set residency** — expert activation is *skewed*, so the hot set is an
  LRU/LFU-by-activation-frequency cache over tiered GPU / pinned-CPU / SSD. This is M11 core.
  fak's pieces: `moe_offload.go` `splitKernel` (`--n-cpu-moe`, all-experts-host-resident-forever =
  the *floor*, not a cache); `paging_ring.go` `pagedRing`, a bounded per-weight VRAM ring (LRU,
  pinned-exempt), explicitly the **Tier-1 GPU ring of a 3-tier GPU/pinned-CPU/SSD expert cache**
  (#2726, epic #2722) — but off the live serve path, f32-only, unwired.
- **A2. Shared-expert pinning** — DeepSeek/GLM carry *always-on* shared experts (`glmSharedExperts`)
  that fire on **every** token, exactly like the router/attention the split keeps on-device. Yet
  `isExpertWeight` (`moe_offload.go:91`) offloads `.mlp.shared_experts.` to host *identically to
  routed experts*, despite the file's own "experts are sparse" rationale (`moe_offload.go:15`). In a
  residency cache, shared experts are the **highest value-per-byte** entries (hit rate = 1.0) and
  must rank always-hot, not compete with cold routed experts. Distinct policy → filed as a leaf.
- **A3. Weight-quant residency ladder** — cache a low-precision (Q4) copy hot on GPU and keep a
  higher-precision copy cold (host/SSD): the *weight-side* analogue of M7's KV-quant ladder.

**B — Routing metadata (the trace; cheap to hold, high leverage).**
- **B1. Per-prefix top-k selection trace** — the experts each prefix position demanded, cached
  *alongside* the KV prefix (tiny vs. the KV itself).
- **B2. Activation-aware prefetch — the fak-unique composition.** fak *already* reuses KV prefixes
  (`radixkv`, vCache). Pairing each cached prefix with its routing trace means a KV-prefix hit tells
  you **exactly which experts the resumed decode will need**, so the expert cache can warm *in
  lockstep* — prefix reuse *drives* expert prefetch. No engine couples its prefix cache to expert
  warming this way; it falls straight out of fak's owned prefix plane. (SOTA prefetch *without* the
  prefix coupling: MoE-Infinity, Pre-gated MoE, SiDA.)
- **B3. Per-session expert load histogram** → EPLB-style placement + **router-affinity routing**
  (send a request to the worker whose resident hot experts match its predicted top-k — the MoE
  analogue of M6 KV-aware routing).

**C — NOT worth caching (honesty fence).** The router logits / gate output are a tiny on-device
matmul — recompute beats caching (which is *why* `moe_offload.go` keeps the router on device); only
the *selection* as metadata (B) has value. Expert FFN **outputs** for prefix positions are already
captured by residual-stream / KV prefix reuse — caching them again is double-counting. The
token→expert permutation/scatter index is recompute-cheap.

**D — Attention is unchanged.** A MoE model's KV plane (dense/GQA/MLA) caches exactly as a dense
model's (M1–M10). So the total MoE cache = **KV plane (M1–M10) ⊕ expert-weight plane (A) ⊕ routing
trace (B)**. Only A and B are new; they are M11. Complementary but *not* a cache axis:
`expert_parallel.go` + `capacity.go:228` EP sharding is a *placement* decision, orthogonal to residency.

**SOTA to rank against:** DeepSeek **EPLB** (expert-parallel load balancing) · SGLang **EP/EPLB** ·
**Fiddler** (CPU-GPU expert orchestration) · **MoE-Infinity** / **Pre-gated MoE** / **SiDA**
(activation-aware expert prefetch) · **ktransformers** (GPU/CPU expert placement) · **ssd-llm**
(SSD→unified-memory streaming, cited in #2726) · vLLM **v0.21 HMA**.

### 4c. SSM / linear-attention — fixed-size, non-addressable state  ⟶ **Gap B / proposed M12**
The GDN hybrid (Qwen3.5/3.6, `qwen35.go` `IsQwen35Hybrid:26`, `linearAttnSeq`) and MiniMax keep a
**fixed-size recurrent state + short-conv window per layer** in `linearAttnCache` — *not* per-token
K/V. Three consequences the KV ladder cannot express:

1. **Constant memory** regardless of context length (the opposite of M1/M3's linear-growth premise) —
   an *advantage* SSM gives for free, which the ladder gives fak no row to claim.
2. **Not mid-span-evictable.** The state is a lossy mix of all past tokens; you cannot drop a poisoned
   span from the middle and keep the rest. fak already encodes this correctly as a **typed refusal**:
   `RecurrentEvictUnsupportedError` (`kvcache.go:44`); `KVCache.Evict` panics it, `CanEvict`/`TryEvict`
   surface it as a verdict, and the KV-MMU eviction loop must consult `CanEvict` and *skip*
   (`kvmmu/hybrid_evict_test.go`, regression #1704). **This honesty — refusing what it cannot do
   bit-exact — is itself a fak lead** over engines that would silently corrupt.
3. **No prefix reuse today.** Because state is only meaningful *as a whole at a boundary*, M2's
   longest-prefix-token-match can't touch it. The reuse primitive SSM needs is **snapshot the state at
   a shared-prefix boundary, key it, and fork it on reuse** — a *positive* mechanism fak does not have.
   `StageSpan`/`RestoreSpan` exist on the `abi.KVBackend` interface but for hybrid caches resolve to
   skip-and-keep-resident. **Net: hybrid/SSM models get ZERO prefix-cache benefit on fak today.** That
   is the gap M12 closes.

**SOTA:** vLLM **Hybrid Memory Allocator** (v0.21, state + offload integration) · SGLang **UnifiedTree
HiCache** (SWA/Mamba by default) · Mamba/Mamba-2, Gated-DeltaNet, `mamba_ssm` state semantics.
Note: mid-conversation *reuse* of recurrent state is near research-frontier even upstream (comparable
to M8 CacheBlend's status), so M12 is a place fak can lead rather than catch up.

### 4d. Hybrid — one request, three state kinds at once
A single Qwen3.6 / MiniMax / (future) Jamba/Nemotron-H/Falcon-H1 forward holds **{full-attn KV,
SWA-windowed KV, recurrent state}** simultaneously, each with different growth/evict/reuse. **M9 /
#2241** tracks the *allocation* side (allocator groups so radix/tiering become window- and
state-aware). M12 tracks the orthogonal *reuse* side. Both are needed for a hybrid model to get the
prefix-cache economics a dense model gets today.

---

## 5. fak's unique value-add (honest) vs. what it isn't caching well yet

**Leads (in code, witnessed):**
- Bit-exact **span-evictable / re-RoPE-able KV** (`Kraw`) — no engine offers exact-span (M2/M8).
- **Typed refusal** of recurrent mid-span eviction (`RecurrentEvictUnsupportedError`) — correctness
  over false capability.
- **Quarantine eviction** in radixkv (`EvictNode`/`EvictPrefix`) — trust-plane primitive no engine has.
- A named **3-tier expert-cache design** already prototyped (`pagedRing`).

**Not caching well yet (the ticket list):**
- **M11** — no live expert-weight cache on the serve path; `pagedRing` is off-path/f32-only. MoE hosts
  fall back to all-host-resident (`splitKernel`) or don't fit. *(→ ticket)*
- **M12** — no recurrent-state prefix reuse; hybrid/SSM models get no prefix-cache benefit. *(→ ticket)*
- **M7** — no KV-side quantization (weights-side only).
- **M8** — non-prefix KV reuse (CacheBlend) is dead code awaiting wire-or-retire (#1347), despite
  `Kraw` being a structural advantage for it.
- Spec-decode draft/EAGLE state (`internal/spec`) exists but isn't wired into the batch loop (#23).

---

## 6. SOTA map (references)
- **KV plane:** vLLM PagedAttention/APC/HMA · SGLang RadixAttention/HiCache/UnifiedTree · Dynamo
  KVBM/NIXL · Mooncake · LMCache (CacheGen/CacheBlend) · llm-d (see #2236 source list).
- **MoE expert cache/placement:** DeepSeek EPLB · SGLang EP/EPLB · Fiddler · MoE-Infinity ·
  ktransformers · ssd-llm · vLLM v0.21 HMA.
- **SSM/hybrid:** vLLM Hybrid Memory Allocator · SGLang UnifiedTree HiCache · Mamba/Mamba-2 · GDN.

## 7. Related fak issues
Epic **#2236** (memory superset) · **#2237** (dossier) · **#2241 / M9** (hybrid memory *plane*) ·
**#2722 / #2726** (expert paging primitive) · **#447** (GDN arch) · **#487 / #934**
(`docs/notes/model-arch-seam-status-487.md`) · **#1704** (hybrid-evict skip) · **#2869** (DS4 mining).

> **Proposed matrix additions** (for the maintainer to fold into `multilevel-default-cache-epic.md`):
> **M11 — MoE expert-weight residency/streaming cache** (#3174) and **M12 — recurrent-state prefix
> reuse (SSM/GDN state checkpoint + handoff)** (#3175). Both filed as children of #2236.
