---
title: "Awesome Caching — every caching concept fak knows, each in its own words"
description: "An awesome-list index of caching concepts for LLM and agent workloads: fak's own kernel-owned KV/prefix/managed-cache primitives (each linked to its self-describing doc or package), the M1–M12 KV concept ladder, model-family caching (dense/MoE/SSM/hybrid), multi-tier serving, and the external engines fak observes. Every entry links to the concept's own self-description; fak status is tagged shipped / partial / plan / lead."
---

# Awesome Caching

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A single index of every **caching concept** fak knows about — for LLM inference and
> agent workloads. Each entry links to the concept's **own self-description** (the doc
> or package where it describes itself, in its own words) so this page stays a thin,
> honest index rather than a second source of truth.
>
> Scoped to this repository for now (an in-repo awesome list following
> [awesome](https://awesome.re) conventions). The plan for making it a public,
> answer-engine-citable list — and for telling each external project *"we added you to
> our awesome caching list"* — lives in [`outreach.md`](outreach.md).

<!-- awesome-caching: maintained index. Follows awesome-list conventions; scoped in-repo. -->

Companion index: [**Awesome Token Efficiency**](../awesome-token-efficiency.md) is the
broader catalog of *token / context / KV-cache efficiency methods* (many not caching).
This list is the **caching-only** slice, and unlike that survey it leads with fak's
**own** primitives and links each to where it lives in the tree.

## How to read this

Every entry is one line: **name → self-description → link(s) → tags.**

- The **link** points at the concept's own words: an in-repo doc's frontmatter, a Go
  package's doc-comment, or (for external systems) the upstream project plus fak's
  in-repo analysis of it.
- **fak status** tags: ✅ **shipped** (in code, witnessed) · 🟡 **partial / seam
  present** · 🔭 **plan or design doc only** · ⭐ **a place fak does something no
  shipped engine does** · 👁 **observe-only** (fak watches an external engine's cache,
  does not control it).

New concept? Add one line in the right section — see [`contributing.md`](contributing.md).

## Contents

- [fak at a glance](#fak-at-a-glance)
- [Kernel-owned cache primitives](#kernel-owned-cache-primitives) — the objects fak *owns*
- [Managed cache & session economics](#managed-cache--session-economics) — steering the provider's prompt cache
- [vCache — a virtual API cache over providers](#vcache--a-virtual-api-cache-over-providers)
- [The KV-cache concept ladder (M1–M12)](#the-kv-cache-concept-ladder-m1m12)
- [Model-family caching (dense / MoE / SSM / hybrid)](#model-family-caching-dense--moe--ssm--hybrid)
- [Multi-tier serving & KV transport](#multi-tier-serving--kv-transport)
- [Explainers (plain-English)](#explainers-plain-english)
- [Proofs (bit-exact correctness)](#proofs-bit-exact-correctness)
- [External engines & systems (the field)](#external-engines--systems-the-field)
- [Research frontier notes](#research-frontier-notes)
- [Contributing](#contributing)
- [AEO & outreach](#aeo--outreach)

---

## fak at a glance

fak is an **agent kernel**: a gateway you put in front of the model that keeps the
provider's prompt-cache prefix byte-identical while shedding old turns, and (in the
fused path) runs the model with an **addressable, bit-exact KV cache it owns as a kernel
object**. The one thing no shipped engine does — remove a tool result from the *middle*
of a kept sequence, bit-identically — falls out of that ownership. This list starts
there and fans out to the whole field.

---

## Kernel-owned cache primitives

The cache objects fak *owns* (not just observes). This is where fak's leads live.

- **[Addressable / bit-exact KV cache](../explainers/addressable-kv-cache.md)** — Every
  production prefix cache is append-only and prefix-addressed: a change at position N
  costs everything after N. fak owns its KV cache as a kernel object, which lets it
  **remove a span from the MIDDLE of a kept sequence, bit-identically to never having
  seen it** — the underdiscussed half of "addressable." ⭐ · proof: [`model-kv`](../proofs/model-kv.md)
- **[RadixKV prefix cache](../../internal/radixkv)** — SGLang's RadixAttention
  (arXiv:2312.07104) rebuilt over fak's kernel-owned KV cache: longest-prefix reuse by
  a radix-tree walk, reference counts that net to zero, and **LRU leaf eviction with
  upward collapse**. The apples-to-apples answer to "how does fak compare to SGLang's
  KV-cache radix attention?" ✅ · proof: [`radixkv`](../proofs/radixkv.md)
- **[kvmmu — mechanical quarantine](../../internal/kvmmu)** — Turns ctxmmu's *logical*
  quarantine verdict ("these bytes may not enter the context window") into a
  *mechanical* one: **eviction of that result's K/V span from the kernel-owned attention
  cache, so the model physically cannot attend to it.** The deepest expression of the
  kernel thesis. ⭐ · proof: [`kvmmu`](../proofs/kvmmu.md)
- **[cachemeta — the cache-entry contract](../../internal/cachemeta)** — The metadata
  contract for first-class cache entries. Stores **no payloads and owns no cache**; only
  names reusable objects, their validity/security/residency metadata, and typed lookup
  verdicts callers can fold without collapsing every non-hit into "false." Foundation
  tier. ✅ · proof: [`cachemeta`](../proofs/cachemeta.md)
- **[l3kv — durable disk-backed L3](../../internal/l3kv)** — Durable disk-backed L3 KV
  residency backend: `StageSpan`/`RestoreSpan` persist a demoted span to blobfs by
  digest (#1472). 🟡
- **[enginecache — remote invalidation binding](../../internal/enginecache)** — Binds
  cachemeta's remote invalidation directives to documented serving-engine control
  endpoints; records `SupportsExactSpan=false` for engines that only expose whole-prefix
  reset, so a miss safely re-sends full context. ✅
- **[vdso — the tool vDSO](../../internal/vdso)** — A 3-tier local fast path that answers
  a tool call with **no engine and no remote round-trip** (the agentic analogue of the
  kernel vDSO serving `gettimeofday()` from userspace). Tier 2 is a content cache keyed
  on `(tool, args-sha256, world-version)`, invalidated on a world bump. ✅
- **[promptmmu — inbound tool pruning](../../internal/promptmmu)** — The ingress dual of
  ctxmmu: drops provably-unreachable tool *definitions* from the advertised `tools[]` so
  fewer uncached bytes go upstream, byte-preserving the cache prefix. Pure upside — the
  model loses nothing it could have done. ✅
- **[syspromptmmu — base-context plan](../../internal/syspromptmmu)** — Authors fak's
  irreducible spine (the gate, the journal, what a capability is) as an immutable,
  version-stamped cacheable tier — a *plan* of ordered prompt segments, each with a
  content-derived witness, never wire bytes. 🔭

---

## Managed cache & session economics

Steering the provider's prompt cache (Anthropic/OpenAI) — the levers that keep a long
session's discount alive without owning the engine.

- **[What is fak's managed cache?](../explainers/what-is-managed-cache.md)** — fak
  managed cache steers Anthropic's prompt cache so an **idle session re-enters on a cheap
  read instead of re-paying**. On by default everywhere (2026-07-10); on Pro/Max the
  payoff is usage-limit headroom, not a smaller bill. ✅
- **[Compact-history + coherence](../../internal/compactcohere)** — The byte-splice
  history compactor sheds the un-cacheable middle past a token budget while keeping the
  provider prompt-cache prefix **verbatim**; `compactcohere` is the policy that keeps
  fak's kernel compaction and the harness's own auto-compaction from fighting each
  other. ✅ · explainer: [long sessions, keep the cache hit](../explainers/long-sessions-keep-the-cache-hit.md)
- **[cacheprice — one price source of truth](../../internal/cacheprice)** — The single
  source of truth for provider prompt-cache read/write multipliers (relative to a base
  input token), so every layer that prices cache economics reads the same constants
  instead of re-declaring `0.1 / 1.25 / 2.0`. ✅
- **[cachevalueledger — realized-reuse ledger](../../internal/cachevalueledger)** — A
  durable, append-only ledger of per-session cache-value observations (turns,
  prompt/reused tokens, reuse ratio) scored by `fak nightrun score` to catch cache-value
  regressions; reports INSUFFICIENT rather than failing on a thin corpus. ✅ ·
  rollup: [cache-value-rollup](../cache-value-rollup.md)
- **[cachewitness — provenance-split evidence](../../internal/cachewitness)** — Reads a
  live gateway's `/metrics` and folds the in-kernel KV-prefix cache family into **one
  provenance-labeled record**, splitting what fak *controls* from what it only
  *observes*. ✅

---

## vCache — a virtual API cache over providers

A cache we build **on top of** providers whose engine we don't control, by steering
request order, shape, and timing onto their own prefix caching.

- **[vCache design](../notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md)** — The design for a
  virtual API cache that maps our caching units onto external providers' own prefix
  caching by controlling request order, shape, and timing. 🔭
- **[vcachescore — the scorecard](../../internal/vcachescore)** — Folds the vCache proof
  leaves into one operator artifact: *"is this workload at least 2× better, what index
  should the agent build, and what action moves it closer?"* Pure off-path; treats cache
  hits as rebates only. ✅
- **[vcachegov — the governor](../../internal/vcachegov)** — The steady-state policy
  layer deciding, per cacheable prefix, whether to heartbeat-pin / lazy-rebuild / ride
  traffic / evict; how many prefixes to warm inside rate-limit headroom; and how to route
  chained requests onto a consistent warm shard. ✅
- **[vcachewarm — warming decisions](../../internal/vcachewarm)** — The dedicated-warming
  decision layer: which warming primitive a caller may spend, where an Anthropic explicit
  breakpoint belongs, when a fanout barrier may release dependents. Issues no network
  calls. ✅
- **[vcacheobserve — per-sub-concept lens](../../internal/vcacheobserve)** — The
  observability surface behind `fak vcache observe`: groups a run's turns by prefix family
  and runs the shipped decision leaves over **real account traffic** vs the scorecard's
  synthetic defaults. ✅ · playbook: [vcache-scorecard-playbook](../serving/vcache-scorecard-playbook.md)

---

## The KV-cache concept ladder (M1–M12)

The ranked ladder of KV-cache concepts fak measures itself against, plus the two rows a
KV-only ladder misses.

- **[Multilevel default cache epic (M1–M10 spine)](../serving/multilevel-default-cache-epic.md)**
  — The progress spine finishing the hardware-capacity bridge: wire demote-not-evict
  into a live loop, derive real pressure for every local tier, and make hardware-aware
  placement the kernel's default. Each rung is a prove-or-refute step bound to a `dos`
  verb. 🔭
- **[Model-progress caching taxonomy](../notes/CONCEPT-MODEL-PROGRESS-CACHING-TAXONOMY-2026-07-07.md)**
  — Classifies what cacheable state each model family produces (per-position KV,
  compressed latent KV, sliding-window KV, attention-sink KV, recurrent state, expert
  weights, routing metadata) and names two rows the M1–M10 KV ladder misses. Reference.
- **M11 — MoE expert-weight residency** — The hot working set of a MoE model is *which
  experts are resident*, not the KV. A weight cache with residency/prefetch semantics the
  KV ladder does not rank; fak's `pagedRing` prototypes a 3-tier GPU/pinned-CPU/SSD
  expert cache (off-path today). 🔭 (see taxonomy §4b)
- **M12 — recurrent-state prefix reuse** — SSM / linear-attention state is fixed-size and
  non-addressable, so longest-prefix-token-match can't touch it; the reuse primitive it
  needs is *snapshot-at-boundary, key, fork-on-reuse*. A place fak can lead. 🔭 (see
  taxonomy §4c)

## Model-family caching (dense / MoE / SSM / hybrid)

- **Dense / GQA — per-position KV** — the ladder's home ground; fak's pre-RoPE `Kraw` +
  absolute positions make a span **bit-exact evictable and re-RoPE-able**. ⭐
- **MoE — expert-weight residency** — cache pressure moves KV → expert weights; shared
  experts (hit-rate 1.0) are the highest value-per-byte entries. 🔭 (M11)
- **SSM / linear-attention — recurrent state** — constant memory, **not mid-span
  evictable**; fak encodes this honestly as a typed refusal (`RecurrentEvictUnsupportedError`)
  rather than silently corrupting. ⭐ (correctness lead)
- **Hybrid — three state kinds at once** — one forward pass holds {full-attn KV,
  SWA-windowed KV, recurrent state}, each with different growth/evict/reuse. 🔭
  All four rows: [model-progress caching taxonomy](../notes/CONCEPT-MODEL-PROGRESS-CACHING-TAXONOMY-2026-07-07.md).

---

## Multi-tier serving & KV transport

- **[Hardware-aware KV cache](../serving/hardware-aware-cache.md)** — fak's cachemeta
  plane plans where a KV span lives across HBM, DRAM, NUMA-far, CXL, disk, and remote
  tiers, with per-tier TTL and **demote-not-evict** placement. 🔭
- **[Regenerable KV](../serving/regenerable-kv-plan.md)** — Treats the KV cache as a
  regenerable build artifact rebuilt from durable transcript text, so a model rollout
  becomes a *backfill*, not a cold start. 🔭
- **[KV-transport governance (NIXL / Mooncake / LMCache)](../serving/kv-transport-governance-nixl-mooncake-lmcache.md)**
  — The governance contract for external KV-transport systems to report P/D disaggregation
  and KV-transfer events to fak for observability and trust. 🔭
- **[P/D disaggregation + KV routing SOTA matrix](../serving/pd-disaggregation-kv-routing-sota.md)**
  — A one-page ride-vs-own matrix across vLLM, SGLang, LMCache, Dynamo, Mooncake, and fak
  over prefix cache, P/D split, KV transfer, routing, autoscaling, metrics, invalidation.
  Reference.
- **[External L3 disaggregated cache, re-imagined](../notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md)**
  — Maps 8 external L3 KV-cache gaps onto shipped fak primitives that make fak the
  *semantics layer* composing with the cache. 🔭
- **[Cache-frontier operating plan](../CACHE-FRONTIER-OPERATING-PLAN.md)** — The operating
  plan for the cache-frontier workstream. Reference.

---

## Explainers (plain-English)

Self-contained, answer-engine-friendly explanations — ideal AEO targets.

- **[Addressable KV cache, in 5 minutes](../explainers/addressable-kv-cache-in-5-min.md)** — the short version of the mid-span-eviction idea.
- **[How the KV cache changes as agentic context grows](../explainers/kv-cache-agentic-context.md)** — appending tool output is the easy case; hit rate erodes from latency-eviction and head-mutation, and it matters more for agents than chat because of the input:output ratio.
- **[The frozen-trajectory cache cliff](../explainers/frozen-trajectory-cache-cliff.md)** — the high prompt-cache hit rate everyone quotes is purchased with a frozen, append-only trajectory; the scaling laws that take it to 0%.
- **[Long sessions: shed history, keep the cache hit](../explainers/long-sessions-keep-the-cache-hit.md)** — how fak sheds old turns while keeping the provider's prompt-cache prefix byte-identical.
- **[The vLLM lifecycle cache-loss bridge](../explainers/vllm-lifecycle-cache-loss-bridge.md)** — bridging session dormancy to an upstream vLLM worker's sleep/pause/wake with explicit evidence of what happened to the KV cache.

---

## Proofs (bit-exact correctness)

The correctness claims behind the primitives above, each a runnable proof.

- **[RadixKV prefix reuse & LRU eviction](../proofs/radixkv.md)** — longest-prefix reuse equals recompute, reference counts net to zero, LRU leaf eviction with upward collapse.
- **[kvmmu KV-span bijection](../proofs/kvmmu.md)** — logical positions stay one-to-one with physical cache slots under append and evict; each named span addressed exactly.
- **[Model KV — slots, eviction, SWA](../proofs/model-kv.md)** — correct slot placement, span-exact eviction, sliding-window masking, prefix-reuse parity.
- **[cachemeta binding-key determinism](../proofs/cachemeta.md)** — the payload-free key is a deterministic, injective fold over its binding axes, so distinct bindings never alias one entry.

---

## External engines & systems (the field)

The engines and systems fak observes, benchmarks, or borrows from. These are the
**outreach targets** — the projects a *"we added you to our awesome caching list"* note
would go to (see [`outreach.md`](outreach.md)). Each entry pairs the upstream project
with fak's in-repo analysis and fak's **proven-today** verdict from the
[external-engine cache capability inventory](../cache-frontier/external-engine-cache-capability-inventory.md).

- **[vLLM](https://github.com/vllm-project/vllm)** — PagedAttention prefix caching + a KV-event stream (`BlockStored`/`BlockRemoved`); fak observes paged-block residency but its only control is a whole-prefix reset. 👁 `passive observe`. M2-lens study-repo pass (5 PRESENT/tracked · 2 ABSENT · 17 PARTIAL) → [study note](../notes/CONCEPT-STUDY-VLLM-M2-2026-07-10.md), leaves #3893–#3897.
- **[SGLang](https://github.com/sgl-project/sglang)** — RadixAttention automatic prefix reuse; fak observes `cached_tokens` + radix-residency but its only control is a whole-radix `flush_cache`. 👁 `passive observe` · fak's own rebuild: [`radixkv`](../../internal/radixkv).
- **[LMCache](https://github.com/LMCache/LMCache)** — KV-cache management (CacheGen compression, CacheBlend non-prefix reuse); study-repo pass mapped 9 PRESENT / 28 PARTIAL onto fak. → [study note](../notes/CONCEPT-STUDY-LMCACHE-2026-07-08.md).
- **[Mooncake](https://github.com/kvcache-ai/Mooncake)** — KVCache-centric disaggregated serving + KV transfer; a row in fak's [transport governance](../serving/kv-transport-governance-nixl-mooncake-lmcache.md) and [P/D matrix](../serving/pd-disaggregation-kv-routing-sota.md).
- **[NVIDIA Dynamo (KVBM / NIXL)](https://github.com/ai-dynamo/dynamo)** — KV Block Manager + the NIXL inference transfer library; an external KV-transport system fak governs for observability.
- **[llm-d](https://github.com/llm-d/llm-d)** — Kubernetes-native distributed inference with KV-aware routing; a source in the memory-superset epic's SOTA map.
- **[DeepSeek EPLB](https://github.com/deepseek-ai/EPLB)** — expert-parallel load balancing; the SOTA to rank M11 (MoE expert residency/placement) against.
- **[ktransformers](https://github.com/kvcache-ai/ktransformers)** — GPU/CPU expert placement for MoE; SOTA for activation-aware expert orchestration (M11).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — pluggable on-device runtime and a prefill baseline; no in-tree cache observation adapter yet. 👁 `unknown` (honest absence).
- **[Ollama](https://github.com/ollama/ollama)** — local daemon runtime + model-pull alias; no in-tree cache observation adapter yet. 👁 `unknown`.
- **LM Studio** — named only by the default-enablement backlog; no in-tree adapter. 👁 `unknown` (unknown by honest absence of evidence, not a guessed capability).

---

## Research frontier notes

Dated triage/study notes that fed the concepts above (kept for provenance; may age):

- [Agentic caching SOTA (2026-06-19)](../notes/AGENTIC-CACHING-SOTA-2026-06-19.md)
- [Cache-hit as a vanity metric (2026-07-01)](../notes/CACHE-HIT-VANITY-METRIC-SELF-FULFILLING-2026-07-01.md) — why a high hit rate can be self-fulfilling and what to measure instead.
- [Study: LMCache → fak](../notes/CONCEPT-STUDY-LMCACHE-2026-07-08.md) · [Study: MinIO memkv](../notes/CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md)
- [Study: vLLM (M2 lens) → fak](../notes/CONCEPT-STUDY-VLLM-M2-2026-07-10.md) — admission pricing, almost-hit, self-inflicted-eviction attribution, partial-hit recompute (5 leaves #3893–#3897).
- [Harness cache-coherence audit](../notes/HARNESS-CACHE-COHERENCE-AUDIT-2026-06-28.md)
- [Session cache-savings ablation](../notes/SESSION-CACHE-SAVINGS-ABLATION-2026-06-29.md)
- Full set: `docs/notes/*CACHE*`, `docs/notes/RESEARCH-kv-*`, `docs/notes/*KV*`.

---

## Contributing

Adding a caching concept is one line in the right section above. The rule: **link to the
concept's own self-description, don't restate it here.** Full convention (including how to
pick a section and the tag legend) is in [`contributing.md`](contributing.md).

## AEO & outreach

How this list is meant to earn its keep — as answer-engine-citable content (AEO) and as a
friendly *"we added you to our awesome caching list"* growth loop to the external projects
above — is drafted in [`outreach.md`](outreach.md). **No external comment is posted from
this repo without explicit human approval** (posting to another project is an
outward-facing action); `outreach.md` holds the templates, the target list, and the
guardrails, not an executed campaign.

---

<sub>An in-repo awesome list. Follows [awesome](https://awesome.re) list conventions,
scoped to this repository. Companion: [Awesome Token Efficiency](../awesome-token-efficiency.md).</sub>
