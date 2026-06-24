---
title: "fak note: agentic caching as a kernel verdict (2026-06-19)"
description: "Maps 2026 SOTA agent-cache layers and argues a fak cache hit is an authorization, state, and coherence verdict, not a boolean reuse event."
---

# Agentic Caching as a First-Class FAK Primitive

> Date: 2026-06-19.
> Scope: current SOTA parity map plus the local `fak`/`fleet` wedge.
> Status: design and positioning memo. Some substrate is shipped; the unified
> agentic-cache plane is not yet complete.

---

## 0. Verdict

The field is no longer asking whether agents need caching. It now has named
layers for prompt-prefix caching, KV-prefix reuse, KV offload, non-prefix segment
reuse, shared/compressed KV pools, tool-result caches, stateful tool-value caches,
semantic intent caches, and plan-template caches.

The first-class local thing is therefore not "we cache." That is table stakes.

The local claim should be:

> **An agentic cache hit is an authorization, state, and coherence verdict over
> a running program, not a boolean reuse event.**

> **Reconciliation with the earlier memo.** This broadens, and does not narrow,
> the authorization-only headline of the first-class cache memo (2026-06-18: "A
> cache hit is an authorization claim, not just a reuse event"). _State_ and
> _coherence_ are added axes a reusable object must still prove, not a relaxation
> of authorization. The one-axis headline is a strict subset, so the older claim
> stays true inside the newer one; the agentic setting only forces the other two
> axes onto equal footing, since a byte-identical span can still be stale or
> incoherent.

For chat serving, a cache hit often means "same prompt prefix" or "similar
query." For an agent, a reusable object may be a KV span, a tool result, a
memory view, a plan template, a policy verdict, a provider-resident prefix, or a
sandbox-derived value. Serving it is legal only when the cache can still prove:

1. **Identity** - same bytes, tokens, model, tokenizer, adapter, position mode,
   prompt serializer, tool schema, or plan schema.
2. **State precondition** - same relevant world, sandbox, tool-call history,
   approved plan state, or source digest.
3. **Freshness** - a validator or witness has not been refuted.
4. **Authority** - the requesting agent/session/tenant can see the entry.
5. **Integrity** - the entry passed an admission gate stronger than model
   self-report.
6. **Causality** - downstream consumers are known enough to invalidate or taint
   them if the entry is refuted.
7. **Economics and quality** - writes, transfers, recompute, and approximate
   reuse faults are measured against task quality, not just hit rate.

That is the wedge. SOTA systems optimize individual cache layers. `fak` can make
cache reuse a kernel verdict across layers because it already owns the syscall,
result admission, taint/scope labels, scoped invalidation, recall image,
radix/KV metadata, and witness discipline.

---

## 1. Definition

An **agentic cache entry** is any reusable intermediate produced while an agentic
program reasons, calls tools, reads memory, mutates state, or forks work:

| Entry kind | Examples | Key extra risk beyond normal caching |
|---|---|---|
| Prompt prefix | stable system/tool prefix, provider cache handle | head mutation, provider TTL, remote telemetry mistaken for trust |
| KV prefix/span | radix node, vLLM block, LMCache object, hosted KV artifact | model/tokenizer/position binding, quality drift for approximate reuse |
| Tool result | search/read/API result, pure computation, static table row | stale mutable world, side effects, wrong TTL, args-only key |
| Stateful tool value | terminal, SQL, browser, code sandbox result | output depends on prior tool history and environment state |
| Semantic/intent key | paraphrase cluster, functionally equivalent task key | false positives execute the wrong cached answer |
| Plan template | recurring workflow decomposition or tool graph | stale or unsafe plan applied to a new state |
| Memory view | summary, facts, QA, graph, timeline, retrieved page | view can omit, launder, or stale-transform source evidence |
| Policy/adjudication | parsed tool call, refusal, capability decision | policy version and reason vocabulary drift |

The unifying API should not ask "is it cached?" It should ask:

```text
Lookup(request, agent, world, policy) -> HIT | MISS | REVALIDATE |
  TRANSFORM | QUARANTINE | FAULT
```

The existing `fak/internal/cachemeta.LookupVerdict` is the right shape for this.
The missing work is to make more planes emit and consume those verdicts.

---

## 2. SOTA Parity Map

### 2.1 Provider prompt-prefix caching

**SOTA baseline.** OpenAI automatically caches exact prompt prefixes for recent
models and recommends stable content first, variable content last, with exact
matches required. Anthropic supports automatic and explicit cache breakpoints,
with 5-minute default lifetime and 1-hour extended TTL. Gemini has implicit
cache hits on 2.5+ models and explicit context caches with TTL. Bedrock exposes
prompt cache checkpoints. "Don't Break the Cache" measures this on long-horizon
agent tasks and finds large cost/TTFT wins, but also shows that naive full-context
caching can increase latency when writes are paid and not reused.

**Parity requirement.**

- deterministic prompt/tool serialization;
- static prefix before volatile content;
- explicit cache-breakpoint strategy where the provider exposes it;
- per-provider telemetry for cached/read/write tokens and first divergence;
- no trust claim from provider cache hits.

**Local state.** `docs/explainers/kv-cache-agentic-context.md` already captures
the prefix mechanics and agent-specific failure modes. The first-class cache memo
already names provider-resident cache as a plane.

**Gap.** Provider cache telemetry is not folded into `cachemeta`/metrics yet, so
local benchmarks can accidentally count provider savings as local wins.

### 2.2 KV prefix, offload, and routing

**SOTA baseline.** vLLM Automatic Prefix Caching hashes KV blocks using parent
hashes, block tokens, and extra axes such as LoRA, multimodal hashes, and cache
salts. SGLang RadixAttention stores reusable prefixes in a radix tree. LMCache
externalizes KV and supports reuse across engine instances. NVIDIA Dynamo KVBM
offloads KV to CPU/disk tiers and integrates with KV-aware routing and
disaggregated serving.

**Parity requirement.**

- model id, tokenizer id, adapter id, position mode, and cache salt in identity;
- residency tier and owner recorded separately from payload;
- cache events exposed as metrics, not just internal engine counters;
- failure to restore/load KV is a typed miss or fault, not silent recompute.

**Local state.** `fak/internal/radixkv` implements RadixAttention-style prefix
discovery over local KV, and `radixkv.CacheEntry()` lowers radix nodes into
`cachemeta.Entry`. `cachemeta.FromKVPrefix()` already records model, tokenizer,
position mode, residency, taint/scope, and lookup fault reasons.

**Scope.** KV-prefix is tracked as a cache plane — `radixkv` is a local kernel
primitive for exact prefix reuse and policy/quarantine span eviction — but fak
does not try to out-perform provider prefix caching intra-session; the
`EXPLAINER-trust-floor-two-lenses` note names that serving-platform play
off-thesis.

**Gap.** Live engine routing/offload events are not yet normalized into the same
cache-entry event stream as tool/context entries.

### 2.3 Non-prefix and segment-level KV reuse

**SOTA baseline.** SparseX targets non-prefix, cross-request, cross-turn, and
cross-agent repeated segments, with RoPE alignment and sparse recomputation to
repair context interactions. MiniPIC takes the complementary position-repair
angle: it stores unrotated K and applies RoPE inside attention at per-request
logical positions, so a cached span can be reused at different positions without
the post-RoPE position binding. LMCache and CacheBlend-style systems make the same
direction clear: non-prefix reuse is real, but it is correction/quality work, not
free exact prefix reuse.

**Parity requirement.**

- exact prefix hits and approximate segment hits must be different verdicts;
- every approximate hit needs a fallback to exact recompute;
- record `position_mode`, recompute plan, fault rate, and task-quality delta;
- never sell approximate hit rate as correctness.

**Local state.** `cachemeta.PositionRecomputeRequired`,
`ReasonPositionMismatch`, and `ReasonApproxFault` are the right vocabulary, but
there is no production segment-reuse path.

**Gap.** Build this as a probationary experiment only after exact prefix and
tool-result coherence are wired into observability.

### 2.4 KV artifacts, snapshot/branch, and shared pools

**SOTA baseline.** "Can I Buy Your KV Cache?" frames precomputed KV as a
publisher-hosted, token-exact prefill artifact for public repeated content.
`thaw-ai/thaw` snapshots and branches live vLLM/SGLang sessions, including KV.
PolyKV shares one compressed KV pool across multiple agents with measured memory
reduction and bounded quality degradation.

**Parity requirement.**

- a KV artifact manifest must bind source text digest, model, tokenizer, adapter,
  precision, position convention, producer, and integrity checksum;
- imported KV is model-bound performance material, not semantic proof;
- compressed/shared pools need quality probes and scope policy;
- third-party or market KV needs provenance and access-control metadata.

**Local state.** `cachemeta.Entry` can represent an external KV artifact by
identity, derivation, validity, security, residency, and metrics.

**Gap.** There is no signed/importable KV manifest or resident-claim lowering
checker in-tree.

### 2.5 Tool-result caching

**SOTA baseline.** ToolCaching uses semantic and system features to decide
cacheability, TTL, admission, and eviction for tool calls. TVCache shows the
stateful version: a tool output can be reused only when the relevant prior tool
history/sandbox state matches. AWS's Agentic AI Lens now treats prompt,
retrieval, tool-output, session, and configuration caches as a production best
practice, with per-layer TTLs and metrics.

**Parity requirement.**

- classify read/pure/static/write tools before admission;
- destructive or state-mutating calls never reuse by args alone;
- mutable reads need external witnesses: etag, content hash, git SHA, lease
  epoch, DB row version, source digest, or sandbox snapshot id;
- stateful tools key on relevant history/state, not just current call args;
- admission and eviction consider latency/cost/value, not LRU alone.

**Local state.** `fak/internal/vdso` has a three-tier local path: pure functions,
content cache, and static table. It canonicalizes JSON args, re-checks
destructive names/hints, has scoped epoch invalidation, exposes mutation and
revocation buses, and can fail closed on revoked or overflowed witness ledgers.
`cachemeta.FromVDSOKey()` describes tier-2 entries as tool-result cache entries.

**Gap.** vDSO still needs first-class `cachemeta` event emission, consumer
tracking, and per-tool witness adapters instead of relying mainly on internal
epochs.

### 2.6 Semantic, intent, and plan-template caches

**SOTA baseline.** Agentic Plan Caching stores reusable plan templates and adapts
them to similar tasks. Structured Intent Canonicalization argues that agent
caching fails when keys optimize for generic similarity instead of consistency
and precision; it proposes an intent-decomposition cascade with abstention. AWS
also names semantic caches and plan-template caches as distinct agent cost
layers.

**Parity requirement.**

- semantic cache keys need precision thresholds, abstention, and testable
  clustering metrics;
- cached responses cannot directly execute effects without fresh policy and
  state checks;
- plan templates are advisory artifacts, keyed by task class, parameters, plan
  schema, tool manifest, policy version, and freshness;
- plan reuse must re-enter `plancfi`/adjudication before any tool effect.

**Local state.** `fak/internal/plancfi` already enforces approved tool-call
graphs per trace and traps deviations. That gives `fak` a stronger place to
attach plan-template reuse than a normal agent harness: a reused plan can become
an approved call graph, then every call still crosses the kernel.

**Gap.** No `PlanePlanTemplate` or intent-key cache exists yet. The first version
should be read-only/advisory and decline aggressively on any uncertainty.
"Abstaining" is a typed non-Hit (`Miss` or `Revalidate`), never a `HIT` — the
`cachemeta.LookupKind` vocabulary (hit/miss/revalidate/transform/quarantine/fault)
defines no `ABSTAIN`, so abstention is expressed as a non-Hit, not as a verdict
named ABSTAIN.

### 2.7 Memory views and compaction caches

**SOTA baseline.** The current memory SOTA is converging on typed, multi-view
memory: raw pages plus summaries, facts, QA pairs, graphs, timelines, and task
views. The companion memo `MEMORY-COMPACTION-SOTA-2026-06-19.md` maps this.

**Parity requirement.**

- raw page stays source of truth;
- each view is a derived cache entry keyed by input digests, view type, producer,
  policy epoch, and scope;
- materialization is a verdict, not automatic insertion into prompt context;
- source coverage, faithfulness, stale-view, and selection-integrity failures are
  measured.

**Local state.** `recall.Page.CacheEntry()` lowers persisted pages into the
common cache metadata contract without paging in the raw bytes.

**Gap.** Derived memory views are not yet represented as cache entries with
coverage and invalidation metadata.

---

## 3. Novelty Posture

Do **not** claim these as novel:

- prompt/provider cache economics;
- KV prefix/radix reuse;
- KV offload, routing, or disaggregated prefill;
- non-prefix/segment KV reuse;
- tool-result caching;
- stateful tool-value caching;
- semantic cache or intent canonicalization;
- plan-template caching;
- virtual/multi-view agent memory.

The defensible claim is the assembly and contract:

> `fak` can make cache reuse part of the same kernel verdict that mediates tool
> calls, result admission, taint/scope, recall page-in, plan-CFI, and witness
> verification. A shared cache hit is allowed only when identity, state,
> freshness, authority, integrity, causality, economics, and quality all pass or
> return a typed non-hit.

That is narrower than "we invented agentic caching" and stronger than "we have a
cache." It is also measurable: the cache layer should emit why it hit, missed,
revalidated, transformed, quarantined, or faulted.

---

## 4. Local Items Already Present

| Local item | What it contributes to agentic caching |
|---|---|
| `fak/internal/cachemeta` | payload-free cache-entry metadata and typed lookup verdicts |
| `fak/internal/radixkv/cachemeta.go` | KV-prefix entries with model/tokenizer/position identity |
| `fak/internal/recall/cachemeta.go` | context-page entries without exposing sealed bytes |
| `fak/internal/vdso` | pure/static/content tool fast path, canonical args, scoped invalidation, revocation ledger |
| `fak/internal/gateway/coherence.go` | mutation/refutation feed that can generalize beyond vDSO |
| `fak/internal/ctxmmu` | write-time result admission and quarantine before context visibility |
| `fak/internal/plancfi` | approved plan/tool graph enforcement after plan reuse |
| `fak/internal/abi.Ref` | digest, taint, scope, inline/blob distinction |
| `docs/explainers/kv-cache-agentic-context.md` | agent-specific prefix cache mechanics and failure modes |
| `MEMORY-COMPACTION-SOTA-2026-06-19.md` | virtual memory/view-cache direction |

This is enough to justify treating agentic caching as a first-class workstream.
It is not enough to claim the full primitive is shipped.

---

## 5. Build Sequence

### Step 1: Extend the cache vocabulary to the full agentic stack

Add explicit planes only when there is a real adapter or test:

- `plan_template`
- `semantic_intent`
- `memory_view`
- `kv_artifact`
- `stateful_tool_value`

Keep `cachemeta` payload-free. Do not import high-tier packages into it.

### Step 2: Make vDSO emit cache-entry events

For every tier-2 fill/hit/eviction/revocation, emit:

- `EntryID`, tool, args digest, tool schema/version;
- witness type/value;
- scope/taint/admission;
- invalidation mode and mutation/refutation sequence;
- consumer trace/session where known.

This turns the strongest local cache into observable ground truth.

### Step 3: Add consumer tracking for shared entries

Minimum proof:

1. Session A and B consume one shared entry under witness W.
2. W is refuted through the gateway/vDSO path.
3. The entry is evicted or quarantined.
4. A and B are named as affected consumers.
5. A follow-up read cannot silently re-serve W.

### Step 4: Add provider prompt-cache telemetry

Normalize OpenAI/Anthropic/Gemini/Bedrock usage fields into local metrics:

- cache read tokens;
- cache write/create tokens;
- prompt serializer hash;
- cache breakpoint mode;
- provider TTL or retention mode where known;
- offline first-divergence location.

This prevents local work from double-counting provider wins.

### Step 5: Prototype plan/intent caching as advisory only

Start with plan-template lookup that returns:

- `HIT` only for exact task class + schema + policy + tool manifest match;
- `REVALIDATE` when current state/witness is missing;
- `MISS` or `FAULT` on low precision, compound intent, or conflicting state.

The output should be an approved `plancfi.Plan` candidate, not direct permission
to execute. Every tool call still crosses `plancfi` and the normal adjudicator.

### Step 6: Put approximate KV reuse behind a fault budget

Non-prefix KV reuse should remain experimental until exact paths are fully
observable. Required counters:

- attempted segment hits;
- exact recompute audits;
- approximate faults;
- task-quality delta;
- position mismatch and recompute-token counts.

### Step 7: Add cache-miss reasons to benchmarks

Every benchmark that claims a cache benefit should report:

- hit/miss/revalidate/transform/quarantine/fault by plane;
- miss reason;
- cache write amplification;
- saved prefill tokens and saved tool latency separately;
- quality/fault metrics for approximate paths;
- provider cached tokens separately from local hits.

---

## 6. Refusal Rules

These are review-time anti-patterns to refuse:

1. **Args-only cache for mutable tools.** Needs a state witness.
2. **One TTL for all tools/data.** Static docs and stock quotes do not age alike.
3. **Fleet-shared tainted result.** Private taint is one-agent risk; shared taint
   is an amplifier.
4. **Semantic cache directly executes effects.** It may suggest; it may not
   bypass adjudication.
5. **Plan cache bypasses plan-CFI.** A cached plan is not an execution permit.
6. **Provider prompt-cache hit treated as trust evidence.** It is cost/latency
   evidence only.
7. **Approximate KV hit without exact recompute fallback.** That is a quality
   bug, not a cache.
8. **KV artifact imported from digest alone.** Needs model/tokenizer/position and
   provenance.
9. **Unbounded consumer graph.** Causality metadata must itself be compactable.
10. **Single blended hit rate.** Prompt, KV, tool, semantic, plan, and memory-view
    hits are different claims.

---

## 7. Source Notes Checked In This Pass

Provider and production docs:

- OpenAI prompt caching:
  https://developers.openai.com/api/docs/guides/prompt-caching
- Anthropic prompt caching:
  https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Gemini context caching:
  https://ai.google.dev/gemini-api/docs/caching
- Amazon Bedrock prompt caching:
  https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
- AWS Agentic AI Lens, caching/data access:
  https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentperf03-bp04.html
- AWS Agentic AI Lens, intelligent caching:
  https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/agentcost02-bp03.html
- vLLM Automatic Prefix Caching:
  https://docs.vllm.ai/en/stable/design/prefix_caching/
- LMCache docs:
  https://docs.lmcache.ai/
- NVIDIA Dynamo KV cache offloading:
  https://docs.nvidia.com/dynamo/backends/v-llm/kv-cache-offloading
- `thaw-ai/thaw`:
  https://github.com/thaw-ai/thaw

Recent research:

- "Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon
  Agentic Tasks", arXiv:2601.06007:
  https://arxiv.org/html/2601.06007v2
- "ToolCaching: Towards Efficient Caching for LLM Tool-calling",
  arXiv:2601.15335:
  https://arxiv.org/html/2601.15335v1
- "TVCache: A Stateful Tool-Value Cache for Post-Training LLM Agents",
  arXiv:2602.10986:
  https://arxiv.org/html/2602.10986v1
- "Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching",
  arXiv:2506.14852:
  https://arxiv.org/html/2506.14852v1
- "Why Agent Caching Fails and How to Fix It: Structured Intent
  Canonicalization with Few-Shot Learning", arXiv:2602.18922:
  https://arxiv.org/html/2602.18922v2
- "PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM
  Inference", arXiv:2604.24971:
  https://arxiv.org/abs/2604.24971
- "SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM
  Serving", arXiv:2606.01751:
  https://arxiv.org/html/2606.01751v2
- "MiniPIC: Flexible Position-Independent Caching in <100LOC", arXiv:2606.13126:
  https://arxiv.org/html/2606.13126v1
- "Can I Buy Your KV Cache?", arXiv:2606.13361:
  https://arxiv.org/html/2606.13361v1
