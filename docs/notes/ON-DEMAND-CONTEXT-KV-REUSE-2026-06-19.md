---
title: "fak note: on-demand context and non-prefix KV reuse (2026-06-19)"
description: "Examines whether fleet can serve context on demand from digest-addressed state: prefix KV reuse is exact while non-prefix segment reuse needs a corrective path."
---

# On-Demand Context And Non-Prefix KV Reuse

> Date: 2026-06-19.
> Scope: focused proof/refutation pass for the "saturation point" question:
> if `fleet` stores all state as pages, cache entries, and KV spans, can context
> become user-queryable and mix-and-match on demand?

## 0. Verdict

Yes, but the exact layer matters.

`fleet` can get to an on-demand context model where a user or agent asks for a
working set and the system materializes the relevant pages, facts, summaries,
timeline rows, tool results, and same-model KV prefixes nearly at will.

It cannot treat arbitrary non-prefix KV chunks as exact lego bricks in a normal
decoder-only transformer. Prefix/radix KV reuse is exact. Semantic page/view
recomposition is flexible but not output-identical. Non-prefix KV segment reuse
is real, but it is a corrective path: position handling, cross-attention repair,
selective recompute, quality probes, and fallback to exact recompute.

The useful product shape is therefore:

```text
raw digest-addressed state
  -> typed, provenance-bound views
  -> materialization verdict
  -> prompt/KV/provider render target
  -> page fault / recompute / refuse when coverage or trust fails
```

The prompt is no longer the memory. The prompt is one render of a queryable
memory image.

## 1. What Is Already Shipped Locally

The repo already has most of the substrate:

| Capability | Current local support | What it proves |
|---|---|---|
| Raw state as durable pages | `fak/internal/recall`, `fak/internal/cdb` | Finished sessions can be loaded as a page table over CAS bytes; `WorkingSet(query,k)` demand-pages a bounded benign set. |
| Admission and sealing | `fak/internal/ctxmmu`, `recall.Resolve` | Sealed pages are refused on page-in unless cleared and re-screened; descriptors for sealed pages do not expose bytes. |
| User/agent context suppression | `recall.RequestContextChange` | Future context assembly can tombstone a page without deleting audit evidence. |
| Common cache contract | `fak/internal/cachemeta` | Cache entries already name media, plane, model/tokenizer/position mode, witness, taint/scope, residency, consumers, invalidation, and quality/fault metrics. |
| Exact prefix KV reuse | `fak/internal/model`, `fak/internal/radixkv` | `TestKVPrefixReuseMatchesRecompute` proves prefix reuse is bit-identical; `radixkv` discovers shared prefixes automatically. |
| KV quarantine | `fak/internal/kvmmu` | A context-MMU verdict can mechanically evict a K/V span from kernel-owned attention state. |

### 1.1 V1 proof added on 2026-06-19

The first modular on-demand context surface is now implemented:

- `fak/internal/contextq` defines `Request`, `Result`, `MemoryViewRecord`,
  `MaterializationVerdict`, `Refusal`, `Omission`, and `RenderPlan`.
- `fak/internal/cachemeta.FromMemoryView` lowers derived context views into the
  common cache metadata contract as `memory_view` entries with source refs,
  policy version, scope, taint, coverage, and faithfulness probes.
- `fak/internal/cdb.PageCacheEntry` exposes page-table cache handles without
  exposing bytes; page bytes still enter only through the existing CDB/recall
  page-in gate.
- `fak debug --cmd context-query` is the user-facing v1 demo.

Local witness command:

```powershell
cd fak
go run ./cmd/fak debug --cmd context-query `
  --dir experiments/contextq/demo-image `
  --out experiments/contextq/context-query-demo.json `
  --query "refund fee trust violation" `
  --pins "WebSearch" `
  --excludes "secret" `
  --budget-bytes 7000 `
  --policy-version contextq-v1
```

That demo materializes three benign slices/views, refuses the trust-violation
page as `sealed_by_trust_gate`, refuses the secret page as
`excluded_by_request`, and writes a typed render plan plus cache handles to
`fak/experiments/contextq/context-query-demo.json`.

Verification commands:

```powershell
cd fak
go test ./internal/cachemeta ./internal/cdb ./internal/contextq ./cmd/fak
go test ./internal/recall ./internal/radixkv ./internal/model
```

This proves Steps 1-3 of the build path in a v1 form: a query endpoint, view
records, and typed materialization verdicts. It does not prove context-layout
compiler optimization or non-prefix KV reuse.

That means the next step is not "invent memory." It is a materialization layer
that asks: "which state should enter this context, in what form, under what
budget and witness?"

## 2. Baseline Proofs And Refutations

### 2.1 Prefix KV reuse is exact

For a causal decoder, token `i` at every layer depends only on tokens `0..i`.
If two prompts share the same token prefix, induction over positions and layers
gives identical hidden states, K/V tensors, and logits for that prefix. Reusing
that prefix KV and prefilling only the suffix is therefore exact, assuming the
same model, tokenizer, adapter, precision regime, serializer, and position
scheme.

Local witness: `fak/internal/model/kvreuse_test.go` checks prefix reuse against
full recompute and requires identical argmax, continuation, and `max|delta|=0`.
`fak/internal/radixkv` extends this from declared prefix to discovered prefix.

External baselines agree. vLLM's automatic prefix caching hashes blocks using
parent prefix, block tokens, and extra identity axes (the "Extra hashes" vLLM cites: LoRA IDs, multimodality input hashes, cache salts[[https://docs.vllm.ai/en/stable/design/prefix_caching/]](https://docs.vllm.ai/en/stable/design/prefix_caching/)). SGLang RadixAttention
stores token prefixes in a radix tree and reuses the matched prefix. OpenAI and
Anthropic prompt caches both require exact prefix structure.

### 2.2 Non-prefix KV reuse is not exact by default

Suppose a span `S` was prefilled inside context `A + S`, and later we want to
reuse it inside `B + S` or `P + S + Q`. In a decoder transformer:

```text
h_i^l = block_l(h_i^(l-1), attention(q_i, K_<=i^(l), V_<=i^(l)))
K_i^l,V_i^l = projections of h_i^(l-1)
```

At layer 1, a token's K/V mostly depends on its embedding and position. At
deeper layers, `h_i^(l-1)` has already attended to earlier tokens. Therefore
the K/V for the same surface span can differ when the preceding context differs.
RoPE or absolute position changes add a second failure mode.

So arbitrary direct reuse of a middle chunk's full KV cache is not exact. It
ignores either positional mismatch, cross-attention with preceding chunks, or
both. CacheBlend's paper illustrates this directly: full non-prefix KV reuse can
give low-quality answers because it ignores cross-attention, while selective
recompute repairs part of the gap.

The refutation is precise: non-prefix repeated bytes are reusable as text/pages
and as candidate KV material, but the KV hit must be labeled approximate or
corrective unless the system also proves the current causal dependencies match.

### 2.3 Position-independent cache helps but does not solve everything

MiniPIC-style designs store unrotated K and apply RoPE inside attention using
per-request logical positions. That attacks position brittleness and makes
multi-position reuse more maintainable. It does not by itself make a segment's
deep-layer hidden state independent of the tokens that preceded it.

Position handling is necessary for flexible segment reuse. It is not sufficient
for exact semantic equivalence.

### 2.4 Exact non-prefix reuse exists only under special contracts

Non-prefix reuse can be exact in narrower cases:

- The reused span is causally isolated by an attention mask or special primitive,
  so it is intentionally independent of earlier text.
- The system recomputes every token/layer whose state depends on the new prefix,
  reducing reuse to lower-level or partial reuse.
- The model architecture treats the chunk as separate encoder/cross-attention
  memory rather than ordinary decoder KV.
- The preceding context, position, model, tokenizer, serializer, and all relevant cache axes match exactly, which collapses back toward prefix/radix conditions. (A **cache axis** is any dimension of the binding key that a cache lookup must match — for a KV manifest this includes model ID, tokenizer ID, precision, position convention, adapter ID, and other identity dimensions; see [`docs/proofs/cachemeta.md`](../proofs/cachemeta.md).)

Everything else needs a fault budget.

## 3. The On-Demand Context Model

The usable version is a managed context image:

```text
RawPage
  digest
  role/tool
  descriptor
  taint/scope
  witness
  source epoch

ViewRecord
  view_id
  # Semantic view shapes (model-agnostic content projections):
  # snippet, fact, QA, summary, timeline, graph_edge, skill_context
  # Storage classes (model-bound materialization forms):
  # kv
  view_type: snippet | fact | QA | summary | timeline | graph_edge | skill_context | kv
  source_pages
  source_digests
  producer
  policy_version
  coverage
  faithfulness_probe
  stale_rule
  cachemeta.Entry

MaterializationVerdict
  HIT(view)
  FAULT(raw_page)
  RECOMPUTE(stale_view)
  REFUSE(scope_or_taint)
  ABSTAIN(no_coverage)
```

This gives the user "mix and match" at the safe layer:

- "Use the final API inventory plus the Qwen readiness pages, but exclude old
  release notes."
- "Pin this plan and the latest benchmark, summarize everything else."
- "Show what state supports this claim."
- "Build a 32k context from only trusted docs and current code facts."
- "Swap the timeline view for raw excerpts because the summary looks lossy."

The materializer then decides whether each requested piece is:

- inserted as raw text;
- inserted as a derived view with source links;
- represented as a stable handle;
- served through provider prompt cache;
- attached as exact same-model prefix/radix KV;
- rejected or faulted to raw bytes.

## 4. Proactive Context

Proactive work should not eagerly build every possible view. It should build the
cheap and likely-useful frontier, then lazily fault deeper.

### 4.1 On every write

When a tool result, file read, model output, or session event lands:

1. Store raw bytes/content in CAS.
2. Emit a `cachemeta.Entry` with witness, taint, scope, producer, residency, and
   invalidation mode.
3. Build a cheap descriptor and source fingerprint.
4. Decide whether it is eligible for shared reuse.
5. Update consumer/dependency edges if it is derived from prior entries.

This is the invariant. Views are optional; raw pages are not.

### 4.2 In the background

For hot or high-value pages:

1. Build small typed views: facts, QA pairs, timeline rows, entity edges,
   benchmark rows, plan status rows.
2. Attach coverage and source pointers.
3. Run cheap faithfulness probes against raw pages.
4. Precompute exact prompt/KV prefixes only for stable bundles that many requests
   will share.
5. Keep non-prefix KV segment attempts in probation with exact recompute audits.

### 4.3 Before a model call

Given `{query, task, user pins, token budget, authority scope}`:

1. Choose the working set from page descriptors and view indexes.
2. Refuse sealed or out-of-scope entries before ranking.
3. Prefer stable handles and derived views when coverage is enough.
4. Fault to raw pages when coverage is weak or the user asks for evidence.
5. Render stable prefix first, working set second, volatile tail last.
6. Record every omission and every view used, so a later answer can be audited.

### 4.4 After the answer

Track:

- pages/views consumed;
- faults requested by the model or user;
- stale/refuted witnesses;
- view quality deltas;
- prompt/KV/provider cache hit/miss reasons;
- omissions that caused follow-up faults.

That feedback is how the system learns which views to build proactively.

## 5. Saturation Point

The commodity prefix-cache opportunity saturates quickly:

- Provider prompt caches already harvest exact prefix reuse.
- vLLM/SGLang/LMCache/Dynamo-style systems already route, offload, and reuse KV.
- Local `radixkv` already proves exact prefix/radix reuse on hit-rate semantics.

The unsaturated opportunity is not "more prefix caching." It is:

1. making state queryable by users and agents;
2. turning memory views into adjudicated cache artifacts;
3. separating semantic recomposition from exact KV reuse;
4. attaching trust, freshness, authority, and consumer edges to every hit;
5. proactively materializing the likely next working set.

The saturation metric should shift from raw cache hit rate to a dimensionally consistent
set covering both efficiency and correctness:

```text
# Efficiency: tasks answered per token spent
efficiency = answered_tasks_with_sources / resident_tokens

# Correctness: what fraction of requests stayed correct
correctness = answered_tasks_with_sources /
              (answered_tasks_with_sources + page_faults + stale_faults + quality_faults)
```

A system that gets a high non-prefix KV hit rate but silently loses quality is
worse than a system that faults to raw pages and stays correct.

## 6. Concrete Build Path

### Step 1: Promote `cdb.WorkingSet` into a user surface

Expose a first-class query endpoint:

```text
context.query(q, budget, scope, pins, excludes)
  -> frames
  -> slices
  -> refused
  -> omissions
  -> render_plan
```

**frames**: page-table rows from the session backtrace (the `cdb.Frame` type), carrying metadata (step, role, descriptor, length, digest, taint, sealed/tombstoned status) but no raw bytes. A frame is a safe descriptor even for sealed pages.

**slices**: handles to materialized context slices (the `Slice` type, alias for `recall.Slice`), each carrying the source step, role, descriptor, and the paged-in bytes.

This is the immediate "on demand context" product. It uses shipped `recall/cdb`
behavior and does not wait for non-prefix KV research.

### Step 2: Add `MemoryViewRecord`

A view record should lower into `cachemeta.Entry`, just like recall pages and
radix prefixes already do. Minimum fields:

```text
view_type
source_page_ids
source_digests
producer
policy_version
scope
taint
coverage
faithfulness_probe
ttl_or_witness
```

Every view is recomputable. No view replaces raw memory.

### Step 3: Add `MaterializationVerdict`

Do not let the materializer return plain text. It should return:

```text
HIT(view_id)
FAULT(page_id)
RECOMPUTE(view_id, stale_reason)
REFUSE(page_id, reason)
ABSTAIN(query, reason)
```

That is *related to*, not the same set as, the existing `cachemeta.LookupVerdict`
(`hit`/`miss`/`revalidate`/`transform`/`quarantine`/`fault`): the materializer's five
kinds collapse the cache-plane lifecycle into materialization outcomes, and add two with
no cachemeta analogue (`RECOMPUTE`, `ABSTAIN`). The two sets are reconciled — not unified
— at the one place they meet, the KV-view gate `contextq.GateKVView`; see the mapping
table in [`docs/proofs/contextq.md`](../proofs/contextq.md) (issue #227).

### Step 4: Add a context layout compiler experiment

Render the same state under several budgets:

- 16k minimal;
- 32k working;
- 64k evidence-heavy;
- provider-cache-friendly;
- local-KV-friendly.

Measure prefix divergence, provider cached tokens, raw bytes paged in,
view coverage, faults, and answer quality.

### Step 5: Keep non-prefix KV reuse as an audited experiment

A first local non-prefix KV experiment should have hard gates:

- exact-prefix baseline;
- full-recompute oracle;
- segment-hit candidate;
- position mode recorded;
- selective recompute plan;
- logits/task-quality delta;
- fault-to-exact path;
- deny promotion if the fault budget is exceeded.

No public claim should call it exact unless it passes the oracle under the same
model/tokenizer/position/serializer axes.

## 7. User-Facing Shape

The user should be able to ask context questions directly:

```text
show context for "Qwen readiness blockers"
pin pages matching "DGX endpoint smoke"
exclude stale release candidates
answer using only pages admitted after witness git:abc123
diff this answer's source set against yesterday
expand the summary into raw excerpts
keep this 32k context target unless faults exceed 3
```

This turns state into an inspectable memory map. The model sees only the
materialized render, but the user can query, pin, exclude, and audit the backing
state.

## 8. Kill Criteria

Stop or narrow the effort if:

- users cannot predict why a page/view was included or omitted;
- derived views lack source page IDs;
- fault rate is high enough that the system keeps falling back to raw transcript;
- non-prefix KV reuse improves hit rate but fails exact recompute probes;
- proactive view generation costs more than it saves;
- provider prefix caching already captures the measured benefit;
- stale or sealed pages can influence selection even when their bytes are not
  emitted.

## 9. Bottom Line

The high-opportunity concept is not arbitrary KV splice-and-play. That is the
fragile part.

The high-opportunity concept is on-demand managed context: raw state is immutable
and queryable; views are derived, sourced, and cacheable; materialization is a
verdict; exact KV/prompt reuse is used where it is mathematically exact; and
approximate segment KV reuse is allowed only behind audits and recompute fallback.

That gets us very close to "mix and match nearly at will" for users, while keeping
the correctness boundary honest.

## Sources Checked

Primary/external:

- OpenAI prompt caching docs: https://developers.openai.com/api/docs/guides/prompt-caching
- Anthropic prompt caching docs: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- vLLM automatic prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
- SGLang paper / RadixAttention: https://arxiv.org/pdf/2312.07104
- CacheBlend paper: https://arxiv.org/abs/2405.16444 and https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf
- SparseX segment-level KV reuse: https://arxiv.org/html/2606.01751v1
- MiniPIC position-independent caching: https://arxiv.org/html/2606.13126
- LMCache KV layer: https://arxiv.org/html/2510.09665v2
- CacheSlide cross-position KV reuse: https://www.usenix.org/conference/fast26/presentation/liu-yang

Local:

- `AGENTIC-CACHING-SOTA-2026-06-19.md`
- `MEMORY-VIEW-CONTRACT-2026-06-26.md`
- `docs/explainers/kv-cache-agentic-context.md`
- `fak/RECALL-RESULTS.md`
- `fak/RADIXATTENTION-RESULTS.md`
- `fak/internal/cachemeta`
- `fak/internal/recall`
- `fak/internal/cdb`
- `fak/internal/model`
- `fak/internal/radixkv`
- `fak/internal/kvmmu`
