---
title: "fak concept-disambiguation scorecard - is every similar-sounding concept crystal-clear"
description: "Inward naming scorecard: each confusable fak concept positioned on the grounded / defined / disambiguated / anchored axes, with one clarity verdict per concept. Two driven numbers: coverage (of the confusable concept space discovered in the tree) and disambiguation-debt."
---

# Concept-disambiguation scorecard - crystal clarity across similar-sounding names

<!-- concept-disambiguation-scorecard: 2026-06-26 @initial - process: tools/concept_disambiguation_scorecard.py - data: tools/concept_disambiguation_scorecard.data/ -->

The sibling scorecards grade fak's code, docs, and competitive standing. This one asks the question that bites a reader as the system grows: **of the massive, growing set of similar-sounding names (cache, vCache, KV cache, cachemeta, the provider prompt-cache), is each distinct concept crystal-clear - one canonical name, a written definition, and an explicit line drawn against the siblings it is confused with?** Every number below is re-derived by `tools/concept_disambiguation_scorecard.py` and cross-checked against the real tree (the grounding token must appear in the production corpus; the glossary anchor must exist; a `distinct_from` reference must resolve). No verdict is hand-typed.

> Regenerate: `python tools/concept_disambiguation_scorecard.py --markdown-dir docs/concept-disambiguation-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Score** | **42.5/100** (grade F) = 4.2/10 |
| **Coverage** | **11.5%** (55/480 confusable tree tokens positioned) |
| **Disambiguation-debt** | **425** (clarity 0 + coverage 425) |
| Crystal-clear concepts | 48 of 75 positioned |
| As of | 2026-06-26 (fak v0.34.0) |

> **Read this right.** The score is deliberately LOW at birth: it grades the WHOLE confusable namespace discovered in the tree, not the few concepts already catalogued. A low coverage number is the honest statement that most similar-sounding names are not yet disambiguated - which is exactly the debt this scorecard exists to retire.

## Standing at a glance

```text
concept-disambiguation chart - 75 concepts - score 42.5/100 (grade F) - disambiguation-debt 425

clarity ladder (count of concepts, best -> fog):
  * crystal       ############################ 48
  o defined       ################............ 27
  ~ drifting      ............................ 0
  x colliding     ............................ 0
  . undocumented  ............................ 0

clarity mix by family (each cell = one concept):
  attention        oooooooooo         (10 concept(s); 0 crystal)
  cache            **************ooo  (17 concept(s); 14 crystal)
  context-ctx      ***oo              (5 concept(s); 3 crystal)
  gateway-engine   *****ooo           (8 concept(s); 5 crystal)
  guard-gate       ********oo         (10 concept(s); 8 crystal)
  policy-capability ******oo           (8 concept(s); 6 crystal)
  score-debt       ***                (3 concept(s); 3 crystal)
  session-runtime  ****ooo            (7 concept(s); 4 crystal)
  witness-proof    *****oo            (7 concept(s); 5 crystal)

coverage by family (positioned / discovered):
  gateway-engine   ##.......................... 9/108
  guard-gate       ###......................... 7/63
  context-ctx      ##.......................... 4/59
  session-runtime  ##.......................... 4/56
  attention        ##.......................... 4/55
  cache            #######..................... 15/62
  policy-capability ###......................... 5/52
  witness-proof    #####....................... 6/35
  score-debt       #################........... 3/5

namespace coverage  [####............................] 11.5%  (55/480 confusable tokens positioned)

legend: * crystal   o defined   ~ drifting   x colliding   . undocumented
```

## The clarity ladder

| Verdict | Means |
|---|---|
| * crystal | grounded + defined + a line drawn against siblings + that line anchored in a doc that exists |
| o defined | grounded + defined + a distinction line, but the line is not written in a discoverable doc |
| ~ drifting | grounded + defined, but no line drawn against its siblings (you know what it is, not what it is NOT) |
| x colliding | shares a canonical name with another concept - a true ambiguity, fixable only by a rename |
| . undocumented | appears in the tree, but the catalog gives no definition |

## The concepts (best verdict first)

| | Verdict | Kind | Family | Canonical - definition |
|---|---|---|---|---|
| * | crystal | subsystem | cache | **KV cache** - The kernel-owned raw attention state: per-position Key and Value tensor rows for the running model, supporting in-place eviction and prefix reuse. |
| * | crystal | subsystem | cache | **vCache** - The virtual API cache: a page-table abstraction that models a remote provider's prefix cache as virtual pages, with a manifest of canonical prefix chains and warmth belief. |
| * | crystal | subsystem | cache | **cachemeta** - The typed metadata contract (tier 1): owns no payloads, names reusable cache entries, and carries their validity / security / residency metadata and typed lookup verdicts. |
| * | crystal | concept | cache | **Prompt cache** - The upstream provider feature: a prefix cached via cache_control breakpoints, reported back as cache_read_input_tokens / cache_creation_input_tokens with a specific pricing multiplier. |
| * | crystal | metric | cache | **Provider cache** - A cachemeta entry on plane=provider: the OBSERVED telemetry record of what the provider's prompt cache did (read/creation token counts), marked non-re-serveable local proof. |
| * | crystal | config | cache | **cache_control** - The WRITE placement mechanism: the cache_control breakpoint you place to tell the provider where to cache a prefix. |
| * | crystal | metric | cache | **cache_read** - The READ telemetry: cache_read_input_tokens, what the provider reports it served from its cache rather than re-prefilling, billed at 0.1x base input. |
| * | crystal | metric | cache | **cache_creation_input_tokens** - The WRITE telemetry: cache_creation_input_tokens, what the provider reports it wrote to its cache on this call, billed at 1.25-2.0x base input depending on TTL. |
| * | crystal | subsystem | cache | **RadixKV** - A local token-trie data structure for fast prefix lookup that points INTO KV-cache spans, with materialization binding so cross-model spans are never reused. |
| * | crystal | subsystem | cache | **enginecache** - The adapter that translates cachemeta invalidation directives into a remote serving engine's control API (SGLang / vLLM prefix-cache reset or span evict). |
| * | crystal | subsystem | cache | **ViewCache** - Storage of materialized lossy projections (summary, QA, facts) over canonical memory pages, with FAULT / RECOMPUTE / HIT materialization verdicts. |
| * | crystal | subsystem | cache | **MemoryView** - The typed virtual-view contract binding a lossy derived projection (summary, graph) to its canonical raw-memory source by content digest + byte span, with provenance. |
| * | crystal | subsystem | cache | **Hardware-aware cache** - The placement POLICY that knows each tier's physical character (HBM / DRAM / NUMA-far / CXL / disk / remote) and demotes hot entries one tier colder under pressure. |
| * | crystal | symbol | cache | **KV transfer** - The migration DIRECTIVE emitted by hardware-aware placement: migrate / offload / restore / route a KV span between tiers. |
| * | crystal | subsystem | context-ctx | **context-MMU (ctxmmu)** - A write-time (post-tool) gate on tool RESULTS that decides if bytes enter context as-is, must be quarantined, or paged out to a pointer. |
| * | crystal | subsystem | context-ctx | **KV-MMU (kvmmu)** - The bridge that turns ctxmmu's logical quarantine verdict into a mechanical one by evicting K/V spans from the kernel's attention cache. |
| * | crystal | subsystem | context-ctx | **recall (session core dump)** - The persisted session core-dump: a page table over a content-addressed swap device, reloadable in a fresh process where a sealed page stays sealed. |
| * | crystal | subsystem | gateway-engine | **kernel** - The fak core: the one implementation of abi.Kernel that coordinates adjudication, vDSO lookup, engine dispatch, and result admission across the tool-call path. |
| * | crystal | subsystem | gateway-engine | **gateway** - The kernel-adjudicated wire: the HTTP and MCP-RPC surface that fronts the kernel for non-Go clients, re-validating untrusted wire arguments before they reach it. |
| * | crystal | subsystem | gateway-engine | **engine** - The inference-engine seam (EngineDriver): the abstract backend interface the kernel dispatches allowed tool calls to (mock, HTTP upstream, fused in-kernel model). |
| * | crystal | subsystem | gateway-engine | **vDSO (tool vDSO)** - The tool vDSO: a local fast path (pure registry, content-addressed cache, static table) that answers a tool call with zero engine round-trip. |
| * | crystal | subsystem | gateway-engine | **model (in-kernel model)** - The in-kernel inference core: a pure-Go forward pass that runs chat token decode over a loaded GGUF checkpoint across several architectures and quant schemes. |
| * | crystal | subsystem | guard-gate | **guard (fak guard kernel)** - The kernel itself: the in-process adjudication system that runs the decision chain and admits results, launched as `fak guard`. |
| * | crystal | concept | guard-gate | **gate (decision point)** - One decision point inside a guard, splitting by WHEN it fires: pre-call adjudicators, post-call result admitters, and git-hook gates. |
| * | crystal | symbol | guard-gate | **ResultAdmitter (post-call gate)** - A post-call gate: inspects a tool RESULT after execution and admits / quarantines / transforms it (ctxmmu, normgate, secretgate). |
| * | crystal | subsystem | guard-gate | **git-hook gate** - A commit-boundary check at git pre-commit / commit-msg (gate_brokenlink, gate_secretshape, gate_provenance, ...). |
| * | crystal | doc-term | guard-gate | **trunk guard** - A git hook that refuses commits made off the trunk (the OFF_TRUNK law), keeping all work on main. |
| * | crystal | subsystem | guard-gate | **repo guard (repoguard)** - A PreToolUse hook that refuses Bash/Write/Edit calls targeting paths outside the workspace tree. |
| * | crystal | subsystem | guard-gate | **gitgate (adjudicator)** - An adjudicator rung that prefilters git hazards (force-push, --no-verify, rebase -i) from shell tool calls at runtime. |
| * | crystal | subsystem | guard-gate | **shipgate (RSI promotion gate)** - An adjudicator that enforces RSI keep-or-revert verdicts on a code change based on witness-verified metric gain, never the candidate's own claim. |
| * | crystal | concept | policy-capability | **capability floor** - The deployable, declarative authorization layer that defines exactly which tools an agent may call by name, loaded as a JSON manifest at runtime. |
| * | crystal | config | policy-capability | **policy manifest** - The versioned JSON structure an operator edits to configure the capability floor, mapping 1:1 to adjudicator.Policy with deny reasons validated against the closed vocabulary. |
| * | crystal | subsystem | policy-capability | **adjudicator** - The in-process DOS reference monitor: the rung that folds a decision chain to prove a tool call allowed, denied, or deferred under the loaded policy. |
| * | crystal | symbol | policy-capability | **abi.Verdict** - The discriminated-union decision an adjudicator returns, keyed by kind (Allow, Deny, Defer, Transform, Quarantine, RequireWitness) with typed payloads. |
| * | crystal | symbol | policy-capability | **reason code** - The closed, additive vocabulary of refusal reasons (DEFAULT_DENY, POLICY_BLOCK, SELF_MODIFY, ...) that every deny verdict cites, never free text. |
| * | crystal | config | policy-capability | **posture (tool admission)** - The policy's default behavior after all provable refusal checks pass: PostureFailClosed (deny everything not allowed) or PostureAdmitAndLog (admit low-risk reads with forensic metadata). |
| * | crystal | subsystem | score-debt | **scorecard** - One deterministic measurement of a surface that folds reality into a single *_debt integer plus an A-F grade (the family is documented in the scorecard skill). |
| * | crystal | subsystem | score-debt | **scorecard control pane** - The fold that sums every scorecard's *_debt into one portfolio number with a pinned ratchet that reds only on a regression above baseline. |
| * | crystal | metric | score-debt | **disambiguation-debt** - This scorecard's integer: clarity defects of positioned concepts plus coverage gaps (confusable tree tokens with no row). |
| * | crystal | subsystem | session-runtime | **Session** - The full drive record for one served run (run-state, budget, priority, pace), keyed by TraceID and persisting across turns. |
| * | crystal | concept | session-runtime | **Turn** - One model round-trip within a session: the agent submits input, the model generates output, and results are admitted to context. |
| * | crystal | symbol | session-runtime | **Slot** - The immutable free/busy signal emitted when a session leaves the eligible set (budget exhaustion, pause, drain, stop), freeing scheduling capacity. |
| * | crystal | subsystem | session-runtime | **Scheduler** - The policy layer that reads a Table's Snapshot and selects the live session that should run next (StrictPriority or WeightedFair). |
| * | crystal | symbol | witness-proof | **World-state witness** - An external reference (commit hash, blob digest, etag, lease epoch) a cache entry is admitted under, so the entry can be refuted when that external state changes. |
| * | crystal | symbol | witness-proof | **Claim** - A worker or agent's assertion that it completed an effect (shipped a phase, created a file, ran a test), awaiting corroboration against independent evidence. |
| * | crystal | symbol | witness-proof | **WitnessResolver** - The component that corroborates claims against independent evidence sources (git history, filesystem, HTTP APIs) and returns Confirmed, Refuted, or Abstain. |
| * | crystal | symbol | witness-proof | **Refutation** - The local decision that a witness is invalid (its cached state is stale or poisoned), triggering eviction of all entries admitted under it. |
| * | crystal | symbol | witness-proof | **Revocation** - The broadcast event published to the coherence bus when a witness is refuted, enabling cross-agent causal invalidation of cache entries. |
| o | defined | symbol | attention | **AttnObserver** - A callback that receives post-softmax attention distributions one (layer, query, head) row at a time for span attribution. |
| o | defined | symbol | attention | **Attended (span field)** - The per-turn witnessed post-softmax attention mass a span accumulated during one forward pass, reset at the turn boundary. |
| o | defined | symbol | attention | **AttendedMass (method)** - The total witnessed attention mass currently attributed across all live segments - the denominator that normalizes each span's attention into [0,1]. |
| o | defined | symbol | attention | **AttentionAccumulator** - Folds per-turn per-span witnessed attention masses into a recency-decayed EMA and an undecayed cumulative sum, keeping a bounded trajectory ring per span. |
| o | defined | symbol | attention | **SpanAttention** - A read-only snapshot of one span's accumulated attention: EMA (recency-decayed), Cumulative (undecayed), and the bounded Trajectory ring. |
| o | defined | symbol | attention | **TurnMass** - One entry in a span's attention trajectory: the witnessed mass it drew in a specific (1-based, session-global) turn number. |
| o | defined | subsystem | attention | **AttentionIndex** - Payload-free metadata for a dynamic-sparse-attention (DSA) index artifact, recording binding axes and dependency graph for content-dependent key selection. |
| o | defined | symbol | attention | **AttentionIndexRequest** - The set of axes a DSA index lookup must match for reuse, with strict causal and exactness requirements for prefix matching. |
| o | defined | symbol | attention | **linearAttnCache** - The cache for Gated-DeltaNet linear-attention layers: accumulated recurrent state and a short-conv window per layer, not per-token K/V. |
| o | defined | symbol | attention | **AttributeRow (method)** - Routes one post-softmax attention row onto the segment ledger, attributing each key position's weight to the live span whose cache range owns that position. |
| o | defined | symbol | cache | **vBlock** - The unit of cacheable work in vCache: a cachemeta entry whose identity carries every axis that must match for provider reuse (content digest, model, tokenizer, endpoints, position in prefix DAG). |
| o | defined | symbol | cache | **KVLayout** - The interface abstracting the per-position bytes a given attention variant caches and how to reconstruct per-head K/V from those bytes (standard vs MLA). |
| o | defined | symbol | cache | **MLAConfig** - The low-rank MLA projection geometry (DeepSeek V2/V3): latent width, RoPE dim, and down/up projection matrices for compressing the K/V cache. |
| o | defined | concept | context-ctx | **compaction** - Provider prefix reuse on the wire: re-sending a byte-identical prefix so the provider serves it from its prompt cache instead of re-prefilling. |
| o | defined | subsystem | context-ctx | **ctxplan (context planner)** - The planner that forecasts the bounded resident working set of context spans for a session and learns from witnessed attention outcomes. |
| o | defined | subsystem | gateway-engine | **modelengine** - The binding that wires the in-kernel model into the kernel as a registered EngineDriver under the id 'inkernel'. |
| o | defined | subsystem | gateway-engine | **modelroute** - The model-routing policy spine: a version-tagged JSON manifest that classifies a tool call by aspect into a plan for which model(s) serve each aspect. |
| o | defined | subsystem | gateway-engine | **xenginekv** - The cross-engine zero-copy KV co-residence seam: a RegionBackend that maps an external engine's KV cache and fak's args/results into one shared addressable arena. |
| o | defined | subsystem | guard-gate | **normgate (result admitter)** - A ResultAdmitter that canonicalizes obfuscated payloads (base64, homoglyph, zero-width) before re-scanning for secrets. |
| o | defined | subsystem | guard-gate | **secretgate (result admitter)** - A ResultAdmitter that classifies discovered credentials as RESULT_SECRET_DISCOVERED events, distinct from injection poisoning. |
| o | defined | symbol | policy-capability | **Policy (loaded)** - The in-memory decision table compiled from a manifest, holding allow-lists, deny maps, arg predicates, posture, and secret patterns the adjudicator consults. |
| o | defined | subsystem | policy-capability | **preflight ladder** - The cheapest-first well-formedness rung ladder that catches malformed / unsafe calls before they fire, registered ahead of the authoritative monitor. |
| o | defined | symbol | session-runtime | **Budget** - A session's remaining work allotment across three independent axes: turns (round-trips), output tokens, and context tokens. |
| o | defined | symbol | session-runtime | **Pace** - The per-turn throttle: max output tokens and minimum gap between turns, applied cooperatively without pausing the session. |
| o | defined | symbol | session-runtime | **RunState** - A served session's lifecycle position in a small total state machine: Running, Throttled, Paused, Draining, or Stopped. |
| o | defined | symbol | witness-proof | **WitnessOutcome** - The result of claim corroboration: Confirmed (evidence supports it), Refuted (evidence contradicts it), or Abstain (no definitive evidence). |
| o | defined | symbol | witness-proof | **TrustEpoch** - A monotonically-increasing integrity clock that increments with each witness refutation, dual to WorldVersion (the consistency clock). |

## Per-KPI (disambiguation-debt = clarity of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| honesty | `kind_grounding_soft` | 76 | 0 | 4 kind/grounding mismatch |
| well-formed | `well_formed` | 100 | 0 | all 75 rows well-formed |
| distinctness | `canonical_unique` | 100 | 0 | every concept has a unique canonical name |
| distinctness | `defined` | 100 | 0 | every concept has a definition |
| distinctness | `disambiguated` | 100 | 0 | every confusable concept names what it is NOT |
| grounded | `grounded` | 100 | 0 | every concept's grounding token appears in the tree |
| grounded | `anchored` | 100 | 0 | every crystal concept's distinction is anchored on disk |
| honesty | `clarity_consistent` | 100 | 0 | every verdict matches its evidence |
| honesty | `hierarchy_soft` | 100 | 0 | hierarchy parents resolve |

## Coverage by family (how much of each confusable space is positioned)

| Family | Positioned | Discovered | Unpositioned |
|---|---:|---:|---:|
| gateway-engine | 9 | 108 | 99 |
| guard-gate | 7 | 63 | 56 |
| context-ctx | 4 | 59 | 55 |
| session-runtime | 4 | 56 | 52 |
| attention | 4 | 55 | 51 |
| cache | 15 | 62 | 47 |
| policy-capability | 5 | 52 | 47 |
| witness-proof | 6 | 35 | 29 |
| score-debt | 3 | 5 | 2 |

