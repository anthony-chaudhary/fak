---
title: "fak concept-disambiguation scorecard - is every similar-sounding concept crystal-clear"
description: "Inward naming scorecard: each confusable fak concept positioned on the grounded / defined / disambiguated / anchored axes, with one clarity verdict per concept. Two driven numbers: coverage (of the confusable concept space discovered in the tree) and disambiguation-debt."
---

# Concept-disambiguation scorecard - crystal clarity across similar-sounding names

The sibling scorecards grade fak's code, docs, and competitive standing. This one asks the question that bites a reader as the system grows: **of the massive, growing set of similar-sounding names (cache, vCache, KV cache, cachemeta, the provider prompt-cache), is each distinct concept crystal-clear - one canonical name, a written definition, and an explicit line drawn against the siblings it is confused with?** Every number below is re-derived by `tools/concept_disambiguation_scorecard.py` and cross-checked against the real tree (the grounding token must appear in the production corpus; the glossary anchor must exist; a `distinct_from` reference must resolve). No verdict is hand-typed.

> Regenerate: `python tools/concept_disambiguation_scorecard.py --markdown-dir docs/concept-disambiguation-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Score** | **45.2/100** (grade F) = 4.5/10 |
| **Coverage** | **15.7%** (102/649 confusable tree tokens positioned) |
| **Disambiguation-debt** | **547** (clarity 0 + coverage 547) |
| Crystal-clear concepts | 104 of 143 positioned |
| As of | 2026-06-26 (fak v0.34.0) |

> **Read this right.** The score is deliberately LOW at birth: it grades the WHOLE confusable namespace discovered in the tree, not the few concepts already catalogued. A low coverage number is the honest statement that most similar-sounding names are not yet disambiguated - which is exactly the debt this scorecard exists to retire.

## Standing at a glance

```text
concept-disambiguation chart - 143 concepts - score 45.2/100 (grade F) - disambiguation-debt 547

clarity ladder (count of concepts, best -> fog):
  * crystal       ############################ 104
  o defined       ##########.................. 39
  ~ drifting      ............................ 0
  x colliding     ............................ 0
  . undocumented  ............................ 0

clarity mix by family (each cell = one concept):
  attention        oooooooooo         (10 concept(s); 0 crystal)
  cache            **************ooo  (17 concept(s); 14 crystal)
  context-ctx      ********ooooooo    (15 concept(s); 8 crystal)
  cross-cluster    **************     (14 concept(s); 14 crystal)
  decision         ****               (4 concept(s); 4 crystal)
  evict            ***                (3 concept(s); 3 crystal)
  gateway-engine   ******ooo          (9 concept(s); 6 crystal)
  guard-gate       ***********************oo (25 concept(s); 23 crystal)
  layout           ***                (3 concept(s); 3 crystal)
  plan             ***                (3 concept(s); 3 crystal)
  policy-capability ******oo           (8 concept(s); 6 crystal)
  pool             **                 (2 concept(s); 2 crystal)
  render-materialize ***                (3 concept(s); 3 crystal)
  score-debt       ***                (3 concept(s); 3 crystal)
  session-runtime  *******oooooooooo  (17 concept(s); 7 crystal)
  witness-proof    *****oo            (7 concept(s); 5 crystal)

coverage by family (positioned / discovered):
  plan             ##.......................... 7/107
  gateway-engine   ###......................... 10/105
  attention        ###......................... 5/55
  session-runtime  #######..................... 14/60
  render-materialize #........................... 2/47
  context-ctx      #######..................... 14/58
  cache            #######..................... 15/57
  guard-gate       ##########.................. 22/63
  policy-capability ###......................... 5/46
  witness-proof    ######...................... 6/30
  pool             ####........................ 2/14
  evict            ##.......................... 1/12
  layout           ######...................... 2/10
  decision         ####........................ 1/8
  score-debt       #################........... 3/5
  cross-cluster    ............................ 0/0

namespace coverage  [#####...........................] 15.7%  (102/649 confusable tokens positioned)

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
| * | crystal | concept | context-ctx | **compaction** - Provider prefix reuse on the wire: re-sending a byte-identical prefix so the provider serves it from its prompt cache instead of re-prefilling. |
| * | crystal | subsystem | context-ctx | **ctxplan (context planner)** - The planner that forecasts the bounded resident working set of context spans for a session and learns from witnessed attention outcomes. |
| * | crystal | subsystem | context-ctx | **contextq (materializer)** - On-demand context materializer: turns a search query into typed handles, materialization verdicts, omissions, and a render plan over CDB images. |
| * | crystal | symbol | context-ctx | **CtxViewPlanner** - Stateless, shared context-view planner wired to the gateway: one per server, shared across all requests, disabled by default. |
| * | crystal | symbol | context-ctx | **CompactionView** - A LOSSY compaction model: strips recovery handles off elided spans to show token savings without recoverability. |
| * | crystal | symbol | cross-cluster | **core-image manifest** - internal/recall's Manifest: the persisted core image of a finished session - the page table, the frozen quarantine-clearance state, and a frozen world-version marker, with the bytes themselves living in a sibling cas.json swap device. |
| * | crystal | symbol | cross-cluster | **policy manifest (on-disk JSON)** - internal/policy's Manifest: the on-disk capability-floor JSON an operator edits to configure the authorization intent - the abstract policy is compiled from this into an in-memory decision table. |
| * | crystal | symbol | cross-cluster | **session.Verdict** - internal/session's Verdict: what Decide returns to the agent turn loop at each boundary - Proceed (gate the next turn), MaxTokens (per-turn output cap), the drive State, Stop, and a closed Reason naming why the slot freed. |
| * | crystal | symbol | cross-cluster | **abi.Verdict (adjudication decision)** - internal/abi's Verdict: the discriminated-union adjudication decision type (Allow, Deny, Defer, Transform, Quarantine, RequireWitness) - the kernel's tool-call authorization decision surface. |
| * | crystal | symbol | cross-cluster | **compute backend** - internal/compute's Backend: the small whole-op HAL interface the forward loop targets (MatMul, RMSNorm, RoPE, Attention, NewKV ...), carrying a Tier capability probe and a CorrectnessClass; the CPU reference delegates to the model's exact arithmetic. |
| * | crystal | symbol | cross-cluster | **memq cell backend** - internal/memq's Backend: supplies memory cells (the page table as SAFE-metadata cells) and trust-gated byte access (Materialize pages one cell's bytes in through the trust gate); optional Tombstoner/Pruner add the durable mutations a backend chooses to support. |
| * | crystal | symbol | cross-cluster | **rung observer** - internal/rungobs's Observer: the passive rung-decision distribution counter - registered via abi.RegisterEmitter, it dedups by call SeqNo and histograms adjudication decisions (rung x kind x reason), read out with Snapshot. |
| * | crystal | symbol | cross-cluster | **cache-reuse observer** - internal/cacheobs's Observer: accumulates in-kernel KV-prefix reuse across served turns - prompt vs reused tokens, bucketed frozen/partial/cold by reuse ratio; the gateway scrapes the process-global Default. |
| * | crystal | symbol | cross-cluster | **planner candidate index** - internal/ctxplan's Index: the planner's candidate access path over a history store - an inverted token index, the append/recency order, and the durable set, maintained incrementally so a Probe returns a BOUNDED candidate set without rescanning all spans; metadata-only and deterministic. |
| * | crystal | symbol | cross-cluster | **simhash index** - internal/simhash's Index: an in-memory brute-force nearest-neighbor store - add (id, Vector, meta), then TopK(query, k) for the k most similar by cosine; a deterministic linear scan sized for a trajectory corpus of thousands of rows. |
| * | crystal | symbol | cross-cluster | **history-image store** - internal/ctxplan's Store: the history image the planner views - it supplies spans (SAFE metadata) and trust-gated byte access (Materialize is the gated page-in; a sealed span's bytes never cross the gate). A real deployment backs it with a recall core image or a memq backend. |
| * | crystal | symbol | cross-cluster | **blob CAS store** - internal/blob's Store: the content-addressed blob store (CAS) - digest->bytes with pin-aware bounded eviction (pinned digests, e.g. a vDSO tier-2 entry or a held quarantine handle, are never evicted), preserving the 'a cache hit equals a fresh call' invariant. |
| * | crystal | symbol | cross-cluster | **page-in refusal** - internal/ctxplan's Refusal: a selected span the trust gate declined to page in (sealed, or its bytes went missing) - reported to the caller, never rendered into the fresh history. |
| * | crystal | symbol | cross-cluster | **effect refusal** - internal/memq's Refusal: a cell an effect declined to touch (sealed / tombstoned / page-in refused) - recorded with a reason on the executor's result. |
| * | crystal | symbol | decision | **Decision (witness)** - Git evidence adjudication verdict with CONFIRMED/REFUTED/ABSTAIN labels |
| * | crystal | symbol | decision | **Decision (kernel)** - Tool-call verdict explanation trace showing why fak gave this verdict |
| * | crystal | symbol | decision | **Decision (scheduler)** - Loop admission advisory: whether to fire a scheduled loop now |
| * | crystal | symbol | decision | **Decision (shared-task)** - Shared-task execution state tracking and reconciliation record |
| * | crystal | symbol | evict | **evict (KV cache)** - Physical tensor span removal and RoPE re-rotation in KV cache for memory compaction |
| * | crystal | symbol | evict | **evict (playbook)** - Logical span pruning from rendered playbook under token budget |
| * | crystal | symbol | evict | **evict (session pool)** - Model instance eviction from a bounded LRU session pool |
| * | crystal | subsystem | gateway-engine | **kernel** - The fak core: the one implementation of abi.Kernel that coordinates adjudication, vDSO lookup, engine dispatch, and result admission across the tool-call path. |
| * | crystal | subsystem | gateway-engine | **gateway** - The kernel-adjudicated wire: the HTTP and MCP-RPC surface that fronts the kernel for non-Go clients, re-validating untrusted wire arguments before they reach it. |
| * | crystal | subsystem | gateway-engine | **engine** - The inference-engine seam (EngineDriver): the abstract backend interface the kernel dispatches allowed tool calls to (mock, HTTP upstream, fused in-kernel model). |
| * | crystal | subsystem | gateway-engine | **vDSO (tool vDSO)** - The tool vDSO: a local fast path (pure registry, content-addressed cache, static table) that answers a tool call with zero engine round-trip. |
| * | crystal | subsystem | gateway-engine | **model (in-kernel model)** - The in-kernel inference core: a pure-Go forward pass that runs chat token decode over a loaded GGUF checkpoint across several architectures and quant schemes. |
| * | crystal | symbol | gateway-engine | **engines registry** - The runtime registry (abi.Registry.engines) that maps engine IDs to their EngineDriver implementations: the kernel's dispatch table of all registered inference backends. |
| * | crystal | subsystem | guard-gate | **guard (fak guard kernel)** - The kernel itself: the in-process adjudication system that runs the decision chain and admits results, launched as `fak guard`. |
| * | crystal | concept | guard-gate | **gate (decision point)** - One decision point inside a guard, splitting by WHEN it fires: pre-call adjudicators, post-call result admitters, and git-hook gates. |
| * | crystal | symbol | guard-gate | **ResultAdmitter (post-call gate)** - A post-call gate: inspects a tool RESULT after execution and admits / quarantines / transforms it (ctxmmu, normgate, secretgate). |
| * | crystal | subsystem | guard-gate | **git-hook gate** - A commit-boundary check at git pre-commit / commit-msg (gate_brokenlink, gate_secretshape, gate_provenance, ...). |
| * | crystal | doc-term | guard-gate | **trunk guard** - A git hook that refuses commits made off the trunk (the OFF_TRUNK law), keeping all work on main. |
| * | crystal | subsystem | guard-gate | **repo guard (repoguard)** - A PreToolUse hook that refuses Bash/Write/Edit calls targeting paths outside the workspace tree. |
| * | crystal | subsystem | guard-gate | **gitgate (adjudicator)** - An adjudicator rung that prefilters git hazards (force-push, --no-verify, rebase -i) from shell tool calls at runtime. |
| * | crystal | subsystem | guard-gate | **shipgate (RSI promotion gate)** - An adjudicator that enforces RSI keep-or-revert verdicts on a code change based on witness-verified metric gain, never the candidate's own claim. |
| * | crystal | symbol | guard-gate | **StampGate (source-stamp result gate)** - A rank-20 ResultAdmitter that stamps every tool result's taint by SOURCE (trusted-local vs untrusted-egress) and clamps the result's ShareScope DOWNWARD. |
| * | crystal | symbol | guard-gate | **ScopeCeilingGate (scope-ceiling result gate)** - The result-side ShareScope CEILING gate (rank 21, above StampGate): confines cross-agent taint visibility to the declared scope boundary - the upward dual of StampGate's downward clamp. |
| * | crystal | symbol | guard-gate | **SinkGate (egress adjudicator)** - The egress adjudicator that DENIES a tool call whose arguments carry untrusted taint into a configured sink, governed by a GatedSinks policy. |
| * | crystal | doc-term | guard-gate | **sealed_by_trust_gate (trust-gate refusal)** - A trust-gate REFUSAL REASON code: a context page that is sealed/tombstoned cannot be demand-paged back in. |
| * | crystal | config | guard-gate | **StrictGatedSinks (strict egress-sink policy)** - The Policy preset that gates ALL egress sinks strictly: the GatedSinks configuration the SinkGate adjudicator enforces. |
| * | crystal | symbol | guard-gate | **mlp.gate_proj (FFN gate projection)** - The FFN/SwiGLU GATE projection weight in an MLP layer; after SiLU it is multiplied with up_proj, distinct from down_proj. |
| * | crystal | symbol | guard-gate | **q_gate_proj (linear-attn query gate)** - Qwen3.5 linear-attention QUERY gating projection (self_attn.q_gate_proj) in Gated-DeltaNet layers; the in_proj_z source absorbs it. |
| * | crystal | symbol | guard-gate | **gate_up_proj (fused SwiGLU gate+up)** - The FUSED gate+up projection (mlp.gate_up_proj.weight): gate_proj and up_proj concatenated into one tensor the loader splits. |
| * | crystal | symbol | guard-gate | **ffn_gate (GGUF FFN gate tensor)** - The GGUF tensor name for the FFN gate weight (ffn_gate.weight), canonicalized to mlp.gate_proj.weight on load. |
| * | crystal | config | guard-gate | **AttnOutputGate (attention output-gate flag)** - A Qwen3.5 CONFIG flag (attn_output_gate) enabling a sigmoid GATE on full-attention output logits. |
| * | crystal | symbol | guard-gate | **rmsNormGatedInPlace (gated RMSNorm)** - The Qwen3.5 gated-RMSNorm compute (Qwen3_5RMSNormGated): x = weight * rmsnorm(x) * silu(gate), applied in place. |
| * | crystal | symbol | guard-gate | **FoldedGate (KV-MMU result gate)** - The kvmmu gate that enforces the kernel's REGISTERED ResultAdmitter chain (most-restrictive-wins fold) over results before admitting them to the KV cache. |
| * | crystal | config | guard-gate | **DefaultGatedSinks (default egress-sink policy)** - The reasonable default GatedSinks configuration: gates EGRESS (exfiltration) and DESTRUCTIVE (irreversible mutation) sinks on session taint, but NOT EXEC (preserves dev work). |
| * | crystal | symbol | guard-gate | **ContractGate (contract validation gate)** - A named validation checkpoint in an official run contract (browseraction/terminalbench/toolsandbox) that gates promotion: candidate_task_ids, official_harness_pin, same_task_ids_required, etc. |
| * | crystal | concept | guard-gate | **guardrail (safety boundary)** - A safety boundary or constraint that prevents an AI system from taking harmful actions; often implemented as policy checks or refusal reasons. |
| * | crystal | symbol | layout | **Layout (tensor)** - Tensor element physical arrangement: RowMajor, ColMajor, or other ordering |
| * | crystal | symbol | layout | **Layout (ctxplan)** - Base/Current/Recent/Deep region profile for layout-aware planning |
| * | crystal | symbol | layout | **MLA KV layout seam** - Attention cache variant seam interface: standardKVLayout vs mlaKVLayout |
| * | crystal | symbol | plan | **Plan (planner)** - Planner's chosen resident view: selected set, elided set, and accounting |
| * | crystal | symbol | plan | **Plan (memq)** - Static pre-execution Explain output: pipeline steps, effects, and mutations |
| * | crystal | symbol | plan | **Candidate** - Scored span the planner may keep resident with cost, benefit, and density metrics |
| * | crystal | concept | policy-capability | **capability floor** - The deployable, declarative authorization layer that defines exactly which tools an agent may call by name, loaded as a JSON manifest at runtime. |
| * | crystal | config | policy-capability | **policy manifest** - The versioned JSON structure an operator edits to configure the capability floor, mapping 1:1 to adjudicator.Policy with deny reasons validated against the closed vocabulary. |
| * | crystal | subsystem | policy-capability | **adjudicator** - The in-process DOS reference monitor: the rung that folds a decision chain to prove a tool call allowed, denied, or deferred under the loaded policy. |
| * | crystal | symbol | policy-capability | **abi.Verdict** - The discriminated-union decision an adjudicator returns, keyed by kind (Allow, Deny, Defer, Transform, Quarantine, RequireWitness) with typed payloads. |
| * | crystal | symbol | policy-capability | **reason code** - The closed, additive vocabulary of refusal reasons (DEFAULT_DENY, POLICY_BLOCK, SELF_MODIFY, ...) that every deny verdict cites, never free text. |
| * | crystal | config | policy-capability | **posture (tool admission)** - The policy's default behavior after all provable refusal checks pass: PostureFailClosed (deny everything not allowed) or PostureAdmitAndLog (admit low-risk reads with forensic metadata). |
| * | crystal | symbol | pool | **Pool (session)** - Bounded-LRU session state container with a fixed ceiling on concurrent sessions |
| * | crystal | symbol | pool | **PoolProfile** - Pooling character of a residency tier describing host count, coherence model, and shareability |
| * | crystal | symbol | render-materialize | **RenderPlan** - Prompt-assembly layout: stable prefix of reused views plus volatile tail of raw faults |
| * | crystal | symbol | render-materialize | **RenderItem** - One cell materialized into context by OpRender query effect |
| * | crystal | symbol | render-materialize | **Rendered** - One span paged into fresh history through the trust gate |
| * | crystal | subsystem | score-debt | **scorecard** - One deterministic measurement of a surface that folds reality into a single *_debt integer plus an A-F grade (the family is documented in the scorecard skill). |
| * | crystal | subsystem | score-debt | **scorecard control pane** - The fold that sums every scorecard's *_debt into one portfolio number with a pinned ratchet that reds only on a regression above baseline. |
| * | crystal | metric | score-debt | **disambiguation-debt** - This scorecard's integer: clarity defects of positioned concepts plus coverage gaps (confusable tree tokens with no row). |
| * | crystal | subsystem | session-runtime | **Session** - The full drive record for one served run (run-state, budget, priority, pace), keyed by TraceID and persisting across turns. |
| * | crystal | concept | session-runtime | **Turn** - One model round-trip within a session: the agent submits input, the model generates output, and results are admitted to context. |
| * | crystal | symbol | session-runtime | **Slot** - The immutable free/busy signal emitted when a session leaves the eligible set (budget exhaustion, pause, drain, stop), freeing scheduling capacity. |
| * | crystal | subsystem | session-runtime | **Scheduler** - The policy layer that reads a Table's Snapshot and selects the live session that should run next (StrictPriority or WeightedFair). |
| * | crystal | symbol | session-runtime | **session.State** - Live, mutable per-session control record: run-state, budget, priority, pace, and revision (an optimistic-concurrency guard). |
| * | crystal | subsystem | session-runtime | **sessionimage.Image** - Loaded, integrity-verified portable session archive: drive (session.json), recall manifest, ctxplan index, trajectory corpus, and image.json meta. |
| * | crystal | symbol | session-runtime | **SessionPlanner** - Persistent per-session context planner: a long-lived lossless store plus candidate index that ingests each turn's new messages incrementally. |
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
| o | defined | symbol | context-ctx | **SpeculationContext** - Transient state for brainstorm epochs: (Speculative, Epoch, ParentEpoch) marking provisional effects produced under speculation. |
| o | defined | symbol | context-ctx | **ContextChangeRequest** - Negative-only context mutation request: one persisted recall-page tombstone with reason, requestor, and witness. |
| o | defined | symbol | context-ctx | **RecallProof** - Self-describing proof of a recall cost calculation: shows the gate's work (PrefixTokens, UnitTokens, ReadMult, siblings, cost breakdown). |
| o | defined | symbol | context-ctx | **ProveRecall** - Pure function that runs the recall cost gate over its inputs and returns a RecallProof showing the gate's work. |
| o | defined | concept | context-ctx | **CtxView** - A planned, budgeted view of context the CtxViewPlanner selects for one turn within a token budget. |
| o | defined | concept | context-ctx | **context window** - The model's hard token budget that the planner allocates resident spans within. |
| o | defined | symbol | context-ctx | **ContextChange (apply)** - The applied negative mutation: tombstones a recall page through the trust gate per a ContextChangeRequest. |
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
| o | defined | symbol | session-runtime | **SessionID** - The stable identity key for a session (its TraceID), constant across all of the session's turns. |
| o | defined | symbol | session-runtime | **NewSession** - The constructor that admits a fresh session into the table with its initial drive state. |
| o | defined | symbol | session-runtime | **SessionTurn** - One accounted served round-trip bound to a session: the begin / debit / served lifecycle of a single turn. |
| o | defined | symbol | session-runtime | **SessionUsage** - Per-session token and cost accounting rolled up across the session's turns. |
| o | defined | symbol | session-runtime | **SessionControlRequest** - An external steer directive against a live session: pause / resume / priority / pace change. |
| o | defined | symbol | session-runtime | **SessionReset** - A directive to clear a session's drive state back to a clean baseline without removing it from the pool. |
| o | defined | symbol | session-runtime | **ListSessions** - The read query that returns a snapshot of all live sessions in the table. |
| o | defined | symbol | witness-proof | **WitnessOutcome** - The result of claim corroboration: Confirmed (evidence supports it), Refuted (evidence contradicts it), or Abstain (no definitive evidence). |
| o | defined | symbol | witness-proof | **TrustEpoch** - A monotonically-increasing integrity clock that increments with each witness refutation, dual to WorldVersion (the consistency clock). |

## Per-KPI (disambiguation-debt = clarity of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| honesty | `kind_grounding_soft` | 76 | 0 | 4 kind/grounding mismatch |
| well-formed | `well_formed` | 100 | 0 | all 143 rows well-formed |
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
| plan | 7 | 107 | 100 |
| gateway-engine | 10 | 105 | 95 |
| attention | 5 | 55 | 50 |
| session-runtime | 14 | 60 | 46 |
| render-materialize | 2 | 47 | 45 |
| context-ctx | 14 | 58 | 44 |
| cache | 15 | 57 | 42 |
| guard-gate | 22 | 63 | 41 |
| policy-capability | 5 | 46 | 41 |
| witness-proof | 6 | 30 | 24 |
| pool | 2 | 14 | 12 |
| evict | 1 | 12 | 11 |
| layout | 2 | 10 | 8 |
| decision | 1 | 8 | 7 |
| score-debt | 3 | 5 | 2 |
| cross-cluster | 0 | 0 | 0 |

