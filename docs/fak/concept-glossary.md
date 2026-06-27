# fak concept glossary - drawing the line between similar-sounding names

fak has grown a large vocabulary, and several roots are badly overloaded. The word
"cache" alone names at least a dozen distinct things; "gate" and "guard" blur into
each other; "witness" means two unrelated ideas in two subsystems. This page is the
single place those lines are drawn. It is the anchor the concept-disambiguation
scorecard points a concept at when it claims to be crystal-clear
(`tools/concept_disambiguation_scorecard.py`).

The rule for an entry: one canonical name, one sentence on what it IS, and one
sentence on what it is NOT (the sibling it is most confused with). When a concept is
not yet in here, the scorecard counts it as coverage debt.

---

## The cache family

The single most overloaded root. The fix is to think in PLANES, not in "the cache".
Four planes, each a different question:

| Plane | Question it answers | Canonical name |
|---|---|---|
| Storage | where do the raw attention tensors live? | KV cache |
| Virtualization | how do I model a cache I do not own? | vCache |
| Metadata | what names a reusable entry and proves it valid? | cachemeta |
| Provider-observed | what did the upstream report it cached? | Provider cache |

- **KV cache** - the kernel-owned raw attention state: per-position Key and Value
  tensor rows for the running model, supporting in-place eviction and prefix reuse.
  *Not* vCache (that is a control plane over a REMOTE cache) and *not* cachemeta
  (that owns no tensors).

- **vCache** - the virtual API cache: a page-table abstraction that models a remote
  provider's prefix cache as virtual pages, with a manifest of canonical prefix
  chains and warmth belief. It is a CONTROL PLANE over a cache you do not own. *Not*
  the KV cache (local raw tensors) and *not* the provider's prompt cache itself
  (vCache is the thing you build to use that cache well).

- **cachemeta** - the typed metadata contract (tier 1): it owns no payloads, it names
  reusable cache entries and carries their validity / security / residency metadata
  and typed lookup verdicts. Every other cache plane builds on it. *Not* vCache (the
  active control loop built ATOP cachemeta).

- **Prompt cache** - the upstream provider feature (e.g. Anthropic): a prefix cached
  via `cache_control` breakpoints, reported back as `cache_read_input_tokens` /
  `cache_creation_input_tokens` with a specific pricing multiplier. It is a feature
  you USE, not code you own. *Not* vCache (the control loop) and *not* the KV cache
  (local tensors).

- **Provider cache** - a cachemeta entry on `plane=provider`: the OBSERVED telemetry
  record of what the provider's prompt cache did (read/creation token counts), marked
  non-re-serveable local proof. *Not* the Prompt cache feature itself (this is the
  recorded observation of it), and *not* a local cache fak can serve from.

- **cache_control** vs **cache_read** vs **cache_creation_input_tokens** - the WRITE
  placement mechanism, the READ telemetry, and the WRITE telemetry, respectively.
  `cache_control` is the breakpoint you place; `cache_read` is what the provider
  reports it served from cache; `cache_creation_input_tokens` is what it reports it
  wrote to cache.

- **RadixKV** - a local token-trie data structure for fast prefix lookup that points
  INTO KV-cache spans, with materialization binding so cross-model spans are never
  reused. *Not* the KV cache (the tensor storage it indexes).

- **enginecache** - the adapter that translates cachemeta invalidation directives into
  a remote serving engine's control API (SGLang / vLLM prefix-cache reset or span
  evict). *Not* cachemeta (the pure contract) and *not* vCache (the policy that may
  trigger it).

- **ViewCache** vs **MemoryView** - ViewCache STORES materialized lossy projections
  (summaries, QA, facts) over canonical pages; MemoryView is the typed CONTRACT
  binding a projection to its canonical source by digest + span. Storage vs contract.

- **Hardware-aware cache** / **KV transfer** - the placement POLICY that knows each
  tier's physical character and the migration DIRECTIVE it emits to move a span
  between tiers. Policy vs directive, both distinct from the KV cache (the storage).

---

## The guard / gate family

- **guard** - the kernel itself: the in-process adjudication system that runs the
  decision chain and admits results (`fak guard`). A guard is a SYSTEM.

- **gate** - one decision point INSIDE a guard. A gate is a POINT, not the system.
  The gates split by WHEN they fire:
  - **adjudicator** - a pre-call gate: inspects a tool call BEFORE dispatch, returns
    Allow / Deny / Defer (e.g. `residencyGate`, the rank-12 engine-residency
    adjudicator registered in `internal/engine`).
  - **result admitter** - a post-call gate: inspects a tool RESULT after execution and
    admits / quarantines / transforms it (ctxmmu, normgate, secretgate).
  - **git-hook gate** - a commit-boundary check at git pre-commit / commit-msg
    (`gate_brokenlink`, `gate_secretshape`, and the `internal/hooks` family
    `gateFileAdmission` / `gateProvenanceLabel` / `gatePublicLeak`).
  - **promotion gate** - admits a cache entry to a shared tier by durability class
    (L3 promotion), distinct from **shipgate** which gates an RSI improvement to the
    codebase on witness-verified gain.
  - **capability-floor gate** - a per-message floor on inter-agent channels
    (`gateSend` / `gateRecv` in `internal/a2achan`): refuses a Send/Recv whose caps
    do not advertise the channel right. A floor on a MESSAGE - NOT a tool-call
    adjudicator and NOT a result admitter.

- **trunk guard** vs **repo guard** vs **gitgate** - branch-state policy (refuse
  OFF_TRUNK), write-target policy (refuse writes outside the tree), and git-command
  prefilter (refuse force-push / `--no-verify`). Three different "guards", three
  different surfaces.

### adjudication gate vs model gate - the headline collision

The word **gate** names two COMPLETELY UNRELATED things in this repo. They share only
the spelling; nothing in the kernel's safety layer touches the model's tensors.

- **adjudication gate** (CONTROL PLANE) - a decision point in the safety layer that
  ALLOWS / DENIES / TRANSFORMS a tool call or its result. All of the gates above are
  adjudication gates. The data-plane result gates and the egress adjudicator:
  - **StampGate** - a rank-20 result admitter that stamps every result's taint by
    SOURCE (trusted-local vs untrusted-egress) and clamps its ShareScope DOWNWARD.
  - **ScopeCeilingGate** - the rank-21 result-side ceiling (the upward dual of
    StampGate): confines cross-agent taint visibility to the declared scope boundary.
  - **SinkGate** - the pre-dispatch egress adjudicator: DENIES a call whose arguments
    carry untrusted taint into a sink, per a **StrictGatedSinks** policy preset.
  - **sealed_by_trust_gate** - a refusal REASON code, not a gate type: a sealed /
    tombstoned context page cannot be demand-paged back in.

- **model gate** (NEURAL NET) - a weight projection or tensor computation that gates
  activations inside the forward pass. NOTHING to do with adjudication; it never sees
  a tool call. The model-gating tokens:
  - **mlp.gate_proj** - the FFN/SwiGLU gate projection weight (after SiLU, multiplied
    with `up_proj`); **ffn_gate** is its GGUF spelling, canonicalized to it on load.
  - **gate_up_proj** - the FUSED gate+up weight (`mlp.gate_up_proj.weight`) the loader
    splits back into `gate_proj` and `up_proj`.
  - **q_gate_proj** - Qwen3.5 linear-attention query gating weight in Gated-DeltaNet
    layers (`self_attn.q_gate_proj`).
  - **block_sparse_moe.gate** - the MoE router gate: the expert-selection routing
    weights in sparse mixture-of-experts blocks.
  - **AttnOutputGate** - a config flag enabling a sigmoid gate on attention output
    logits; **rmsNormGatedInPlace** is the gated-RMSNorm compute (`x = w * rmsnorm(x) *
    silu(gate)`), a COMPUTE that consumes a gate, not a weight.

  Rule of thumb: if it decides about a tool call or result it is an **adjudication
  gate**; if it lives in a `.weight` tensor name or the forward pass it is a **model
  gate**. The inflections (`gated`, `gates`, `guards`, `guarded`) are grammar, not
  concepts - the scorecard ignores them.

---

## The witness / evidence family

- **world-state witness** - an external reference (commit hash, blob digest, etag,
  lease epoch) that a cache entry is admitted under, so the entry can be refuted when
  that external state changes. Lives in `internal/vdso`.

- **measurement witness** - an RSI validation artifact proving a candidate improvement
  was real (a metric gain confirmed independently). Unrelated to the cache witness
  beyond the shared word.

- **Claim** vs **WitnessResolver / WitnessOutcome** - a Claim is a worker's SELF-REPORT
  of an effect; the WitnessResolver corroborates it against independent evidence and
  returns a WitnessOutcome (Confirmed / Refuted / Abstain). Self-report vs
  corroboration.

- **Refutation** vs **Revocation** - refutation is the LOCAL decision that a witness is
  invalid; revocation is the BROADCAST event other agents consume. Decision vs
  broadcast.

---

## The session / scheduling family

- **Session** - the full drive record for one served run (run-state, budget, priority,
  pace), keyed by TraceID. **Turn** - one model round-trip within a session. **Slot** -
  the free/busy SIGNAL emitted when a session leaves the eligible set. Record vs
  round-trip vs signal.

- **Table** vs **Snapshot** vs **Scheduler** - Table is the mutable per-session store;
  Snapshot is the read-only sorted copy it returns; Scheduler reads a Snapshot and
  picks the next winner. Store vs copy vs policy.

- **session.Verdict** vs **abi.Verdict** - the per-turn boundary decision
  (Proceed / Stop) vs the kernel adjudication decision (Allow / Deny / Defer). Same
  word, two layers.

- **session.State** vs **sessionimage.Image** - session.State is the LIVE, mutable
  per-session control record (run-state, budget, priority, pace, revision);
  sessionimage.Image is the PERSISTED, integrity-verified archive bundling the drive
  (session.json), the recall manifest, the ctxplan index, and the trajectory corpus.
  Live drive record vs persisted archive.

- **SessionPlanner** vs **session.State** - SessionPlanner holds the per-session
  CONTEXT-PLANNER state (a long-lived lossless store plus candidate index); session.State
  holds the per-session RUN-CONTROL state (run-state / budget / pace). Context planning
  vs run control.

---

## The gateway / engine family

- **kernel** - the central coordinator of the whole tool-call path (adjudicate ->
  vDSO -> dispatch -> admit). **gateway** - the WIRE: the HTTP / MCP surface that
  fronts the kernel for non-Go clients. **engine** - the dispatch SEAM the kernel
  sends allowed calls to. **vDSO** - the local fast path that answers without an engine
  round-trip. **serve** - the CLI command that wires kernel + gateway + engine
  together. Coordinator vs wire vs seam vs fast-path vs launcher.

- **model** vs **modelengine** vs **compute** - the in-kernel forward-pass algorithm,
  the binding that registers it as an engine backend, and the device HAL it runs
  tensor ops on. Algorithm vs registration vs device.

- **engines registry** vs **engine** - the runtime dispatch table (abi.Registry.engines)
  that maps engine IDs to EngineDriver instances, versus the abstract EngineDriver
  interface itself. Table vs contract.

- **engines registry** vs **engine** - the runtime dispatch table (abi.Registry.engines)
  that maps engine IDs to EngineDriver instances, versus the abstract EngineDriver
  interface itself. Table vs contract.

---

## The policy / authorization family

- **capability floor** vs **policy manifest** vs **Policy (loaded)** - the abstract
  authorization intent, its on-disk JSON representation, and the compiled in-memory
  decision table. Intent vs file vs compiled form.

- **adjudicator** vs **verdict** vs **reason code** - the enforcer, the decision it
  returns (Allow / Deny / Defer / Transform / Quarantine), and the closed-vocabulary
  WHY a deny cites. Enforcer vs decision vs reason.

- **DEFAULT_DENY** vs **POLICY_BLOCK** - the fail-closed outcome when nothing
  affirmatively allowed a call vs an explicit deny-rule match. Both are deny reason
  codes; the distinction is implicit-vs-explicit.

- **posture** vs **secret posture** - the default-deny behavior on the call-admit path
  vs the behavior when a RESULT bears a credential. Orthogonal knobs.

---

## The context-management family

- **context-MMU (ctxmmu)** vs **KV-MMU (kvmmu)** - ctxmmu gates RESULT BYTES on the
  text side (admit / quarantine / page-out); kvmmu turns that logical verdict into a
  mechanical one by evicting K/V spans on the attention side. Same trust decision,
  two layers.

- **recall** vs **compaction** - recall is the persisted session core-dump (a page
  table over a content-addressed swap device); compaction is provider prefix reuse on
  the wire. Persistence vs reuse - unrelated beyond both touching "context".

- **contextq** vs **ctxplan** - contextq is the on-demand MATERIALIZER: it turns a
  search query into typed handles, materialization verdicts, omissions, and a render
  plan over CDB images. ctxplan is the OPTIMIZER: a bounded-candidate planner that
  forecasts which spans keep resident under a token budget. Materializer vs optimizer -
  one fetches the spans, the other chooses which stay.

- **memq** vs **contextq** - memq is the general memory-operation ALGEBRA (a pipeline
  of scan / filter / rank / limit / budget / render / tombstone / consolidate ops over
  typed cells); contextq is ONE concrete materialization expressed through that algebra.
  Algebra vs operation.

- **CtxViewPlanner** vs **SessionPlanner** - CtxViewPlanner is the STATELESS shared
  seam (one per server, shared across every request, off by default); SessionPlanner is
  the STATEFUL per-session planner (a long-lived lossless store plus candidate index
  that ingests each turn incrementally). Stateless-shared vs stateful-per-session
  (SessionPlanner also appears under the session family below).

- **CompactionView** vs **compaction** - CompactionView is the LOSSY savings MODEL: it
  strips recovery handles off elided spans to show token savings without recoverability;
  compaction is provider prefix reuse on the wire. Savings model vs wire reuse.

---

## The scorecard / debt family

- **scorecard** - one deterministic measurement of a surface that folds into a single
  `*_debt` integer (the family is documented in the `scorecard` skill). **control
  pane** - the fold that sums every `*_debt` into one portfolio number with a pinned
  ratchet. Measurement vs fold.

- **disambiguation-debt** (this scorecard) vs **conflation-debt** - naming clarity
  (distinct names for distinct concepts) vs provenance honesty (a reported number
  labeled WITNESSED vs OBSERVED). Names vs numbers - two different honesty axes that
  are themselves easy to confuse.

---

## The eviction family

- **evict (KV cache)** - physical tensor span removal with RoPE re-rotation for memory
  compaction in the attention cache. *Not* playbook pruning (that is logical
  span removal, not tensor compaction).

- **evict (playbook)** - logical span pruning from the rendered playbook under token
  budget, returning the evicted bullets for legibility. *Not* KV cache eviction
  (that is physical tensor compaction, not logical pruning).

- **evict (session pool)** - model instance eviction from a bounded LRU session pool
  to stay within budget. *Not* playbook pruning (that removes context spans, not
  entire sessions).

---

## The decision family

- **Decision (witness)** - git evidence adjudication verdict with CONFIRMED/REFUTED/ABSTAIN
  labels recorded in git notes. *Not* kernel Decision (that is an explanation trace).

- **Decision (kernel)** - tool-call verdict explanation trace showing why fak gave this
  verdict, including the args digest and adjudication chain. *Not* witness Decision
  (that is a stored adjudication verdict, not a live trace).

- **Decision (scheduler)** - loop admission advisory returning whether to fire a scheduled
  loop now with an admit boolean and reason. *Not* kernel Decision (that explains a
  tool-call verdict, not loop admission).

- **Decision (shared-task)** - shared-task execution state tracking and reconciliation
  record with a decision ID and state machine transitions. *Not* scheduler Decision
  (that advises on loop firing, not task reconciliation).

---

## The render / materialize family

- **RenderPlan** - prompt-assembly layout: a stable prefix of reused views plus a volatile
  tail of raw faults. *Not* RenderItem (that is one cell materialized, not the whole
  layout).

- **RenderItem** - one cell materialized into context by OpRender query effect, carrying
  its span and cache entry binding. *Not* Rendered (that is a ctxplan span paged through
  trust gate, not a memq effect).

- **Rendered** - one span paged into fresh history through the trust gate. *Not* RenderItem
  (that is a memq materialization effect, not a ctxplan trust-gate result).

---

## The plan family

- **Plan (planner)** - the planner's chosen resident view: selected set, elided set, and
  accounting. *Not* Plan (memq) (that is a static pre-execution explain output).

- **Plan (memq)** - static pre-execution Explain output: pipeline steps, effects, and
  mutations. *Not* Plan (planner) (that is a resident view selection, not a query plan).

- **Candidate** - a scored span the planner may keep resident with cost, benefit, and
  density metrics. *Not* Plan (the planner's output selection).

---

## The pool family

- **Pool (session)** - bounded-LRU session state container with a fixed ceiling on
  concurrent sessions. *Not* PoolProfile (that describes tier pooling character, not
  the container itself).

- **PoolProfile** - pooling character of a residency tier describing host count,
  coherence model, and shareability. *Not* Pool (that is the container itself, not its
  profile description).

---

## The layout family

- **Layout (tensor)** - tensor element physical arrangement: RowMajor, ColMajor, or other
  ordering carried on every Tensor. *Not* Layout (ctxplan) (that is a region profile for
  planning, not tensor storage order).

- **Layout (ctxplan)** - Base/Current/Recent/Deep region profile for layout-aware planning.
  *Not* Layout (tensor) (that is tensor storage order, not a planning profile).

- **MLA KV layout seam** - attention cache variant seam interface: standardKVLayout vs mlaKVLayout.
  *Not* Layout (tensor) (that is element ordering, not cache variant).

---

## Cross-cluster canonical-name collisions

The worst confusability is one TOKEN that names two unrelated things in two unrelated
packages: a reader meets the name in package A, builds a mental model, then meets it in
package B and is already wrong. These seven are not renamed (each is the natural local
name in its own package), so the line is drawn here instead - one sentence on what each
IS and what it is NOT. The concept-disambiguation scorecard positions both halves as a
`cross-cluster` pair that cite each other in `distinct_from`.

- **core-image manifest** (`recall.Manifest`) - the persisted core image of a finished
  session: page table, frozen quarantine clearance, and a frozen world-version marker.
  *Not* the **policy manifest** (`policy` package), which is the on-disk capability-floor
  JSON; same token, a session snapshot vs an authorization config.

- **policy manifest** (`policy` package) - the on-disk JSON an operator edits to configure
  the capability floor. *Not* `recall.Manifest` (the session core-image snapshot).

- **session.Verdict** (`session.Decide`) - the per-turn loop gate: Proceed, MaxTokens, the
  drive State, Stop, and a closed Reason for why the slot freed. *Not* **abi.Verdict** (the
  discriminated-union adjudication decision); same token, a turn-boundary control value vs
  a tool-call authorization decision.

- **abi.Verdict** (`abi` package) - the adjudicator's discriminated-union decision (Allow,
  Deny, Defer, Transform, Quarantine, RequireWitness). *Not* `session.Verdict` (the per-turn
  boundary gate).

- **compute backend** (`compute.Backend`) - the small whole-op HAL interface the forward
  loop targets (MatMul, RMSNorm, RoPE, Attention, NewKV). *Not* the **memq cell backend**
  (`memq.Backend`), which supplies memory cells and trust-gated byte access; same token, a
  tensor-op surface vs a cell/page-in source.

- **memq cell backend** (`memq.Backend`) - supplies memory cells (the page table as safe
  metadata) and trust-gated byte access (Materialize). *Not* `compute.Backend` (the
  model-math HAL interface).

- **rung observer** (`rungobs.Observer`) - the passive rung-decision distribution counter:
  histograms adjudication decisions by rung x kind x reason. *Not* the **cache-reuse
  observer** (`cacheobs.Observer`), which accumulates KV-prefix reuse; same token, the
  verdict mix vs prefill reuse.

- **cache-reuse observer** (`cacheobs.Observer`) - accumulates in-kernel KV-prefix reuse
  across served turns, bucketed frozen/partial/cold. *Not* `rungobs.Observer` (the
  decision-distribution counter).

- **planner candidate index** (`ctxplan.Index`) - the planner's candidate access path over
  a history store: an inverted token index plus recency tail and durable set, so a Probe
  returns a bounded candidate set. *Not* the **simhash index** (`simhash.Index`), the
  brute-force nearest-neighbor vector store; same token, an inverted span index vs a
  cosine-similarity corpus.

- **simhash index** (`simhash.Index`) - an in-memory brute-force nearest-neighbor store
  (add a Vector, then TopK by cosine). *Not* `ctxplan.Index` (the planner's inverted
  candidate access path).

- **history-image store** (`ctxplan.Store`) - the trust-gated history image the planner
  views: it supplies spans and gated byte access (Materialize). *Not* the **blob CAS store**
  (`blob.Store`), the content-addressed digest->bytes store; same token, a gated span view
  vs a content store.

- **blob CAS store** (`blob.Store`) - the content-addressed blob store: digest->bytes with
  pin-aware bounded eviction. *Not* `ctxplan.Store` (the planner's gated span interface).

- **page-in refusal** (`ctxplan.Refusal`) - a selected span the trust gate declined to page
  in (sealed, or its bytes went missing), reported but never rendered. *Not* the **effect
  refusal** (`memq.Refusal`), a memory cell an effect declined to touch; same token, a
  context span vs a memory cell.

- **effect refusal** (`memq.Refusal`) - a cell an effect declined to touch (sealed /
  tombstoned / page-in refused), recorded with a reason. *Not* `ctxplan.Refusal` (a planner
  span the gate declined to page in).
