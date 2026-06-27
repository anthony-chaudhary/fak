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
    Allow / Deny / Defer.
  - **result admitter** - a post-call gate: inspects a tool RESULT after execution and
    admits / quarantines / transforms it (ctxmmu, normgate, secretgate).
  - **git-hook gate** - a commit-boundary check at git pre-commit / commit-msg
    (`gate_brokenlink`, `gate_secretshape`, ...).
  - **promotion gate** - admits a cache entry to a shared tier by durability class
    (L3 promotion), distinct from **shipgate** which gates an RSI improvement to the
    codebase on witness-verified gain.

- **trunk guard** vs **repo guard** vs **gitgate** - branch-state policy (refuse
  OFF_TRUNK), write-target policy (refuse writes outside the tree), and git-command
  prefilter (refuse force-push / `--no-verify`). Three different "guards", three
  different surfaces.

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
