---
title: "fak concept glossary — disambiguating overloaded names"
description: "The single place fak draws the line between similar-sounding names — cache, gate, guard, witness — so overloaded vocabulary stops causing confusion."
---

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

### cache anchor vs compaction budget - the knobs that confuse everyone

These two are the pair the goal behind this page keeps re-confusing. They shape the
SAME outbound `/v1/messages` body that `fak guard -- claude` forwards, but they answer
two orthogonal questions - and the anchor GATES the budget, so setting one without the
other in mind produces the "I lowered the budget and nothing changed" surprise.

- **cache anchor** (the `cache_control` breakpoint) - WHERE the cached prefix ENDS.
  It is a positional WRITE mark (prefix order tools -> system -> messages): the provider
  caches everything BEFORE it, and compaction copies everything THROUGH it verbatim so
  that cached prefix survives byte-for-byte. It is a boundary, and it says nothing about
  size. *Not* the budget (an anchor never caps how many tokens ride along).

- **compaction budget** (`--compact-history-budget`, default 48000) - a resident-token
  TARGET for the span AFTER the anchor. Once that un-cached middle sprawls past the
  budget, whole old turns are dropped down to it. It is a size threshold on the
  compactible tail, *not* a cache boundary (it never moves or crosses the anchor).

- **anchor-starved** (the trap, #1407) - because the budget can only shed the span
  AFTER the anchor, an anchor that lands LATE leaves nothing for the budget to touch.
  Real Claude Code traffic marks its breakpoint on a RECENT turn, so the default anchor
  (`CompactAnchorFirstBP`) sits near the end, the protected prefix swallows almost the
  whole conversation, and **lowering the budget does nothing** - compaction can never
  fire. The `AnchorStarved` diagnostic names exactly this state (under-budget WITH a
  protected prefix already past the budget). The lever here is the ANCHOR, not the
  budget: `CompactAnchorHead` re-anchors on the stable system/tools head so the whole
  message array becomes compactible - but that bursts the recent breakpoint's cached
  suffix, so it fires only when the burst repays within the session horizon
  (`CacheBurstPaysBack`, #1408). Grounded in `internal/agent/anthropic_compact.go`.

- **compact-history-budget** vs **ctx-view-budget** - two DIFFERENT budgets, conflated
  because both bound the forwarded body. `--compact-history-budget` (48000) DROPS old
  whole turns past the anchor: a cache-preserving shed. `--ctx-view-budget` (8000) is
  the O(1) `ctxplan` planner's resident-VIEW budget: it re-materializes history as a
  bounded planned view in place of the raw transcript. Shed vs planned view - a size
  cut on the tail vs a re-plan of what stays resident. The sizing rule for both is in
  the [long-context defaults doctrine](../long-context-defaults.md): HardContextCap is
  a hard cap, not a target, and the resident budget must leave output reserve.

### managed cache - the knob, the lever, the tier, and the restart plan

"Managed cache" names fak actively DRIVING the provider's prompt cache instead of
merely forwarding the client's `cache_control` bytes. Several names ride that
phrase; they live at different layers, two collide in spelling, and one is a
different sense entirely:

- **managed cache** (the posture: `fak guard --managed-cache auto|on|off`, epic
  #1844 C6) - should THIS guard session actively manage the prompt cache on the
  outbound Anthropic wire? AUTO activates only when the session provably bills an
  operator API key (`--api-key-env` resolved a key, no subscription token pinned);
  subscription OAuth, non-Anthropic wires, and local models stay passive - fak
  never speculates with billing it cannot see. `resolveGuardManagedCache` resolves
  the knob once at startup into **guardManagedCachePosture** (mode, active,
  reason), rendered in the banner. *Not* the Prompt cache (the provider feature it
  manages), *not* vCache (the virtual control plane fak builds), and *not*
  managed context (the gateway's context program - same "managed", different
  resource).

- **Config.CacheTTL1H** (the gateway LEVER, `internal/gateway/gateway.go`) vs
  **CacheTTL1h** (the pricing TIER, `internal/gateway/cache_pricing.go`) - one
  spelling, two concepts. The lever is the Config bool an ACTIVE posture arms:
  each outbound request's existing stable-head `cache_control` breakpoint is
  upgraded to the 1-hour tier (`maybeUpgradeAnthropicCacheTTL1H` ->
  `agent.UpgradeAnthropicStableCacheTTL1h`), so an idle gap past 5 minutes
  re-enters on a 0.1x cache read instead of re-writing the whole prefix. The tier
  is the PRICE of that choice: 1h writes cost 2x once (vs 1.25x at 5m) - the
  lever ARMS the upgrade, the tier PRICES it. `FAK_ABLATE_TTL_1H=1` is the
  ablation arm that forces the same lever for A/B measurement, independent of
  posture. Every attempt is witnessed by `fak_gateway_cache_ttl_upgrade_total`
  (outcome-labelled; recorded only while the lever is on, so a zero panel with
  the lever active means every head was ineligible - visible, not silent).

- **managed-cache restart plan** (`internal/resume`) - the OTHER sense of the
  phrase: not a live session's wire posture but the restart verdict for a
  DORMANT one. `Diagnose` finds the transcripts that crashed on a rate limit and
  never resumed; `Plan` prices RESUME_FULL vs CUT (RESET always priced as the
  alternative) against the projected cache posture, so the restart is "a new
  session with cache managed" instead of a blind cold re-prefill of the whole
  resident transcript. *Not* the guard posture above (live wire vs restart
  pricing) and *not* the ctxplanner / memq `Plan` types (see the plan family).

### cache coverage extension (#4681) - six more names that confused readers

These six were discovered by the coverage engine but never positioned; each is a
genuine cache concept a reader could not pin, not an inflection.

- **cachesweep** (`fak cachesweep`, `internal/cachesweep`) - a HOUSEKEEPING verb that
  reaps stale or orphaned cache artifacts the dispatcher and session layers leave
  behind. *Not* the KV cache (live tensor storage) and *not* cachemeta (the metadata
  contract whose expired entries it may reap).

- **dispatchcache** (`internal/dispatchcache`) - a per-lane DISPATCH-ROUTING
  memoization cache so a re-tick does not recompute the full pairwise issue-to-worker
  scan. *Not* the KV cache (inference tensor storage) and *not* DuplicateRiskCache
  (the narrower memo that caches only the duplicate-risk scan).

- **CacheTTL5m** (`gateway.CacheTTL5m`, `"5m"`) - the default EPHEMERAL provider cache
  tier: the 5-minute TTL window a prefix lives at when the managed-cache posture is OFF
  or auto-unupgraded. *Not* CacheTTL1h (the upgraded 1-hour tier the managed-cache
  posture drives toward) and *not* DefaultVCacheAnchor (the star-anchor gate default).

- **messageHasCacheControlForElide** (`agent.messageHasCacheControlForElide`) - the
  DEEP, any-depth `cache_control` detector the elision shrinker needs because it can
  reach nested tool_result content. *Not* messageHasCacheControl (the shallow,
  one-level variant the compaction guards use) and *not* toolResultContentHasCacheControl
  (the tool_result-content-only variant).

- **prefix_cache** - the provider-side (vLLM / SGLang) or fak in-kernel (RadixKV)
  prefix-reuse cache, observed via `prefix_cache_hit_rate` / `prefix_cache_{queries,hits}`
  metrics. *Not* the KV cache (the raw tensor storage it indexes into) and *not* the
  prompt cache (the Anthropic `cache_control` breakpoint mechanism).

- **kvcached** - an EXTERNAL kernel-module-level KV-cache management tool studied in
  `BORROW-KVCACHED-STUDY-2026-07-10` for its `in_shrink` guard. *Not* the KV cache
  (fak's own tensor storage) and *not* vCache (fak's virtual cache control plane).

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

- **guardaccuracy** (`internal/guardaccuracy`) - NOT a guard and NOT a gate: the
  labeled command corpus (`testdata/corpus.json`, schema `fak-guard-accuracy-corpus/1`)
  that MEASURES how well a gate decides. Each row is a `(tool, args)` command paired
  with the reversibility preview class the guard MUST assign; a benign row escalated is
  a false POSITIVE, a dangerous row left reversible a false NEGATIVE. A guard is the
  system and a gate DECIDES a call; guardaccuracy scores that decision's
  false-positive / false-negative RATE against ground truth. Grown as a ratchet - every
  wild misfire becomes a permanent row, never just a local test patch.

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

### guard-wrapper components vs gate decision points

More similar-sounding names in this family. The recurring line: a **guard component**
is an INSTALL/RENDER/RESOLVE step of the guard wrapper (it edits the child launch or
paints the pane, making no per-call decision), while a **gate** is one admit/refuse
decision over a call, a report, or a diff.

Guard-wrapper components (part of the SYSTEM, not a gate):

- **guard_mcp** (`cmd/fak/guard_mcp.go`) vs **guard_codex** (`cmd/fak/guard_codex.go`)
  vs **installGuardPiExtension** (`cmd/fak/guard_pi.go`) - three INSTALL steps that
  inject routing into the guarded child before it runs: guard_mcp registers fak's own
  MCP self-query servers, guard_codex is the OpenAI-Codex provider install path, and
  installGuardPiExtension rewrites a Pi child to route through the gateway via a
  `-e <ext.ts>` extension. None decides a call; each edits the launch. Distinct from
  **guardMcpStatusAudit**, which AUDITS whether the MCP injection is live, and from
  **installGuardCodexConfig**, the routing-rewrite STEP inside guard_codex.

- **GuardAssumption** (`internal/assumecheck`) - the adjudication that folds a stated
  assumption plus its evidence into an Allow/Refuse verdict for `fak assume check`. It
  gates a CLAIM's truth, not a tool call or its result; **guardaccuracy** scores a
  gate's decision, whereas GuardAssumption IS the decision over an assumption.

- **guardUnattestedBuildWarning** vs **guardInfoStalenessNote**
  (`cmd/fak/guard_freshness.go`) - the SAME build-staleness fact rendered two ways:
  guardUnattestedBuildWarning is the one-shot WARN under the launch banner's build row;
  guardInfoStalenessNote is its PERSISTENT twin in the guardInfo pane. Banner-once vs
  pane-persistent. **guardInfoVisualIdentityRow** (`cmd/fak/info_visual.go`) is the same
  pane's header row - which fak is watched and for how long - identity, not freshness.

- **resolveGuardRemoteServe** (`cmd/fak/guard.go`) - resolves the guard's OWN
  `--remote-serve` endpoint, distinct from **guardProvider**, which resolves the
  UPSTREAM provider wire the guard forwards to. Own-endpoint vs upstream-endpoint.

Gate decision points (one admit/refuse):

- **CheckGateTriaged** vs **AdvisoryGateTriaged** - a report-boundary CI gate that
  passes only when every cadence/milestone finding is triaged. CheckGateTriaged
  ENFORCES (non-zero exit); AdvisoryGateTriaged is the advisory twin that warns without
  failing CI until enforcement is on. Enforce vs advise, same triage condition.

- **gateTierDeclaredTree** vs **gateUntieredLeaf** (`internal/hooks`) - both enforce
  that every `internal/<leaf>` declares a support tier. gateTierDeclaredTree audits the
  WHOLE tree; gateUntieredLeaf gates only the leaves a STAGED diff touches, at the
  pre-commit boundary. Tree-scope vs staged-scope.

- **GateSpendLabeled** (`internal/metrics`) - refuses a spend rollup with unlabeled cost
  categories (the spend twin of GateBudgetLabeled). A data-hygiene gate over a report,
  not a tool-call adjudicator or result admitter.

Config / metric knobs on the prefix and pre-staged guards:

- **prefixGuard** (feature) vs **FAK_ABLATE_PREFIX_GUARD** (ablation OFF switch) vs
  **fak_prefix_guard_\*** (the determinism-witness metric, `observePrefixGuard`) - the
  feature that keeps the cacheable prompt prefix stable, the knob that turns it off to
  measure its value, and the counter that observes whether the prefix is actually
  stable. Feature vs control vs measurement.

- **FAK_PRESTAGED_PATH_GUARD** (`internal/hooks/gate_barecommitsweep.go`) - the env
  opt-out for the git-hook gate family that refuses committing paths staged outside the
  sanctioned by-path flow. `=off` disables the whole FAMILY, distinct from
  **ALLOW_BARE_COMMIT**, which skips the check ONCE.

### hardware-gate rung vs hardware-gate sensor

- **hwgate** (`cmd/fak/guard_hardware_gate.go`) - the guard RUNG that decides what to
  do when an agent stops for lack of local hardware, folding hwgatelint findings through
  the off|shadow|warn|enforce ladder. A DECISION logic, NOT a scanner.
- **hwgatelint** (`internal/hwgatelint`) - the SENSOR package that scans agent
  final-output text for local-hardware stop patterns (NO_LOCAL_GPU / NO_LOCAL_RUNTIME /
  LOCAL_BOUNDARY) and returns sanctioned-compute-node redirects. A TEXT SCANNER, NOT the
  decision rung that folds its findings. The `fak hwgate-lint` CLI command is the
  operator shell over this sensor.

### gateway stop gate vs guard stop hook

- **StopGate** (`internal/gateway/gateway.go`) - a GATEWAY-level gate that checks
  declared completion evidence at a model-final boundary before allowing a stop to
  finalize. It gates a MODEL's stop, not an agent's tool call. Distinct from the
  **guard stop hook** (`x2-guard-gate-stophook`), which is the guard's post-turn stop
  logic that produces stop dispositions.

### guard-stops tally vs stop disposition vs stop hook

Three closely named concepts in the guard's stop subsystem:

- **guard-stops** (`fak guard-stops`) - the operator-facing TALLY COMMAND that folds the
  guard's stop-history ledger into a summary for the soak to promote read. A TOOL, not a
  type or a hook.
- **guardStopDisposition** (`cmd/fak/guard_stops.go`) - the closed VOCABULARY of typed
  terminal outcomes each stop carries (hardware_gate_continue, hardware_gate_warn,
  hardware_gate_shadow, operator_directed, etc.). A TYPE, not a command or a hook.
- **guard stop hook** (`x2-guard-gate-stophook`) - the HOOK that produces stops and
  emits dispositions. A MECHANISM, not a command or a type.

### sweep guard vs gitgate vs trunk guard

- **SweepGuard** (`internal/gitgate/sweepguard.go`, `internal/wipattr/sweep.go`) - the
  attribution-aware sweep guard that classifies each dirty hunk a path-scoped git op
  would sweep as OWNED-by-self / OWNED-by-peer / SHARED / ORPHAN and refuses the
  irrecoverable ORPHAN case. It makes gitgate's blunt shared-tree mutation refusal
  PRECISE by consulting wipattr attribution. A specific GITGATE RUNG, not the general
  gitgate adjudicator and not the branch-state trunk guard.

### OPERATOR_GATE vs gate vs guardrail

- **OPERATOR_GATE** - a closed-vocabulary refusal-reason CATEGORY that routes a stop to
  the operator instead of auto-replanning (RELAY_NO_PROGRESS, RELAY_PARKED_UNSAFE,
  UNTIERED_LEAF). A REASON CLASS, not a gate TYPE (a decision point) and not guardrail
  (the safety-boundary concept). A gate DECIDES a call; OPERATOR_GATE CLASSIFIES a
  refusal's routing.

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

### witness naming across the subsystems

The word **witness** recurs in three unrelated subsystems - assume-check, concept-bench,
and the reporting/metrics layer. They share the spelling, not the meaning; the axis that
separates them is *what is being witnessed and where*.

- **assume-check evidence kinds** (`internal/assumecheck`) - how an assumption's evidence
  is GATHERED. **WitnessCommandProbe** runs a command, **WitnessConfigFlag** reads a
  config flag, and **WitnessLedgerRead** is a bespoke authority read of a ledger (NOT one
  of the generic probe kinds - it has no generic driver). **WitnessStatus** is the
  per-assumption FIELD declaring how the witness is wired, and **WitnessWired** is the
  status VALUE meaning it is fully wired. Field vs value vs gather-method - none is a
  witness OUTCOME (that is WitnessConfirmed / WitnessRefuted, in `internal/abi`).

- **concept-bench witness sources** (`internal/conceptbench`) - which authority PROVED a
  benchmarked concept. **WitnessSource** is the referee field; its values name the prover:
  **WitnessDosArbitrate** (`dos_arbitrate`, the lane concept), **WitnessDosCheckReason**
  (`dos_check_reason`, the refusal-vocabulary concept), **WitnessDosCommitAudit**
  (`dos_commit_audit`, the commit-stamp concept, combined with **WitnessDosVerify**), and
  **WitnessHandoffSchema** (`fak.task-handoff.v1`, the hook-protocol concept - proof by
  schema emission, not by a DOS tool call).

- **reporting / metrics witnesses** - **WitnessRef** (`internal/milestonereport`) is a
  criterion's `witness_ref`: the commit/artifact reference backing that criterion's grade
  - a report field, not a cache witness (world-state witness lives in `internal/vdso`).
  **SpendWitnessed** (`internal/metrics`) is a spend PROVENANCE value (witnessed from real
  session evidence vs estimated). **ProgressWitnessed** (`cmd/fak/info_watchdog.go`) is a
  watchdog COUNTER of resume events witnessed (the "N resumed" info line), distinct from
  **ProgressWitnessedAt**, the TIMESTAMP of the last one.

- **RegisterWitnessResolver** (`internal/abi`) - the registration seam that installs a
  WitnessResolver backing the require-witness verdict. Register (install a resolver) vs
  **WitnessResolver** (corroborate a claim) vs **resolveWitness** (the kernel driver that
  folds the registered resolvers at adjudication). Three stages of one pipeline.

### witness grading on issue contracts (#4719)

Three names in `internal/issuecontract` that grade a ticket's done-condition witness
before dispatch. The readout, its strongest value, and the flag that escalates
non-strong grades to holds.

- **WitnessGrade** (`issuecontract.WitnessGrade`) - the advisory-first pre-dispatch
  FORGEABILITY readout for a ticket's done condition: grades the witness as strong
  (independent oracle named), weak (no oracle), forgeable (self-claim only), or missing
  (no witness declared). *Not* a WitnessOutcome (a gate-level corroboration verdict) and
  *not* the WitnessResolver (the engine that corroborates). It grades an ISSUE'S witness
  before dispatch, not a claim at adjudication.

- **WitnessGradeStrong** (`issuecontract.WitnessGradeStrong`, const `"strong"`) - the
  strongest of four WitnessGrade values, meaning the done-condition witness names an
  independent oracle (go test, make ci, fak, dos, git show, a fixture, a render, an exit
  code, a ledger, etc.) that can corroborate the claimed effect. *Not* WitnessGrade (the
  readout struct that holds it) and *not* WitnessConfirmed (a gate-level outcome). A grade
  value vs the grade readout vs a corroboration outcome.

- **StrictWitness** (`issuecontract.StrictWitness`, CLI `--strict-witness`) - the POLICY
  FLAG that promotes any non-strong WitnessGrade to a hold: when true, an issue whose
  done-condition witness is weak, forgeable, or missing is held for triage instead of
  dispatched. *Not* WitnessGrade (the readout it gates) and *not* WitnessGradeStrong (the
  grade value it checks against). A gate on a grade vs the grade itself vs its strongest
  value.

### loop governor witness health (#4719)

Three names in `internal/loopmgr` for the witnessed/claimed-ratio health of a loop. The
policy floor, the refuse reason, and the health descriptor.

- **MinWitnessRate** (`loopmgr.Policy.MinWitnessRate`, JSON `min_witness_rate`) - the
  governor POLICY FLOOR: a loop whose witnessed/claimed completion ratio drops below this
  rate is refused new admissions, because its runs are predominantly self-reported rather
  than independently confirmed. *Not* ReasonWitnessCollapse (the refuse reason the
  governor emits when below it) and *not* WitnessCollapse (the health descriptor that
  reports the majority-unwitnessed state). Threshold vs reason vs descriptor.

- **ReasonWitnessCollapse** (`loopmgr.ReasonWitnessCollapse`, const `"WITNESS_COLLAPSE"`)
  - the structured REFUSE REASON the loop governor emits when it denies admission to a
  loop whose witnessed/claimed completion ratio fell below Policy.MinWitnessRate. *Not*
  MinWitnessRate (the policy floor it enforces) and *not* WitnessCollapse (the health
  descriptor that reports the state without refusing). A reason code vs a threshold vs a
  descriptor.

- **WitnessCollapse** (`loopmgr.WitnessCollapse`, bool on HealthRow / int on HealthRollup)
  - the advisory HEALTH DESCRIPTOR: true on a per-loop row when a MAJORITY of ended runs
  went unwitnessed (Witnessed*2 < Runs), and the fleet-wide count of such loops on the
  rollup. It is descriptive, not a gate. *Not* ReasonWitnessCollapse (the governor refuse
  reason) and *not* MinWitnessRate (the policy floor). It describes the symptom; the
  governor acts on the threshold.

### trunk-red witness ledger (#4719)

Two names in `cmd/fak/trunk_red_ledger.go` for the shared pre-existing-red admission
ledger. One writes the witness row; the other renders its convergence payoff.

- **emitTrunkRedWitness** (`cmd/fak.emitTrunkRedWitness`) - the RECORDER function that
  appends one pre-existing-red admission to the shared trunk-red ledger (FAIL-OPEN: a
  no-op when recording is disabled or no failing package was parsed), folding the ledger
  to return occurrence and session counts. *Not* buildwitness (a structural compile guard)
  and *not* worktreewitness (a detached-trunk command harness). Each witnesses a different
  kind of build fact.

- **trunkRedWitnessNote** (`cmd/fak.trunkRedWitnessNote`) - the RENDERER of the
  convergence line appended to a gate's pre-existing-red advisory, ONLY when the row was
  actually written, making the shared-break payoff visible (how many clones are stuck on
  this break across how many sessions). *Not* emitTrunkRedWitness (the recorder that
  writes the ledger row it reads from). One writes the witness; the other renders its
  payoff.

### other witness concepts positioned (#4719)

Seven more witness-rooted names, each a genuine concept a reader could not pin, not an
inflection.

- **WitnessedEnvelope** (`issuecontract.WitnessedEnvelope`, JSON
  `witnessed_operating_envelope`) - the OBSERVED operating-envelope section on a
  project-work item / issue contract / handoff: the baseline, target, and
  completion-standard envelope an operator has directly verified, distinct from the
  TargetEnvelope that is merely declared. *Not* a world-state witness (a cache-coherence
  binding) and *not* a WitnessOutcome (a gate verdict). It witnesses an ISSUE'S scope, not
  a cache entry or a claim.

- **CrossAuditRefute** (`modelroute.CrossAuditRefute`, `CrossAuditVerdict = "REFUTE"`) -
  the cross-audit verdict meaning a model-route quality audit found evidence that
  CONTRADICTS a model's claimed routing behavior, alongside CrossAuditPass /
  CrossAuditInconclusive / CrossAuditUnavailable. *Not* the cache-coherence Refutation (a
  local invalidation of a world-state witness) and *not* abi.WitnessRefuted (a gate-level
  outcome). Three refusals in three subsystems sharing the refute root.

- **ContinuityWitness** (`resume.ContinuityWitness`) - the per-RESUME verified-progress
  verdict: a fold of trajctl W3 ScoreRows across the launch boundary that reports whether
  the objective's witnessed curve actually advanced (Witnessed + Advanced), or whether the
  resume only produced turns without moving verified progress. *Not* a world-state witness
  and *not* a WitnessOutcome. It witnesses resume PROGRESS, not a cache entry or a claim.

- **ReconstructionWitness** (`modelscore.ReconstructionWitness`, JSON
  `reconstruction_witness`) - the PROVENANCE string on a model-score record naming how the
  model was reconstructed (e.g. "committed benchmark snapshot"), required for the
  reconstructed-from-blog provenance tier. *Not* a world-state witness and *not*
  ProvenanceWitnessed (an evidence-strength label on a context-envelope). It witnesses a
  MODEL'S ORIGIN, not a cache entry or a context row.

- **WitnessName** (`deletioncert.WitnessName`, JSON `witness_name`) - the NAME FIELD on a
  deletion certificate carrying the vDSO world-state witness identifier under which the
  evicted entries were admitted (empty if none). *Not* the world-state witness value
  itself and *not* the WitnessResolver. A reference to a witness vs the witness vs the
  resolver.

- **WitnessDir** (`mlpscore.WitnessDir`, `qwen36parity.WitnessDir`) - the on-disk
  DIRECTORY constant where witness artifacts are stored, declared independently in
  multiple packages with different paths (docs/mlp/witnesses for MLP manifests,
  experiments/agent-live for Mac parity witnesses). *Not* a world-state witness and *not*
  the WitnessResolver. Where witnesses are STORED vs what they BIND vs what RESOLVES them.

- **WitnessSameTasks** (`benchcatalog.WitnessSameTasks`) - the FAIRNESS FENCE for any
  two-arm (raw vs fak) benchmark ablation: a pure function that takes the task ids each
  arm RECORDED consuming and reports whether they match in order, so the delta is
  attributable to fak BECAUSE both arms ran the same problems (empty is a mismatch, never
  a silent pass). *Not* a world-state witness (a cache-coherence binding) and *not*
  WitnessTask (the taskmgr entry point that applies a witness to a task's claim). It
  witnesses BENCHMARK FAIRNESS (same problem set), not a cache entry or a claim.

### ignored false friends in the witness-proof family

Two discovered tokens are NOT concepts and are ignored by the family's `ignore` list:

- **refutil** (`internal/refutil`) - a false friend: the package name is "ref util"
  (reference-utility helpers for materializing ABI refs), not "refute". The `refut` root
  matches it by substring coincidence only.

- **nwitness** - a scanner artifact: the identifier regex reads `\nWitness` / `\nwitness`
  in Go string literals (where `\n` is a newline escape) as the identifier `nWitness` /
  `nwitness`. It is not a real identifier or concept.

- **witnesssametaskids** - a non-existent variant: the `WitnessSameTasks` function
  witnesses that two benchmark arms consumed the same task ids, but no identifier
  `WitnessSameTaskIDs` (or `witness_same_task_ids`) exists in the tree. The token is the
  conceptual reading of the function's purpose, not a real symbol; `witnesssametasks`
  (the function itself) is positioned as its own row.

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

- **sessionjournal** vs **ScratchJournal** vs **SessionLedger** - sessionjournal
  (internal/sessionjournal) is the CRASH-RECOVERY journal: a boot-epoch fold of
  open/beat/close lifecycle events that classifies each session LIVE/CRASHED/STALE/CLOSED
  for resume targeting. ScratchJournal is the in-process append-only ledger scratch_lease
  implements today. SessionLedger is the planned #2392 generalization of ScratchJournal.
  Crash-recovery classification vs current in-process ledger vs planned generalization.

- **sessionread** vs **sessionctl** vs **sessionsearch** - sessionread is the closed
  READ vocabulary spine (outbound session-read seams: context-restore, context-spans,
  context-value, each scope-checked); sessionctl is the redirect CONTROL op (an inbound
  steer); sessionsearch is cross-session RECALL over the guard journal. Read vs control
  vs recall.

- **CLAUDE_SESSION_ID** vs **FAK_SESSION_ID** - CLAUDE_SESSION_ID is the env var carrying
  the UPSTREAM Claude Code harness's session identity, resolved by resume/mcp when the
  caller omits an explicit session id. FAK_SESSION_ID is the env var carrying a FAK-SERVED
  session's identity across a guard relaunch. Upstream harness identity vs fak-served
  identity.

- **sessionCtxRestore** vs **sessionCtxValue** - sessionCtxRestore is the per-trace
  ordered stash of context-RESTORE entries (oldest first, backing the sessionread
  OpContextRestore seam); sessionCtxValue is the per-session rolling managed-CONTEXT
  accumulator (tracking resident token count, growth-per-turn, ring state). Restore
  entries vs context-value accumulator.

- **SessionRef** vs **SessionFromRef** - SessionRef BUILDS the fully-qualified checkpoint
  ref by prepending the refs/fak/locks/ namespace to a session id; SessionFromRef PARSES
  the bare session id back out of a ref by stripping that prefix. Build vs parse.

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

- **AdjudicateMemoryWrite (memq)** vs **adjudicator** - the deny-by-structure rule set
  that judges a durable MEMORY WRITE (a memq cell body) by structure alone, vs the
  pre-call reference monitor that folds a tool-CALL decision chain under the loaded
  capability policy. Memory-write admission vs tool-call admission; AdmissionVerdict vs
  abi.Verdict.

- **ContainmentPolicy (toolprocgate)** vs **Policy (loaded)** - the runtime-enforcement
  knobs that bound the blast radius of a console/terminal crash (per-surface agent cap,
  surface quarantine, fleet breaker) folded into a spawn-admission ContainmentVerdict, vs
  the adjudicator's tool-call decision table. Crash-blast-radius spawn gate vs tool-call
  authorization.

- **AuditIndependencePolicy (modelroute)** vs **Policy (loaded)** - the versioned
  admission policy that decides whether an AUDITOR may audit an AUTHOR (required identity
  axes + diversity knobs), vs the adjudicator's tool-call decision table.
  Audit-independence admission vs tool-call authorization; AuditIndependenceDecision vs
  abi.Verdict.

- **POLICY_MALFORMED (resume)** vs **POLICY_BLOCK (adjudicator)** - the closed refusal a
  PRESENT-but-unparseable resume source-governor policy FILE earns, vs the adjudicator's
  explicit deny-rule match on a tool call. A malformed-policy-file rail vs a tool-deny
  rule; both carry 'policy' but in different domains (resume source governor vs
  tool-call adjudicator).

- **preflight_focus (dispatchtick)** vs **preflight ladder** - the dispatch spawn-admission
  WIP-breadth backpressure term (folds measured fleet breadth vs the pinned WIP cap, emits
  FOCUS_WIP_SATURATED) that throttles opening a NEW objective, vs the per-call
  well-formedness schema rungs. Dispatch spawn WIP gate vs per-call schema preflight.

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

- **ContainmentDecision (toolprocgate)** - the closed verdict-plus-evidence struct
  returned by DecideContainment that adjudicates whether a tool-process spawn is
  admitted, deferred, or refused based on crash-blast-radius containment policy
  (fleet breaker, surface quarantine, co-location cap). *Not* kernel Decision (that
  explains a tool-call Allow/Deny, not a spawn-admission gate) and *not* ResetDecision
  (that decides gateway cache-health cut-vs-reset, not process blast radius).

- **SteerDecision (trajctl)** - one regime-gate steering decision for one objective at
  one turn boundary: an Action (nudge/arm/suppress/none) plus the Signal that triggered
  it, produced when the recent score curve is unhealthy. *Not* kernel Decision (that
  records a past call's adjudication, not a live-run intervention) and *not* witness
  Decision (that confirms/refutes git evidence after the fact, not a controller actuator).

- **WalkDecision (gardenbundle)** - one budgeted item's triage outcome from a garden
  walk: a Disposition (act/review/defer), the ready command, and a reason. *Not*
  DriveDecision (that picks the worst-first super-loop member to enter, not one issue's
  handling) and *not* TierDecision (that joins a past dispatch tier to its witnessed
  outcome, not a forward triage disposition).

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

### "candidate" and "plan" across the other subsystems

Both roots are overloaded far beyond the ctxplanner. The line each draws is against the
canonical **Candidate** (a scored context span) or **Plan (planner)** (a resident view):

- **dispatch candidates** - `CandidateBlockedBy` (dispatchtick prereq grammar - who a
  candidate waits on), `candidatesPath` (the ready-set file on disk), `decodeCandidates`
  (its JSON decoder). These are launch-eligibility records in the dispatch ready-set, not
  scored context spans. `decodeCandidates` is distinct from `parseCandidates` /
  `decodeIssueContractCandidates` - each decodes a *different* candidate corpus.
- **scored/counted candidates elsewhere** - `candidateIDs` (kvmmu rescore / modelroute
  audit: the id vector parallel to a score array), `CandidatesExamined` (an issue-repair
  loop counter), `nCandidate` (a signal-select count). An id-vector / a counter, not the
  Candidate object; distinct from `candidate-count` (a dispatch *price*) and
  `CandidateBound` (the ctxplan set *cap*).
- **promotion / resume candidates** - `KnownBadCandidate` (a guardrsi fleet-correlated
  failure pattern proposed for filing) and `PrefixCandidates` (resume partial-id prefix
  matches, #3782). Neither is a context span.
- **"plan" as a side-effect or ablation plan** - `buildKnownBadIssuePlan` (a gh
  create/update plan - *not* a ctxplan/memq plan), `planProfile` (multisubmit's pure
  plan-construction step, split from the impure `runProfile`), `FAK_ABLATE_BP_PLAN` (the
  env knob for the `bp_plan` breakpoint-plan ablation - the knob-name, distinct from
  `FeatureBreakpointPlan` the sweep token and `BreakpointPlan` the layout it ablates), and
  `plannedOpen` (a checkpoint-scorecard predicate: a *planned* subsystem whose dir is still
  absent - distinct from `planless` and `ClosedNotPlanned`).

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

## The loop family

The most overloaded word in the repo. Six families of "keep work happening" machinery
grew at different times, and the same token names a read-only walker, a bulk fleet
launcher, a ledger, an admission gate, and a scorecard. Draw the line on sight; the
full operational map is [the loop-family map](../notes/CONCEPT-CONTINUAL-WORK-LOOP-MAP-2026-07-02.md).

- **loop (the ring)** - the generic engineering abstraction: sense -> decide -> act ->
  witness, at any altitude (tool-call / turn / session / fleet). *Not* any one named
  mechanism below - it is the shape they all instantiate.

- **super loop (`fak superloop`)** - a read-only INTENT walker: an interior node that
  reads its member loops' status worst-first and mutates nothing. *Not* the `/super-loop`
  skill (a bulk detached wave LAUNCHER that spawns fleets) - same word, opposite risk.

- **`fak loop` (ledger + governor)** - the durable hash-chained ledger
  (`.fak/loops.jsonl`) plus the admission governor (`loopmgr.Admit`); it records and
  gates, it does not itself loop. *Not* `fak loop drive` (that is an actual Ralph loop
  that settles one `GOAL.md` witness).

- **bench-loop (`fak bench-loop`)** - the benchmark control surface that folds the
  benchmark registry + run catalog + nightrun ledger into the single next benchmark
  action. *Not* `loopbench` (`internal/loopgate/loopbench.go`, the verified-vs-naive
  exit-gate micro-bench) - unrelated code paths sharing a spelling.

- **loop-index (`fak loop-index-scorecard`)** - the Orient->Plan->Act->Verify->Ship->Learn
  STAGE-coverage scorecard: are the agentic-coding loop stages witnessed at floor. *Not*
  an index OF loops (that role is the loop-family map, a doc, not a scorecard).

- **loopfleet** - the cross-ledger loop-health FOLD (`internal/loopfleet`): rolls up
  loopmgr / nightrun / dojo / cadence / dispatch ledgers into one live/stale/dark view.
  *Not* `loopmgr` (that governs ONE loop's admission; loopfleet only reads many).

- **bgloop** - the always-on background-loop supervisor (`internal/bgloop`,
  `fak bgloop`) that keeps a detached loop process alive. *Not* `fak loop` (the ledger)
  and *Not* a super loop (an intent walker) - bgloop is the process babysitter.

- **loopback (network)** - the 127.0.0.1 network address / same-host bind, swallowed by
  the `loop` root but from a different domain entirely. *Not* any work-loop concept -
  it is a networking term, drawn here only because the token collides.

---

## The trajectory-control family

The "trajectory" root now names two tenses and the "score" root three subsystems.
The live control plane is **trajctl** (`internal/trajctl`, epic #2533); the full
positioning is [the trajectory-control page](../observability/trajectory-control.md).

- **trajectory control (trajctl)** - the LIVE forward-progress control plane over
  declared objectives: anything you want to progress gets a named objective and a
  witnessed score curve, and steering reads curves, never points. *Not* `fak traj`
  (`internal/trajectory` + `internal/trajhook`, the RETROSPECTIVE corpus toolkit -
  trajhook's `Score` is notability of a PAST turn, trajctl's score is progress of a
  LIVE objective), *not* the scorecard family (deterministic measurement of a REPO
  surface, not a run), *not* score-signal (the CI-regression issue feeder), *not*
  `fak signal steer` (the operator input channel trajctl merely uses as one
  actuator), and *not* loopdrive's GOAL.md witness (the binary done-bit trajctl
  stretches into a curve).

- **ScoreRow** - one scored observation of one objective: normalized value, the
  scorer method+version that produced it, a witness rung, and an evidence pointer,
  appended to the `fak-trajctl/1` JSONL ledger. *Not* a trajhook `Finding` (a
  worst-first notability flag on a past turn) and *not* a scorecard `*_debt`
  integer (a repo-surface fold with a CI ratchet).

- **witness rung (W0-W3)** - the evidence-strength ladder every ScoreRow carries:
  W3 deterministic evidence (witnessed commit, green suite), W2 transcript-derived
  heuristic, W1 structured judge verdict, W0 self-report - recorded, never
  load-bearing; the read-time audit (`witnessStrength`) DEMOTES a dangling W3 row
  to W0 rather than keep it. *Not* the witness/evidence family's world-state or
  measurement witness (those witness a CACHE entry or an RSI gain; a rung grades a
  SCORE's evidence), and *not* dispatchtick's `WitnessOK` (a resolution
  corroboration constant).

- **regime gate** - the rule that HEALTHY means DO NOTHING: every steering rung
  above "annotate" fires only when the recent curve is unhealthy (STALL / DRIFT /
  DETOUR_OVERRUN), because intervening in a high-scoring session degrades it
  (arXiv:2602.03338). The dogfood run's `HEALTHY -> withhold` decision is this
  gate working. *Not* an adjudication gate (no tool call is allowed or denied -
  it gates the CONTROLLER's own interventions) and *not* a model gate (nothing
  neural).

- **detour objective** - a side-quest promoted to a first-class CHILD objective
  with its own scorers and a budget, so an error repair or infra fix is scored on
  ITS goal while the parent sits visibly paused; `DETOUR_OVERRUN` fires when the
  detour outlives its budget - trajectory control means the detour RETURNS. *Not*
  drift (declining alignment on the SAME objective - a detour is declared, drift
  is decay) and *not* a distraction to suppress (the score records whether the
  repair landed).

---

## The dev-tier / operator-surface family

The `dev` root spans a CLI namespace, a GitHub label, a catalog package, and two
verb tiers - the confusable class #1420 tracks and epic #2228 (C6, #2235) splits.
The operator-heaviness meter now partitions the flat top-level verb surface into two
tiers by construction (`frontdoor_verbs + dev_verbs` == the old flat count, the
continuity witness), classified from the live dispatch switch via `internal/devindex`.
These are the names for that split and its neighbors.

- **frontdoor verb** - a top-level `fak` verb classified `devindex.TierFrontdoor`:
  the product surface an operator FACES (what `fak help` lists), and the headline
  heaviness input (the `frontdoor_verbs` meter). *Not* a **dev verb** (the tooling
  tier), *not* the **`fak dev`** namespace (the command that gates the dev tier), and
  *not* **devindex** (the catalog that ASSIGNS the tier).

- **dev verb** - a top-level verb NOT classified frontdoor (`devindex.TierDev`): the
  `fak dev <verb>` tooling tier - repo-workflow verbs, scorecards, RSI. It stays
  MEASURED in the `dev_verbs` meter even after its bare spelling is gated behind
  `fak dev` (the honesty fence: a gated verb is hidden from the front door, not from
  the meter, so the heaviness drop comes from the frontdoor meter only). *Not* a
  **frontdoor verb**, and *not* `TierHidden` (internal re-exec seams, never a user
  verb).

- **`fak dev` (namespace)** - the CLI namespace that dispatches the dev-tier verbs
  (`resolveDevVerb`, `cmd/fak/dev.go`) and behind which the bare dev spellings are
  gated. A COMMAND surface. *Not* **dev-ex** (a GitHub developer-experience routing
  LABEL an issue carries, not a command), *not* **devindex** (the catalog it reads to
  decide what is dev-tier), and *not* a single **dev verb** (one entry, vs the
  namespace that groups them).

- **dev-ex** - the `dev-ex` GitHub issue LABEL (developer-experience routing on the
  dispatch board). A tag an issue carries, *not* the **`fak dev`** command namespace
  and *not* **devindex** the package; same `dev` root, an issue-routing label vs a CLI
  surface vs a catalog.

- **devindex** - `internal/devindex`, the CATALOG that classifies every verb's tier
  (`TierFrontdoor` / `TierDev` / `TierHidden`) from the live dispatch switch - the
  WITNESSED source the two heaviness meters read. *Not* the **`fak dev`** namespace
  (the CLI surface that consumes this catalog) and *not* **dev-ex** (the label).

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

---

## See the distinctions in action

The kernel **Decision** (the tool-call verdict explanation trace) and the **guard**
(the adjudication system) above are both visible in a single offline run — replay a
tool-call trace and read the per-call verdict table:

```bash
go run ./cmd/fak run --trace testdata/tau2/tau2-smoke.json
```

## Read next

- [edge-quickstart.md](edge-quickstart.md) — runs the same adjudication path end to end.
- [deployment-guide.md](deployment-guide.md) — how the guard, gateway, and engine wire together in production.


### SilentCacheInvalidation

The post-fire reconciliation signal (#2791): a compaction that FIRED - which by construction proves the protected prefix was spliced byte-identically, since verifySplicedBody turns any byte-inequality into a prefix_mismatch identity return - yet whose provider reported zero cache_read and nonzero cache_creation, evidencing the provider re-created the very prefix fak preserved (a TTL expiry or capacity eviction fak cannot prevent).

**Distinct from:** NOT CacheBreakEvent: that is a WITNESSED break fak itself authored and can see in its own splice verdict. SilentCacheInvalidation is the provider breaking a prefix fak PROVED it preserved - invisible to bytes.Equal, hence silent. Also NOT the #2785 induced-creation burst: a head-anchored fire bursts the recent suffix on purpose but still READS its protected head, so its cache_read stays positive and it is excluded here.
