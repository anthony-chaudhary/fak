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

## Start here: which glossary do you need?

This page serves one primary audience: a **fak contributor** resolving
implementation vocabulary — Go identifiers, package names, and internal families
that collide in spelling. It is current, machine-verified reference material:
every entry is anchored on disk and the concept-disambiguation scorecard checks
those anchors, so an entry here is a live claim, not a historical note. New
entries are appended at the tail of the page by the positioning tools, which is
why late-positioned symbols appear after the "Read next" section.

Product terms live one page over. The public vocabulary — session, agent,
context, model, memory, tool vs skill, steering, the preflight/inflight/prefill
split, and the cache-economics words a `fak manage` run prints — is defined in
plain language in the [fak glossary](../glossary.md), and it resolves there
without internal shorthand.

| Your term looks like | Open | Examples |
|---|---|---|
| a product word, or a word a `fak` command printed | [the fak glossary](../glossary.md) — the default for public readers | session, rebate, preflight |
| a Go identifier, package name, or internal family | this page | `abi.Verdict`, `ctxmmu`, `WitnessGrade` |

One checkable next action — verify this page's entries are real and anchored
(read-only; prints the current disambiguation grade):

```bash
python tools/concept_disambiguation_scorecard.py
```

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
SAME outbound `/v1/messages` body that `fak manage -- claude` forwards, but they answer
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

### managed cache - the family, the knob, the lever, the tier, and the restart plan

"Managed cache" is used two ways that constantly get conflated: (a) the **family**
of features by which fak shapes the provider prompt cache, and (b) the specific
`--managed-cache` posture that arms just ONE of them (the 1h-TTL upgrade). Reading
(b) as if it were (a) is the recurring confusion - it makes our own sessions look
"uncached" when they are not. Several names ride the phrase; they live at different
layers, two collide in spelling, and one is a different sense entirely:

- **managed cache** (the FAMILY) - the set of features that shape provider prompt
  caching: the provider 5m prompt cache (riding the client's own `cache_control`
  breakpoints), tool-prune, star-anchor breakpoint placement, compaction shed,
  defer-cold-tools, and the 1h-TTL upgrade, among others. On our own OAuth Claude
  Code seats this family is EFFECTIVE BY DEFAULT - carried by the 5m prompt cache
  (median cache-read share ~78-85% on substantial sessions) and tool-prune (~96-99%
  of sessions); the other members are inert-by-design or provider-blocked there. So
  "is caching working on our sessions?" = yes, via this family - *even though* the
  `--managed-cache` posture below resolves passive. Audit:
  `docs/notes/MANAGED-CACHE-FAMILY-OWN-SESSIONS-AUDIT-2026-07-18.md`.

- **managed cache** (the POSTURE: `fak manage --managed-cache auto|on|off`, epic
  #1844 C6) - should THIS guard session author the 1h-TTL upgrade on the outbound
  Anthropic wire? It governs ONLY the 1h-TTL member, not the whole family. AUTO
  activates only when the session provably bills an operator API key (`--api-key-env`
  resolved a key, no subscription token pinned); subscription OAuth, non-Anthropic
  wires, and local models stay passive - fak never speculates with billing it cannot
  see. Forced `on` does NOT override that on a subscription-OAuth seat: the provider
  400s a `ttl:"1h"` body on an OAuth credential every turn, so `on` DEGRADES to
  passive there (guard fix `43cbdb14a4da`) instead of crash-looping the seat - on
  OAuth, `auto` and `on` cache identically. `resolveGuardManagedCache` resolves the
  knob once at startup into **guardManagedCachePosture** (mode, active, reason),
  rendered in the banner. *Not* the Prompt cache (the provider feature it manages),
  *not* the family above (this posture is one member of it), *not* vCache (the
  virtual control plane fak builds), and *not* managed context (the gateway's
  context program - same "managed", different resource).

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
  the lever active means every head was ineligible - visible, not silent). These
  upgrade counters (`fak_gateway_cache_ttl_upgrade_total`,
  `cache_ttl_upgrades_upgraded`, `CacheTTLUpgraded`) count fak AUTHORING the upgrade
  on the outbound body, NOT the provider ACCEPTING it - so a nonzero upgrade count
  on a subscription-OAuth seat is a crash signal (each authored turn is a provider
  400), not a success. `upgraded` and `placed_and_upgraded` are both authoring
  outcomes; only skip reasons like `volatile_head` mean no upgrade was written.

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
  decision chain and admits results (`fak manage`). A guard is a SYSTEM.

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
  config flag, and **WitnessLedgerRead** is a purpose-built authority read of a ledger (NOT one
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
- **NonCandidate** - the study-classification disposition for a source record that is
  release metadata or otherwise outside the actionable mechanism queue. It is not a
  rejected planner Candidate and does not describe the final ticket selection.
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


### q2_0_witness_test

The stub-build (non-Apple-Silicon) test file q2_0_witness_test.go: it pins the ternary Q2_0 reference's math obligations -- bit layout, ternary code set, round-trip error bound, and ref-GEMV-vs-dense parity -- in every build that cannot execute the Metal kernel.

**Distinct from:** A source FILE name, not a run-status or claim verdict: it names where the Q2_0 math obligations are asserted, whereas the witness-proof family's other rows name whether a claim was corroborated at runtime. It records no verdict and gates no dispatch.


### gateVerbTierTree (whole-tree verb-tier gate)

Whole-tree fak hygiene gate (internal/hooks/gate_verbtier.go, reason VERB_UNTIERED) that refuses a dispatched cmd/fak verb whose token devindex.TierOf cannot resolve to a tier — the pre-push twin of devindex.TestVerbTierCoverageIsTotal (epic #2653).

**Distinct from:** Audits the VERB-tier table (every cmd/fak dispatch verb carries a tier); distinct from gateTierDeclaredTree, which audits the LEAF/package tier (every internal/<leaf> declares a support tier). Both whole-tree hooks gates over a devindex source-of-truth; different subject table.


### ForkSessionID

The forked session id whose lookahead rollout produced a RolloutEvidence/Lesson (#5204): the twin session spun off to roll the trajectory forward under the fork-rollout runner.

**Distinct from:** Identifies the throwaway FORK session a lookahead rollout ran in (evidence provenance), distinct from the live drive session it was forked from.


### local_cache_hit

A served prompt token reused from a KV prefix already resident on THIS box (an in-session prefix or a shared local KV store); one of the three cacheobs provenance-axis buckets (#3896, vLLM's by_source label).

**Distinct from:** A LOCAL-residency reuse - NOT a cross-fabric external_kv_transfer (the disaggregated tier) and NOT a local_compute re-prefill; the near-free reuse a single box already earns.


### AuditRefute

The shipgate closure-gate verdict token REFUTE: the independent cross-model auditor affirmatively refuted the closure claim, so a high-risk issue closure is blocked at the ship gate (#3860).

**Distinct from:** A shipgate-local projection of the model-route receipt verdict used to decide closure admission; distinct from CrossAuditRefute, which is the modelroute wire verdict on the receipt itself rather than the ship-gate decision token.


### CrossAuditPolicy

The shipgate enforcement policy for high-risk issue closures: the calibrated auditor-family allowlist, required calibration version, receipt freshness window, and staged-enablement prerequisites that decide whether a closure receipt opens the gate (#3860).

**Distinct from:** Governs whether an issue closure may land at the ship gate; distinct from AuditIndependencePolicy, which governs how modelroute selects an independent auditor when producing a receipt upstream of the gate.


### DefaultCrossAuditPolicy

The calibrated CrossAuditPolicy instance built from the measured issue-3854 calibration evidence (two independent families at issue-resolution-audit/v2) with the issue-3859 dogfood loop not green, so the closure gate ships enforcement-capable but in dry-run (#3860).

**Distinct from:** The measured-evidence default instance of the shipgate closure policy; distinct from DefaultAuditIndependencePolicy, which defaults modelroute auditor-independence selection rather than closure enforcement.


### AdjudicateClosure

The fail-closed shipgate decision function for a high-risk issue closure: structural deny first, then a calibrated, independent, fresh PASS receipt or an audited break-glass; runs dry-run while the calibration and dogfood prerequisites are unmet (#3860).

**Distinct from:** Adjudicates issue-closure admission at the ship gate; distinct from AdjudicateMemoryWrite, which adjudicates memory-write tool calls in memq rather than issue closures.


### ClosureDecision

The typed outcome of the shipgate high-risk closure gate: whether the closure is allowed, whether enforcement was on, whether dry-run would have blocked, and the closed-vocabulary reason (#3860).

**Distinct from:** Decides issue-closure admission at the ship gate; distinct from RouteDecision, which decides guardroute request routing rather than whether an issue closure may land.


### fak_gateway_kv_prefix_prompt_tokens_by_source_total

The gateway's per-turn split of in-kernel prompt tokens by PROVENANCE source (local_compute / local_cache_hit / external_kv_transfer), orthogonal to the reuse-depth family; the three sum to the by-source prompt tokens. external_kv_transfer isolates the disaggregation dividend — tokens a remote / L3 KV tier served that a single box would otherwise have re-prefilled.

**Distinct from:** Splits the SAME prompt tokens by WHERE reuse came from (source axis); distinct from x2-gateway-engine-fakgatewaykvprefix, the reuse-DEPTH family that measures HOW MUCH was reused.


### KindLoopFleet (whole-fleet member kind)

KindLoopFleet is the superloop member kind (#4955) whose single registry member with Ref=all ENUMERATES into one MemberStatus per ledgered loop on the canonical roster, so a walk covers every folded loop without hand-naming each one.

**Distinct from:** Not KindLoop (one hand-named loop per member) and not KindSuperloop (a container intent descended by the walk): KindLoopFleet is the fleet-wide enumerator whose statuses come from the cross-ledger loop-health fold at read time.


### superloop.LoopFleetStatuses (fleet member expansion)

LoopFleetStatuses is the pure expansion behind KindLoopFleet: it turns one fleet member plus the shell-read folded loops and skipped-ledger gaps into per-loop MemberStatus rows, surfacing every gap as UNMEASURED so it blocks Satisfied.

**Distinct from:** Not the impure cross-ledger fold itself (loopfleet.Fold reads ledgers; this folds its already-read rows), and not BuildRoster (the three-source union readout): LoopFleetStatuses only shapes walk statuses for one enumerating member.


### superloop.RosterLoop (folded-loop input row)

RosterLoop is the plain-data input row the shell hands the pure roster builder for ONE folded loop: its stable identity (Kind), folded state, and dark flag, mirroring a loopfleet.LoopHealth row without importing it.

**Distinct from:** Not loopfleet.LoopHealth (the impure fold's full health row) and not RosterEntry (the deduped OUTPUT entry tagged with sources): RosterLoop is only the pure package's import-light input shape.


### RosterSourceLoopRegistry (loopmgr-registry roster source tag)

RosterSourceLoopRegistry is the roster source token marking that the loopmgr job registry (the persisted schedule definitions in tools/loop-registry.json) declares a loop, whether or not its ledger has folded rows yet.

**Distinct from:** Not the superloop registry (registered operator intents; that source is RosterSourceSuperloop) and not the fold source (RosterSourceFold, a measured ledger row): this tag only records a declared loopmgr schedule.


### RosterSourceSuperloop (superloop-registry roster source tag)

RosterSourceSuperloop is the roster source token marking that the super-loop registry claims an entry: either the entry IS a registered intent, or some intent hand-names the loop as a KindLoop member ref (which also sets Named).

**Distinct from:** Not the loop-superloops-registry concept (the registry itself) and not RosterSourceLoopRegistry (a loopmgr schedule declaration): this tag records provenance of a roster entry from registered intents.


### EvictionVictim

cacheprice.EvictionVictim(residents) returns the INDEX of the disaggregated-KV-pool prefix to evict first: the lowest retention DENSITY (DisaggregationRetentionValue per CapacityTokens), compared exactly by cross-multiplication (float-free), ties breaking toward the larger footprint.

**Distinct from:** A pure pricing/SELECTION function over RemoteResident value structs — it only RANKS which prefix should go and returns an index; it never mutates a live cache. Distinct from engine evictors that actually remove a span.


### ContextRestoreEpisodes

The pure fold in internal/dojo that turns the context-span ledger's drop/restore counts into the dojo's one scored episode for the context-restore/restore_recall KPI cell.

**Distinct from:** SCORING of restore recall in the dojo gym - NOT recall (the persisted session core dump) and NOT context-MMU (the live result gate); it reads reduced counts, never the ledger schema, and stays UNMEASURED when restores are unrecorded.


### loadContextSpanLedger

The cmd/fak adapter that reduces the durable gateway-usage ledger's compaction-dropped turns into the ContextSpanLedger counts the pure restore-recall fold consumes.

**Distinct from:** ADAPTER from the on-disk ledger to reduced counts - NOT compaction (provider prefix reuse on the wire) itself and NOT ContextRestoreEpisodes (the pure fold that scores the counts).


### dojo_lever_context_restore

The cmd/fak registration file for the context-restore dojo lever - the RegisterLever seam entry that binds the restore-recall KPI cell into fak dojo run and the --live fold.

**Distinct from:** The LEVER registration (shell wiring) - NOT recall (the persisted session core dump) and NOT claim_context_restore (the internal/dojo claim anchor plus pure fold it delegates to).


### restore_recall

The context-restore cell's KPI metric: the fraction of dropped context spans a later turn pages back in via fak_context_restore, folded from fak's own context-span ledger.

**Distinct from:** A dojo KPI METRIC (a claimed-vs-measured calibration fraction) - NOT recall (the persisted session core-dump subsystem) and NOT a quality gate; it scores UNMEASURED honestly until the ledger records restores.


### claim_context_restore

The internal/dojo file carrying the context-restore cell's one anchored ESTIMATE claim literal plus the pure restore-recall fold over ContextSpanLedger counts.

**Distinct from:** The CLAIM anchor the RSI recalibrate arm rewrites - NOT dojo_lever_context_restore (the cmd/fak lever registration and ledger adapter) and NOT the recall subsystem.


### contextRestoreLever

The cmd/fak lever type that adapts the workspace's durable context-span ledger and emits the context-restore cell's scored episodes for a dojo run.

**Distinct from:** The dojo LEVER (shell adapter object) - NOT context-MMU (gating live result bytes into context) and NOT ContextRestoreEpisodes (the pure fold it calls with reduced counts).


### ContextSpanLedger

The reduced three-field view (dropped spans, restored spans, restore-recorded honesty bit) of fak's durable context-span drop/restore ledger that the restore-recall fold consumes.

**Distinct from:** A reduced COUNT view for scoring - NOT the gateway-usage ledger itself (the on-disk source) and NOT context-MMU; its honesty bit separates a genuine zero-restored from restores-not-recorded.


### MedianCacheReadFraction

Median share of post-compaction window input tokens served as provider cache reads (cached_input_tokens / input_tokens), rolled up per regrowth cohort in the #4768 compact-audit aggregate.

**Distinct from:** A telemetry-derived pricing statistic over Codex rollout token samples - NOT a cache subsystem: it owns no tensors and no eviction, unlike kv-cache; it prices regrowth net of reuse.


### compaction_summary

Regrowth attribution class for the compacted row's replacement_history: the summary transcript the Codex compactor injects into the fresh window after a fire (#4768), measured by row length only.

**Distinct from:** A content-class LABEL in the compact-audit regrowth attribution - NOT CompactionBudget (a threshold knob) and NOT the compaction event itself: it names the injected summary bytes attributed to the post-fire window.


### environment_context

Upstream Codex marker (<environment_context>) that tags an injected environment header message in rollout transcripts; the #4768 regrowth scanner classifies rows containing it as reinjected instruction payload.

**Distinct from:** A verbatim upstream protocol marker matched in row heads - NOT a fak-defined concept and NOT compaction-summary-class (compactor output): it marks setup/instruction payload injected by Codex itself.


### loop_constraint.go (agent constraint seam)

loop_constraint.go is the agent package's loop-side consumer of the out-of-band add-constraint op (#2756): applyConstraints drains the sessionctl constraint mailbox at the turn boundary and carries the tightened floor's standing directive as a system notice, and constraintDenied denies a floor-forbidden tool call before dispatch with a typed receipt carrying the closed CONSTRAINT_* reason.

**Distinct from:** Not loop_redirect.go (which folds an operator redirect into the live OBJECTIVE - what the session pursues): loop_constraint.go folds operator constraints into the live FLOOR - what the session MAY DO - and enforces it per tool call; and not the trajctl metaloop, which walks intents, not turns.


### loop_park.go (agent park seam)

loop_park.go is the agent package's loop-side consumer of the out-of-band operator approve/deny inbox (#2757): parkEscalatedDeny intercepts an ESCALATE-gated deny at the dispatch site when the session's inbox is open, parks it on the sessionctl pending-action queue, and honors the external verdict — approve re-proposes the call through the normal syscall boundary (byte-identical plus the gate's confirm echo, or operator-modified args freshly adjudicated), deny/timeout abort with a typed receipt carrying the closed PARK_* reason.

**Distinct from:** Not loop_constraint.go (which folds operator constraints into the live FLOOR before dispatch): loop_park.go resolves a call the gate ALREADY refused with an ESCALATE disposition, parking it for an EXTERNAL operator verdict instead of returning the deny straight to the model; and not the trajctl metaloop, which walks intents, not turns.


### GatedAction

GatedAction is the sessionctl payload of one call the adjudication gate refused with an ESCALATE disposition, parked on the out-of-band operator inbox (#2757): the tool name, the raw proposed args, the closed refusal reason, and the gate's preview — what an external operator reads to approve or deny the action.

**Distinct from:** The parked PAYLOAD an operator judges, not a gate itself: unlike toolprocgate (which gates background tool process transitions) GatedAction carries an already-refused reversibility/ESCALATE call awaiting an external verdict; the call was never dispatched.


### ParkGatedAction

ParkGatedAction is the sessionctl blocking park op of the out-of-band operator inbox (#2757): it publishes one GatedAction as an addressable pending action and blocks the calling loop until an operator verdict arrives, the park window elapses, or the context is cancelled, witnessing every outcome on the trace's park Next records at the consume point.

**Distinct from:** The loop-side WAIT half of the inbox: unlike ResolveGatedAction (the operator-side verdict op that wakes it) ParkGatedAction is called by the parked arm itself, and unlike GatedAction (the payload) it is the rendezvous; timeout and abort are explicit closed-reason outcomes, never silent drops.


### ResolveGatedAction

ResolveGatedAction is the typed operator approve/deny op of the out-of-band inbox (#2757): it resolves the pending action addressed by id with a ParkVerdict (approve, optionally carrying modified args, or deny), waking the parked loop exactly once; malformed verdicts and unknown/already-resolved ids refuse with the closed PARK_MALFORMED / PARK_UNKNOWN_ACTION reasons.

**Distinct from:** The operator-side WAKE half of the inbox: unlike ParkGatedAction (the loop's blocking wait) it is sent out of band and never blocks; unlike the reversibility gate's _fak_confirm echo (agent self-confirm) it is an EXTERNAL verdict addressed to one specific parked action.


### PendingGatedActions

PendingGatedActions is the read op of the out-of-band operator inbox (#2757): it returns the addressable queue of parked gated actions for a session trace in park order, as a snapshot copy — the list an operator (CLI 'fak session pending', the spine-bound route) consults before resolving one action by id.

**Distinct from:** A read-only LIST of parked GatedAction payloads: unlike ResolveGatedAction it decides nothing, and unlike ConstraintPendingLen (the add-constraint mailbox depth) it enumerates gate-refused calls awaiting an external verdict, keyed for addressable resolution.


### EnableGateParking

EnableGateParking opens the out-of-band operator inbox for one session trace (#2757) and sets its park window (a non-positive timeout selects DefaultParkTimeout): from then on the loop parks ESCALATE-gated denies for that session instead of returning them straight to the model. Parking is opt-in per session so a run with no operator listening keeps the historical gate behavior byte-for-byte.

**Distinct from:** The opt-in SWITCH of the inbox: unlike ParkGatedAction it parks nothing itself, and unlike GateParkingEnabled (the loop-side predicate that reads it) it is the operator-side write that opens the inbox and bounds each park.


### GateParkingEnabled

GateParkingEnabled is the loop-side predicate of the out-of-band operator inbox (#2757): it reports whether EnableGateParking opened the inbox for a session trace — the check the agent dispatch seam makes before parking an ESCALATE-gated deny, so an unopened inbox falls straight through to the historical deny path.

**Distinct from:** The read PREDICATE paired with EnableGateParking's write: it never opens, parks, or resolves anything; a false answer is the byte-for-byte historical loop, not a refusal.


### loop_count

The dispatch codex-loop iteration counter (LoopCount, JSON loop_count) reported per session tick - how many times the codex loop has re-entered for one session.

**Distinct from:** A COUNT of codex-loop re-entries used for gating/telemetry - NOT the loop lease/park (looppark) and NOT a crash-loop detector (loop-crashloop); it only tallies iterations.


### superloop_spinning (walk SPINNING finding)

superloop_spinning is the superloop walk-verdict finding token emitted when at least one member loop is SPINNING: ticking on cadence (live/stale) while its ledger-verified progress high-water mark did not advance (#4956). It binds the closed relay reason RELAY_NO_PROGRESS and demands a revive/redirect, never an auto-replan.

**Distinct from:** Not the dark finding (a member that stopped ticking — the liveness axis) and not the debt finding (aggregate measured debt over the floor): superloop_spinning is the progress axis — the member IS ticking but produced nothing a successor could re-verify via relay.ReadVerifiedProgress.


### attn_gate

The Qwen3.5-hybrid attention output-gate flag (attnGate in the qwen35/GGUF config) - whether the attention block multiplies its output by a learned per-head gate.

**Distinct from:** A per-layer attention ARCHITECTURE flag (output gating) - NOT a guard/gate admission check and NOT the MoE router gate weight (gateWeight); it lives in the model config, not the control plane.


### attn_q_norm

The per-head query-normalization weight tensor (blk.N.attn_q_norm.weight, canonicalized to self_attn.q_norm.weight) that applies QK-norm to the attention query projection in Qwen3.5-class GGUFs.

**Distinct from:** The QUERY-side RMSNorm weight inside attention - NOT the block input norm (attn_norm) and NOT the value projection (attn_v); it normalizes Q before the score.


### full_attention_interval

The Qwen3.5 hybrid-attention layer period (cfg.FullAttentionInterval, GGUF full_attention_interval): every Nth layer runs full (global) attention while the rest run the local/linear path.

**Distinct from:** The CADENCE of full-attention layers in a hybrid stack - NOT the attention kind itself (flashAttention) and NOT the sliding-window size; it is how often full attention recurs.


### cmdPolicy

The argv handler for the fak policy subcommand (func cmdPolicy in cmd/fak) - the CLI entry that inspects and prints the effective admission policy.

**Distinct from:** The COMMAND-dispatch entrypoint for policy inspection - NOT a policy object itself (fetchPolicy/tierPolicy) and NOT the admission decision; it is the argv switch case that routes to policy code.


### FirePolicy

The rsiloop decision rule (rsiloop.FirePolicy) that decides whether an RSI tuning step fires at a given horizon margin (BaselineFirePolicy fires at MinHorizonMargin 0).

**Distinct from:** The FIRE/no-fire rule for the RSI self-tuning loop - NOT an admission policy (fetchPolicy) and NOT a preemption policy; it governs when to trigger a tune, not what to admit.


### FLEET_POLICY_DIR

The env var (FLEET_POLICY_DIR) naming a DIRECTORY of fleet dispatch-policy files loaded at tick preflight.

**Distinct from:** Points at a DIRECTORY of policy files - NOT the single-file FLEET_POLICY_PATH and NOT an in-code admission policy; it is a load-location knob read from the environment.


### FLEET_POLICY_PATH

The env var (FLEET_POLICY_PATH) naming a SINGLE fleet dispatch-policy file loaded at tick preflight.

**Distinct from:** Points at ONE policy FILE - NOT the directory form FLEET_POLICY_DIR and NOT the policy object it loads; a single-path environment knob.


### fak_context

The fak-managed conversation-context lifecycle (the fak_context_ events - session_fak_context_events, context_lifecycle_events) surfaced by vcache/cachevalue status.

**Distinct from:** The fak-side CONTEXT lifecycle object/event stream - NOT the Go command context (commandContext) and NOT a single context budget; it is fak's model of the running conversation window.


### fak_context_planner

The failure-domain/component (fak_context_planner) that plans the fak conversation-context window - the vcache CausePlanning path deciding what context to keep or compact for the next turn.

**Distinct from:** The PLANNER that shapes the fak context window - NOT the base context plan struct (basecontextplan) and NOT the raw context lifecycle events (fak_context); it is the decision component, cited as a WITNESSED planning failure domain.


### context_compacted

The transcript event marker (event_msg/context_compacted) the codex lifecycle emits when a session's context was compacted - the paired real compaction marker used by compactaudit.

**Distinct from:** The AFTER-THE-FACT compaction EVENT marker - NOT the compaction budget (compactionbudget) that triggers it and NOT the regrowth that follows; it records that a compaction happened.


### CostEvictions

The radixkv counter (Stats.CostEvictions) tallying evictions made by the legacy cost-aware eviction strategy.

**Distinct from:** The COUNTER for cost-aware evictions - NOT the eviction policy enum (EvictionLRU) that selects a strategy and NOT a single evicted entry (evictionvictim); it is a running tally.


### EvictionLRU

The radixkv EvictionPolicy enum value (EvictionLRU) selecting least-recently-used eviction ordering.

**Distinct from:** The LRU POLICY selector - NOT the cost-aware counter (CostEvictions) and NOT the act of evicting to a budget (evicttobudget); it names which ordering the evictor uses.


### DISPATCH_POOL

The env var (DISPATCH_POOL) naming which dispatch worker POOL a worker belongs to, forwarded into the guard Pool field alongside DISPATCH_LEASE.

**Distinct from:** A worker's POOL-membership label from the environment - NOT the in-memory seat pool (buildseatpool) and NOT a paged KV pool (pagedkvpool); it is a routing/grouping key, not an allocator.


### DISPATCH_WITNESS_REQUIREMENT

The env var (DISPATCH_WITNESS_REQUIREMENT) declaring what witness a dispatch worker must produce, forwarded into the guard Witness field.

**Distinct from:** The REQUIREMENT setting for a dispatch worker's witness - NOT a witness result (witnessResult) and NOT the witness status (witnessStatus); it states what must be witnessed, not the outcome.


### WitnessedFiles

The logvault accessor (v.WitnessedFiles) returning the set of files a run actually witnessed under a prefix (e.g. dispatch-runs), used by guard_audit.

**Distinct from:** The LIST of files a run witnessed - NOT the witness file PATH (witnessPath) and NOT the witness event (eventWitness); it enumerates witnessed artifacts.


### witnessPath

The filesystem path parameter (witnessPath) to a witness artifact read by graders (GradeWitness, AnalyzeNegatedQA).

**Distinct from:** The PATH to one witness file on disk - NOT the enumerated WitnessedFiles set and NOT the witness content; it is where a grader reads the witness from.


### WitnessToolDescriptors

The conceptbench witness constant (WitnessToolDescriptors, bound to mcp.go toolDescriptors()) naming the ResolveTool referee surface a scenario grades against.

**Distinct from:** A witness POINTER to the tool-descriptor referee surface - NOT a witnessed-files list and NOT a witness result; it identifies WHERE the referee reads tool descriptors.


### errReplicaUnsupported

The sentinel error (errReplicaUnsupported) the portable NUMA replica-store path returns when a caller asks for a NUMA-replicated allocation the platform cannot provide.

**Distinct from:** A capability-UNSUPPORTED sentinel for NUMA replicas - NOT a maturity score (maturityScore) and NOT a general exact-span-supported flag; it marks one allocation feature as absent on this platform.


### FamilyCoverage

The covmatrix/conceptcatalog measure (FamilyCoverage, JSON family_coverage) of how much of one model architecture's family is supported - a per-family fraction on the complete-model-support face.

**Distinct from:** Coverage of a MODEL/architecture family's support - NOT this scorecard's confusable-token coverage and NOT a maturity score; it grades model-support breadth, keyed by architecture.


### Qwen35GDNParityCosineMin

The acceptance floor constant (model.Qwen35GDNParityCosineMin) - the minimum cosine similarity the Qwen3.5 gated-delta-net path must hit against the reference to pass parity.

**Distinct from:** A PARITY acceptance THRESHOLD (cosine floor) for Qwen3.5 GDN - NOT a coverage fraction and NOT a maturity score; it is a device-vs-reference correctness gate value.


### TestVerbTierCoverageIsTotal

The named coverage guarantee (TestVerbTierCoverageIsTotal) referenced by the verb-tier gate: it reds CI when any live dispatch verb resolves to no tier, asserting verb-tier coverage is total.

**Distinct from:** The TOTAL-verb-tier-coverage assertion - NOT a maturity score and NOT model FamilyCoverage; it guarantees every dispatch verb has a declared tier, parsed through the same verb parser the gate uses.


### FAK_AWQ_KERNEL

The env/CPUID gate (FAK_AWQ_KERNEL) that opts into the AWQ 4-bit matmul kernel on amd64; unset leaves the default path provably untouched.

**Distinct from:** The opt-in switch for the AWQ dequant matmul KERNEL - NOT the in-kernel batch/radix decode features and NOT an engine; it selects one quantized matmul implementation.


### FAK_INKERNEL_BATCH

The env gate (FAK_INKERNEL_BATCH) enabling the in-kernel decode path to co-batch concurrent requests; unset decodes byte-identically via serial Session.Step per lane.

**Distinct from:** The in-kernel CO-BATCHING toggle - NOT the AWQ kernel (FAK_AWQ_KERNEL) and NOT the in-kernel radix budget (FAK_INKERNEL_RADIX); it controls cross-request batching in decode.


### FAK_INKERNEL_RADIX

The env gate (FAK_INKERNEL_RADIX, budget via FAK_INKERNEL_RADIX_BUDGET) enabling in-kernel radix prefix reuse in the decode planner.

**Distinct from:** The in-kernel RADIX prefix-reuse toggle - NOT the co-batching gate (FAK_INKERNEL_BATCH) and NOT the AWQ kernel; it governs prefix sharing, sized by an edge-token budget.


### FeatureVDSO

The ablation feature (FeatureVDSO) - the one runtime-settable rung-1 sweep knob selecting the vDSO fast path in the ablation registry.

**Distinct from:** A runtime ablation FEATURE flag for the vDSO fast path - NOT an engine and NOT an in-kernel decode toggle; it is an ablation-registry rung, swept to measure a fast-path's effect.


### KernelLedger

The cachevalue-status source field (Sources.KernelLedger, JSON kernel_ledger) naming the kernel-side ledger a savings report reads from.

**Distinct from:** The KERNEL ledger SOURCE label in a savings report - NOT the savings ledger or usage ledger it sits beside and NOT an engine; it identifies where kernel-side numbers come from.


### NewVLLMEngine

The constructor (NewVLLMEngine(VLLMConfig)) that builds fak's vLLM engine adapter, defaulting an empty WorkerID and normalizing trailing slashes.

**Distinct from:** The CONSTRUCTOR for the vLLM engine adapter - NOT the engine interface (routeEngine/httpEngine) and NOT the vLLM config struct; it instantiates one engine binding.


### FAK_SECRETGATE

The env opt-in (FAK_SECRETGATE) that arms internal/secretgate Admit; when off, Admit is a no-op and only the normgate secret check runs.

**Distinct from:** The opt-in for the SECRET admission gate - NOT the file-admission gate (gateFileAdmissionTree) and NOT a guard lifecycle env; it arms secret scanning specifically.


### FLEET_CODEX_LOOP_GATE

The env gate (FLEET_CODEX_LOOP_GATE) controlling whether the fleet dispatch tick admits the codex-loop step (dispatch_tick_codex_gate).

**Distinct from:** The dispatch-tick GATE for the codex loop - NOT the loop_count metric it guards and NOT a guard lifecycle env; it is an admit/deny switch at the tick.


### FLEET_DOGFOOD_GUARD

The dispatch guard (FLEET_DOGFOOD_GUARD) gating the fleet's self-dogfooding path in dispatch_tick/worker.

**Distinct from:** The GUARD for fleet self-dogfooding - NOT the codex-loop gate (FLEET_CODEX_LOOP_GATE) and NOT the secret gate; it guards the dogfood workflow specifically.


### GATED_UNGRADED

The livecodebench abstain status (GATED_UNGRADED) returned when the grader cannot run against a healthy Docker-isolated host - an honest abstain, never a fabricated zero.

**Distinct from:** An ABSTAIN grade label (gated, ungraded) - NOT a guard/gate admission check and NOT a failing score; it marks a result withheld because the execution host was unusable.


### gateFileAdmissionTree

The hooks tree-twin gate (gateFileAdmissionTree) implementing FILE_ADMISSION in the gate tree, built so it never fires on the grandfathered evidence files.

**Distinct from:** The FILE_ADMISSION gate's tree-twin implementation - NOT the secret gate (FAK_SECRETGATE) and NOT the acceptance gate; it decides file admission inside the hooks gate tree.


### gate_weight

The MoE router gate weight (gateWeights[i]) for a routed expert - the softmax weight AttributeRoute folds one token's expert picks by, and a Qwen3.5 HAL layer weight.

**Distinct from:** The ROUTER gate WEIGHT for a MoE expert - NOT an admission gate/guard and NOT the attention output gate (attnGate); it is a numeric routing coefficient, not a control-plane check.


### guardLifecycleSocketEnv

The constant (guardLifecycleSocketEnv = FAK_GUARD_LIFECYCLE_SOCKET) naming the env var that carries the guard lifecycle IPC socket path used by precompact/stophook.

**Distinct from:** Names the SOCKET-PATH env for guard-lifecycle IPC - NOT the paired token env (guardLifecycleTokenEnv) and NOT an admission gate; it is the transport address, not the auth secret.


### guardLifecycleTokenEnv

The constant (guardLifecycleTokenEnv = FAK_GUARD_LIFECYCLE_TOKEN) naming the env var that carries the guard lifecycle IPC auth token used by precompact/stophook.

**Distinct from:** Names the AUTH-TOKEN env for guard-lifecycle IPC - NOT the socket-path env (guardLifecycleSocketEnv) and NOT an admission gate; it is the shared secret, not the transport address.


### guard_format_layout

The guard exit-summary block LAYOUT (guard_format_layout.go) - the pure text layout of the guard summary onto which color is layered at print time (guardColorizeSummary).

**Distinct from:** The guard summary TEXT layout - NOT a KV tensor layout (mlakvlayout/standardkvlayout) and NOT a model layout; it lays out human-readable guard output, not memory.


### ModelDecision

The modelaccept record (type ModelDecision) capturing the accept/reject decision for one model in the inventory (keyed byDecision).

**Distinct from:** A per-MODEL acceptance decision record - NOT a routing decision (routeDecision) and NOT a tier decision (tierDecision); it records whether one model is accepted into support.


### replanning

The property/act of re-deriving a plan for the same turn (ctxplan PlanView, supervisoragent action) - a deterministic re-admission that reuses the existing admit receipt, not a new authority.

**Distinct from:** Re-deriving the SAME turn's plan (idempotent re-admission) - NOT the initial plan step (planstep) and NOT a new authorization; replanning the same turn twice cannot drift.


### SimulateExpertCacheBatch

Deterministic weight-free simulator replaying B agents advancing one decode step together; streams each distinct (layer,expert) group once per step and reports coalesced page-ins (DistinctStreamed) vs B un-coalesced streams (NaiveStreamed) and CoalesceRatio.

**Distinct from:** Cross-AGENT sibling of SimulateExpertCache: measures the coalescing win when several agents' top-K selections overlap within one shared decode step, where SimulateExpertCache replays a single agent's sequential per-token route against the same LRU.


### ExpertCacheBatchAdmission

Result struct pairing a byte-bounded ExpertCachePlan with a BatchCoalesceTrace: the admitted routed-group capacity plus the cross-agent coalescing evidence measured under exactly that capacity.

**Distinct from:** The batch admission RESULT (plan + coalesce trace over B agents) rather than the single-agent per-token hit/miss trace: it stays weight-free and additionally carries the plan that bounds the step working set.


### AdmitExpertCacheBatchTrace

Computes a byte-bounded routed-group capacity via PlanExpertCache then simulates the supplied B-agent batch steps under that exact capacity, failing closed when one step's distinct union exceeds the plan.

**Distinct from:** Adds the B-agent batch-simulate + step-union bound on top of the bare capacity planner PlanExpertCache: it both plans and admits a coalesced batch trace, where PlanExpertCache only sizes the resident set.


### ReasonArtifactWitness

Go constant naming the shared-task patch-result reason ARTIFACT_WITNESS_MISSING: a disaggregated artifact ref was removed without a digest-shaped deletion witness, so the fold quarantines the patch.

**Distinct from:** The Go identifier for the wire token ARTIFACT_WITNESS_MISSING; names one missing-witness verdict reason on the shared-task co-editing surface rather than a general witness status.


### ARTIFACT_WITNESS_MISSING

Wire reason token on a shared-task patch result: an artifact-ref deletion lacked the digest-shaped deletion witness the contract requires, so the write is held as quarantined.

**Distinct from:** The serialized wire value carried in patch-result JSON; ReasonArtifactWitness is the Go constant that names it.


### ReasonBodyWitness

Go constant naming the shared-task patch-result reason BODY_WITNESS_MISSING: a disaggregated note-body or task-body ref changed without a digest-shaped deletion witness.

**Distinct from:** Body-ref sibling of ReasonArtifactWitness: the same deletion-witness rule applied to note and task body refs instead of artifact refs.


### BODY_WITNESS_MISSING

Wire reason token on a shared-task patch result: a body-ref deletion lacked its digest-shaped deletion witness, so the fold holds the patch as quarantined.

**Distinct from:** Body-ref counterpart of ARTIFACT_WITNESS_MISSING; ReasonBodyWitness is the Go constant naming it.


### ReasonMissingDecision

Go constant naming the shared-task patch-result reason MISSING_DECISION: a replace of /open_decisions/<id>/state addressed a decision ID not present on the record.

**Distinct from:** Names one typed-conflict reason on the shared-task record surface, not a decision log or routing verdict.


### MISSING_DECISION

Wire reason token in shared-task patch-result JSON: the targeted open-decision ID does not exist on the current record, so the resolution write returns a typed conflict.

**Distinct from:** The serialized wire value; ReasonMissingDecision is the Go constant that names it.


### DecisionID

Stable identifier field of one open decision row on a shared task record; append id-newness and state resolution are both keyed by it.

**Distinct from:** A record field naming one decision instance on the co-editing surface, not a journal entry or a routing decision.


### OpenDecisions

Append-only list field /open_decisions on the shared task record holding unresolved Decision rows; stale appends still merge by decision-ID newness.

**Distinct from:** The collection field on the record; DecisionID keys one row inside it.


### ApprovalDecisionID

Store policy field naming the open-decision ID whose approved state unlocks patches the policy holds as APPROVAL_REQUIRED.

**Distinct from:** A policy binding to a decision ID, not a field of the decision row itself.


### replaceDecisionState

Fold helper applying replace /open_decisions/<decision_id>/state: resolves a decision in place and returns a typed conflict for a missing ID.

**Distinct from:** The apply-op helper for decision resolution, distinct from the DecisionID field it dereferences.


### DUPLICATE_DECISION

Wire conflict reason in shared-task patch results: an append proposed an open decision whose DecisionID already exists on the record (id-newness rule).

**Distinct from:** Fires on append of an already-present ID; MISSING_DECISION fires on resolution of an absent ID.


### decisionStatePath

Fold helper parsing an op path of the form /open_decisions/<decision_id>/state into its decision ID.

**Distinct from:** The path parser used by replaceDecisionState, not the state-writing operation itself.


### ReasonUnsupportedPatch

Go constant naming the shared-task patch-result reason UNSUPPORTED_PATCH: the op and path combination falls outside the contract's closed patch grammar.

**Distinct from:** A per-patch denial reason on the co-editing write-gate, not a feature-maturity score.


### UNSUPPORTED_PATCH

Wire reason token in shared-task patch-result JSON: the fold denied a patch whose op or path is outside the supported contract grammar.

**Distinct from:** The serialized wire value; ReasonUnsupportedPatch is the Go constant naming it.


### resultForUnsupported

Fold helper building the denied patch result carrying UNSUPPORTED_PATCH at the record's current revision.

**Distinct from:** The constructor for the unsupported verdict, not the reason token it carries.


### DenialPolicy

Store policy field selecting how the shared-task fold reports a policy-refused patch: deny outright or hold as quarantined.

**Distinct from:** Shapes write-side verdicts on the co-editing gate, not an outbound capability or fetch policy.


### ViewPolicy

Reader-scope redaction policy for shared-task views: MaxScope plus IncludeQuarantined decide what View and EventsView reveal to a caller.

**Distinct from:** Read-side redaction policy; DenialPolicy shapes write-side verdicts on the same surface.


### normalizeViewPolicy

Fold helper defaulting an empty ViewPolicy scope before redaction so an unset reader scope never widens visibility.

**Distinct from:** The normalization helper, not the ViewPolicy type it clamps.


### disaggregatedStore

Fold predicate reporting whether an artifact ref points at a disaggregated store, which makes a digest-shaped deletion witness mandatory on removal.

**Distinct from:** A shared-task witness-rule predicate, not a commit or process guard; it matches this family only through the substring in disaggregated.


### disaggregatedBodyRef

Fold predicate reporting whether a note or task body ref is disaggregated and therefore requires a digest-shaped deletion witness on removal.

**Distinct from:** Body-ref sibling of disaggregatedStore under the same witness rule.


### CacheGiB

coalescebench config field: the resident expert-cache budget in GiB (the RAM tier sitting over SSD) that bounds how many routed (layer,expert) groups stay resident in the deterministic LRU the bench replays through.

**Distinct from:** A bench INPUT KNOB naming the cache SIZE budget (GiB -> whole resident groups via capacityGroups), not a cache mechanism or trace: distinct from enginecache (an engine-level KV/weight cache) and from the SimulateExpertCacheBatch simulator it feeds.


### handleFakAgentSessions (gateway route)

Server.handleFakAgentSessions is the /v1/fak/agent/sessions HTTP handler (#3258, epic #3256): POST a goal and it runs ONE kernel-governed owned-loop agent session (agent.RunGovernedArm over the server's planner) and streams the session back as NDJSON events — session.start, per-call adjudicated call rows, session.end with the ArmMetrics witness.

**Distinct from:** It is the HTTP front door that RUNS a governed agent session end to end and streams its events; unlike applySessionControl (a session-table mutator) or the session capacity/slot vocabulary, it owns no capacity accounting — every tool call it makes crosses the in-kernel syscall boundary.


### IndexLockReclaimDecision

The reap-or-keep verdict for a stale git .git/index.lock: a Reap flag plus a closed-vocabulary reason, decided purely from the commit-lane observer's evidence (lock presence, process-probe success, live-writer count, staleness past the grace window).

**Distinct from:** It is the ACTUATOR's act-or-not verdict on reclaiming an orphaned .git/index.lock, NOT the commit-lane status Verdict (the observer's clear/busy/stale/blocked lane read) and NOT the witness Decision (a CONFIRMED/REFUTED/ABSTAIN evidence-grading verdict).


### session_fatigue

The read-only lens that folds the fak.guard-stop.v1 ledger into a per-gate approval-without-inspection rate and names the gates that have crossed into rubber-stamp territory; flags a gate only when it clears BOTH a fatigue rate and a minimum fire count, so a 1-of-1 approval cannot score a perfect 1.00 and be called evidence.

**Distinct from:** sessionobs scores how well a session is OBSERVED — it grades the telemetry. session_fatigue grades the DECISIONS instead: it measures whether a confirm gate is still carrying a judgement or is being waved through, and it is strictly read-only. Naming a rubber-stamped gate is all it does; coarsening one is the regime mechanism (#2389/#2405) and the autonomy dial (#2759), not this token.


### sessionQuarantineRetentionPolicy

The cmd/fak accessor that reads FAK_SESSION_QUARANTINE_RETENTION and returns the bounded retention policy governing how many, how old and how large the quarantined copies of a corrupt session registry may grow before the recovery path reaps them; an unparseable value returns the conservative default plus an error the caller warns about rather than failing on.

**Distinct from:** DefaultAdmissionPolicy decides whether NEW work is let in; this decides how long WRECKAGE is kept after the fact. It is a housekeeping bound on already-quarantined evidence, never an admission or scheduling decision, and by design it can never refuse or delay a session — a malformed policy degrades to the default instead of failing startup.


### sessionQuarantineRetentionEnv

The cmd/fak constant naming the environment variable that overrides the corrupt-registry quarantine retention policy. It is the NAME of the knob, not the knob's parsed value and not the policy itself.

**Distinct from:** sessionQuarantineRetentionPolicy is the accessor that READS this knob and yields a parsed policy; this constant is only the string key it looks up. Renaming this constant changes which environment variable operators set; changing the policy changes what retention actually does.


### FAK_SESSION_QUARANTINE_RETENTION

The operator-facing environment variable bounding corrupt-registry quarantine evidence: 'off' disables cleanup entirely, 'count=N,age=DURATION,bytes=N' overrides individual dimensions with 0 meaning unbounded, and unset keeps session.DefaultQuarantineRetention. A malformed value warns and falls back to the default; it never prevents MCP startup.

**Distinct from:** This is the WIRE NAME an operator exports, whereas sessionQuarantineRetentionEnv is the Go constant holding that name and sessionQuarantineRetentionPolicy is the parsed result. It bounds quarantined evidence only — it does not affect live session descriptor TTLs, and setting it 'off' retains wreckage rather than disabling recovery.


### claudeSessionUUID

The cmd/fak resolver for the STABLE Claude Code session UUID (the transcript id) that a guard-session descriptor publishes as SessionDescriptor.AgentUUID, so a wip checkpoint's owning session becomes joinable to a live descriptor (#5343). Reads CLAUDE_CODE_SESSION_ID, then CLAUDE_SESSION_ID, then FAK_SESSION_ID; empty when none is set.

**Distinct from:** FAK_SESSION_ID is a DIFFERENT identity, not a fallback spelling of this one: under fak manage a child sees it set to the VOLATILE trace id, which changes every run, so preferring it would publish a populated-looking field that joins to nothing. That is why it is read LAST here. resolveGuardSessionID resolves the guard's own session identity for gating; this resolves the transcript UUID for JOINING checkpoints to descriptors, and the two coincide only by accident.


### MechanismStaleContext

The closed-vocabulary MechanismClass label for an audit finding whose failure mechanism is acting on stale repository state - overwriting, clobbering, or reverting a peer's newer work, or building on an outdated base. It classifies HOW a change failed cross-model audit, never why.

**Distinct from:** STALE_RECALL is a memory-recall verdict: a stored claim whose witness no longer verifies, refused at injection time before it reaches a prompt. MechanismStaleContext is a post-hoc audit finding label about the diff a model already produced, and despite the -Context suffix it names no Go context.Context and no context-window budget: it is one member of a fixed enum, carrying no lifetime, cancellation, or token accounting.


### RenderAuditClusterReport

Renders the cross-model failure-clustering dogfood section from an already-folded AuditClusterResult: a correlation-not-causation fence, then sufficient clusters split from insufficient or confounded ones, then route-policy proposals.

**Distinct from:** RenderLedgerGapReport renders absence - the holes between expected and observed nightrun ledger rows. RenderAuditClusterReport renders present rows grouped by mechanism and author provenance, and is deliberately lossy in one direction: it emits only closed-vocabulary fields (mechanism class, counts, permille rates, typed flags) and never the auditor's free-text reason, so intent-attribution prose in a receipt cannot reach a rendered row.


### SessionKey

The deterministic, surface-independent cross-surface session identity derived by hashing a normalized conversation id under a versioned scheme tag; it doubles as the sessionledger trace name, so continuity rides the ledger's durable hash chain.

**Distinct from:** session-id (SessionID) names one session INSTANCE and is minted per session; SessionKey is DERIVED — a pure function of the conversation identity that yields the same value in any process and after any restart, which is what lets a conversation started on one surface resume on another against the same warm KV prefix. gateway.SessionPrefixKey answers the same question in-process over an in-memory map that evaporates on eviction; SessionKey resolves against the durable ledger instead.


### refuseHostScopedPlanForHostMem

The injectable core of RefuseHostScopedPlanIfTooBig (capacity.go): given a plan and an explicit host (total, free, known), it refuses when the plan's host-scoped demands exceed BudgetAfterHeadroom — the FRACTION-only host budget. Taking the host explicitly is what makes the refusal testable without a live /proc/meminfo.

**Distinct from:** This is the FRACTION-only budget check; refusePagedHostPlanForHostMem is the demand-paged sibling that additionally subtracts an ABSOLUTE page-cache floor. They are not two spellings of one check: the fraction reserve scales with the box, while the paged floor is a property of the backing device's buffered-read cliff, so the two disagree on exactly the hosts where the choice matters. With floorBytes <= 0 the paged form reproduces this one byte-for-byte.


### pagecachefloor

The OS page-cache reserve in fak's host-memory budget: an absolute byte floor held back from MemAvailable so demand-paged (mmap/pread) weights keep a read-through tier.

**Distinct from:** Not the prompt-cache concepts (cache-read/cache-control), which meter provider token reuse; this is host RAM the kernel spends caching file-backed weight pages, and it is an ABSOLUTE floor rather than the fraction BudgetAfterHeadroom applies.


### RefusePagedHostPlanIfTooBig

The demand-paged host fit guard: refuses a MemoryPlan whose host-scoped demands exceed HostBudgetForPagedWeights, the tighter of the fractional headroom budget and MemAvailable minus the absolute page-cache floor.

**Distinct from:** Unlike RefuseHostScopedPlanIfTooBigForHost, which checks the fraction-only budget, this also carves out the page-cache floor, so it refuses a plan that fits the headroom term but would squeeze the read-through tier the mapped weights fault through.


### refusePagedHostPlanForHostMem

The injectable core of RefusePagedHostPlanIfTooBig: takes the host (total, free, known) triple explicitly so the demand-paged refusal is testable without a live /proc/meminfo probe.

**Distinct from:** Unexported test seam, not the entry point: RefusePagedHostPlanIfTooBig probes the live host via HostSystemMemoryInfo and delegates here, mirroring how refuseHostScopedPlanForHostMem backs the fraction-only guard in capacity.go.


### GradeNotDebt

The mode-debt scorer's grade for a dial that is correctly harness-held and model-unreachable: a safety dial the model cannot reach is not implicit-mode debt at all, so it is excluded from the lift worklist entirely rather than ranked at the bottom of it.

**Distinct from:** Distinct from mode_debt, the headline metric this grade REMOVES a dial from. GradeNotDebt is a per-dial verdict meaning 'never rank this'; mode_debt is the fleet-level integer that ranked dials sum into. Also distinct from GradeClean, which means a dial IS debt-eligible and passed all four regime criteria -- GradeNotDebt means the criteria do not apply, so grading such a dial CLEAN would falsely claim it had been lifted.


### NotDebt

The Scorecard roll-up COUNT of dials that graded GradeNotDebt: how many surveyed dials were excluded from the lift worklist as correctly harness-held safety dials. Derived by Score so no consumer re-folds the grades.

**Distinct from:** Distinct from GradeNotDebt, the per-dial grade it counts -- one is a verdict on a single dial, the other an integer over the whole census. Also distinct from the sibling Debt field: Debt is RANKED debt only (Hard+Soft), so NotDebt and Clean both contribute zero to it. Reading NotDebt as a debt figure inverts its meaning, since it counts precisely the dials that are NOT debt.


### egress_posture

The verdict-meta key the adjudicator's egress band stamps on a refusal to name WHICH egress stance produced it -- currently 'restrict', the strict-allowlist posture in which WebFetch flips from default-allowed to allowlist-only. It answers 'why was this host refused' for a reader of the decision journal, distinguishing a posture-driven refusal from a rule-driven one.

**Distinct from:** Distinct from SecretPosture, the adjudicator's OTHER posture knob: SecretPosture governs what happens to credential-shaped spans in tool output (mask, quarantine, fail-closed) and is about DISCLOSURE, while egress_posture governs which destinations a tool call may reach and is about REACHABILITY. Both live on the same Policy and both spell their values as postures, so a reader scanning verdict meta can easily attribute one refusal to the other. Also distinct from the hardwired metadata floor, which produces its own refusal and stamps no egress_posture at all -- absence of this key is how a floor refusal is told apart from an operator-configured one.


### PolicyKnob

A registry ROW in PolicyKnobRegistry naming one amendable policy surface together with its amendment class (FROZEN / RATCHET / GATED_WIDEN / SELF_AMENDABLE) and permitted direction. It is metadata ABOUT a policy field, not a field itself, and carries no runtime value.

**Distinct from:** egress_posture is an actual adjudicator.Policy knob whose value shapes a live decision; PolicyKnob is the registry entry that DESCRIBES such a knob's amendability. Reading a PolicyKnob tells you who may move a surface and which way — never what the surface is currently set to. The registry is exhaustive over exported Policy fields by reflection, so every knob has exactly one PolicyKnob row, but a PolicyKnob row also exists for non-field compiled-in floor elements that are not knobs at all.


### AmendGatedWiden

The amendment class meaning a GATED OPERATOR CHANNEL (overlay, reload, operator escalation) may widen this policy surface, and the agent may never widen it on its own. One of four closed classes alongside FROZEN, RATCHET and SELF_AMENDABLE.

**Distinct from:** A PolicyKnob row carries an AmendGatedWiden value; the class is the vocabulary, the row is the assignment. Against its own siblings: RATCHET permits any authorized channel to tighten and nobody to widen, so it is about DIRECTION; GATED_WIDEN permits widening but restricts WHO, so it is about CHANNEL. A knob can therefore be widened under GATED_WIDEN in a way RATCHET forbids outright — the two are not points on one strictness scale, and reading GATED_WIDEN as 'looser RATCHET' is the specific error this row exists to prevent. SELF_AMENDABLE is the agent-writable frontier and is deliberately empty.


### CoverageEntries

The modver adapter that lifts a flat {module: statement-coverage-percent} map into the map[string]ScoreEntry that Report.JoinScores consumes, tagging each entry ProvenanceWitnessed because the percent is read off a real go-coverprofile artifact rather than modeled (#2467).

**Distinct from:** The LIFT from percent to scored entry (provenance tagging), distinct from CoverageScores which computes the percents by folding a profile statement-weighted per module, and distinct from CoveragePct which is a scorecard's own coverage field rather than a module-version score.


### CoverageScores

The modver fold that decodes a go-coverprofile and returns the flat {module: percent} map, statement-WEIGHTED per module (covered statements over total statements across every file mapping to that module) rather than averaged per file, with repeated file+span blocks merged once and a malformed profile returned as an error instead of a partial fold (#2467).

**Distinct from:** The COMPUTATION of per-module coverage percents from a profile, distinct from CoverageEntries which merely lifts those percents into scored entries for JoinScores, and distinct from the per-file scorecard adapter which takes an arithmetic mean because it has no statement counts to weight by.


### policyExclusion

The operator-configured exclude / include_only gate that drops a discovered account row from the fleet registry, extracted from the inline discovery checks so the discovery path and the seat-stamping path share one decision.

**Distinct from:** Not the policy document itself (policy-manifest) and not a refusal verdict (policyblock): it is the per-row filter decision derived from an already-loaded manifest, and it is the operator-configured counterpart to the structural exclusion checks.


### checkPolicyFile

The fak policy --check entry point that reads the named policy file once and routes it by payload shape: a plain runtime manifest goes to the manifest validator, a fak-org-policy/v1 envelope goes to the signed-envelope verifier.

**Distinct from:** Not the manifest validator and not the envelope verifier it dispatches to, and not the policy document itself: it is the CLI-level router that decides which checker owns the file, keyed on the payload's schema shape rather than on its filename or extension.


### guardSelfTightenOverlay (self-tighten overlay schema)

cmd/fak/guard_self_tighten_overlay.go: the on-disk schema of the overlay the WRAPPED AGENT may author for itself (.fak/agent/self-tighten.json) - a ratchet-only subset of the policy manifest carrying only Deny, BlockHosts and SelfModifyGlobs, each of which can only narrow the floor. It declares no allow / allow_prefix / posture field and is decoded with DisallowUnknownFields, so a forged widening cannot even be spelled: it fails to decode and the overlay is refused wholesale rather than partially applied (#5181, epic #5170 Track F).

**Distinct from:** The AGENT-authored tighten-only overlay schema, the one amendment channel the wrapped agent may write for itself - NOT guardAllowOverlay (operator-authored, widen-only allow lists) and NOT guardDenyOverlay (operator-authored, tighten-only but trusted on arrival). Being agent-authored is exactly why it alone is admitted through the amendment gate instead of being unioned into the floor on sight, and why it deliberately does not live under the self-modify-protected .fak/guard/ tree the operator overlays use.


### guardAdmitSelfTightenProposal

cmd/fak/guard_self_tighten_overlay.go: the admit-and-install step for an agent-authored tighten proposal. It routes the pair (installed floor, proposed floor) through admitSelfTightenOverlay and replaces policy.Runtime.Adjudicator with the proposal ONLY on an admit verdict, refusing with AmendmentFrozenViolation when there is no runtime to amend. A refusal returns the class and reason and leaves the live floor untouched, so a proposal is installed wholesale or not at all (#5411).

**Distinct from:** The INSTALLER that holds the authority to replace the live floor - NOT admitSelfTightenOverlay (guardselftighten), which is a pure classifier that judges a delta and mutates nothing. This is the single place an agent-authored proposal reaches the running adjudicator, and adding it is what turned that classifier from unreachable code into an armed gate. It also differs from guardApplyDenyOverlay, which mutates a runtime with no verdict at all because its overlay is operator-authored and trusted. It deliberately takes an already-built proposal rather than the overlay, so the delta barrier can be exercised with a widening the schema barrier could never spell.


### guardApplySelfTightenOverlay

cmd/fak/guard_self_tighten_overlay.go: the launch-boundary entry point loadGuardCapabilityFloor calls to fold the agent's self-tighten overlay into the capability floor. It builds the union of the installed floor with the overlay, submits that proposal to the amendment gate, and returns the verdict, the amendment class and the count of elements added so the floor-source provenance can record them. An empty overlay short-circuits to a no-op admit without building a proposal, so the ordinary launch stays byte-identical to the pre-overlay floor.

**Distinct from:** The launch-boundary COMPOSITION - union, then gate, then provenance - which owns no verdict of its own: guardAdmitSelfTightenProposal holds the admit decision and the sole write to the live runtime, and this only sequences and reports it. It also differs from guardApplyDenyOverlay, which applies an operator-authored overlay straight onto the runtime with no gate. Its scope is the launch boundary only: mid-session reload paths do not call it, so a running session's behaviour is unchanged.


### guardSharedHookSettingsPath

The cmd/fak/guard.go resolver that answers which single --settings file every guard hook installer must name, so SessionStart, toolproc, Stop and PreCompact converge on one payload instead of each passing the path it was handed.

**Distinct from:** It RESOLVES WHICH FILE the installers share; it does not write one. writeGuardSettingsFileAtomic performs the write, and guardStopHookInstall / the PreCompact analogue are per-hook RESULT RECORDS of an install that has already chosen its path. The distinction is load-bearing rather than cosmetic: #5510 showed that when a caller's payload names a different settings file, Claude's last-wins --settings silently discards guard's entire hook stack, so the identity of this one path is what keeps the stack armed.


### KVPrefixReuseSupported

Config predicate reporting whether a *KVCache is a COMPLETE session prefix for this architecture — i.e. whether cloning the cache carries the whole of what the session already ingested. True for cached architectures whose per-layer K/V rows are the entire state; false for the gemma4 recompute bridge, whose state is the token history and whose cache stays empty.

**Distinct from:** ExactSpanSupported asks whether an engine can EVICT an exact span from a cache it already holds; this asks whether the cache IS the state at all. A recompute architecture answers yes to neither, but for different reasons: eviction is inert because there are no rows, while prefix reuse is unsound because the rows were never where the prefix lived.


### PIN_EVICT_REFUSED (survival-class compaction refusal)

The closed refusal token a history compaction names when the plan it was about to forward would have evicted a page the kernel classes PINNED - the session's active steer, its live continuation seed, or a standing system invariant (#2421). Registered in dos.toml [reasons.PIN_EVICT_REFUSED] and in the internal/agent compaction bail vocabulary; on it the outbound body is forwarded UNCHANGED rather than compacted lossily.

**Distinct from:** A REFUSAL to evict, decided on contract grounds before or after the drop, not an eviction outcome or state: StateEvictable labels a span's residency state and EvictUnderBudget performs a budget-gated eviction, while this token is what the compactor returns when it declines to evict at all.


### ctxplan.ClassEvictable (survival class)

The least-protected member of ctxplan's survival-class vocabulary (#2421): a context page that may be dropped and is then genuinely gone - aged transcript prose. It is the ZERO value of SurvivalClass, so an unstamped or unrecognised page falls to it and can never be silently promoted into the protected set by a kind string the model supplied.

**Distinct from:** A page-KIND-derived survival class in the compaction contract, distinct from StateEvictable, which is a ctxresidency runtime STATE of a span; and distinct from its sibling ClassReplayable, which is equally droppable but whose full bytes stay recoverable through the content-addressed store.


### ctxplan.CheckEviction (survival-class adjudication)

The verification half of the survival-class contract (#2421): given typed pages and the page IDs some other planner proposes to drop, it returns PIN_EVICT_REFUSED when any of them classes PINNED and empty otherwise. It is what makes the guarantee hold for eviction plans ctxplan did not author - a byte splicer on a wire body, say.

**Distinct from:** ADJUDICATES a drop produced elsewhere, distinct from PlanEviction which AUTHORS one, and distinct from KVCache.TryEvict, which performs a fallible exact-span removal rather than judging whether a removal is permitted.


### ctxplan.PlanEviction (survival-class eviction planner)

The planner half of the survival-class contract (#2421): it plans the drop that brings typed pages down to a token budget while honouring each page's class - refusing whole with PIN_EVICT_REFUSED when the PINNED floor alone exceeds the budget, and otherwise shedding the EVICTABLE set before it touches a single REPLAYABLE page.

**Distinct from:** Class-aware and refusal-capable, distinct from EvictUnderBudget, which evicts to a budget with no survival contract and therefore no outcome in which it declines; and distinct from CheckEviction, which judges a plan rather than producing one.


### git_daily_debt

git_daily_debt is the debt key of the git-daily health scorecard (internal/metrics/git_daily_health.go, const GitDailyDebtKey, schema fak-git-daily-health/1): the count of concrete, re-derivable repairs the card found while grading Daily lock-aware Git hygiene from its fak-git-daily/1 ledger over three axes - adoption (is the OS trigger still landing runs), outcome_health (what share of recorded ticks refused a tier or hit an incident), and fold_drift (the trailing streak of non-ok ticks that is the #4602 signature).

**Distinct from:** Unlike climb_ratchet_debt (milestone ratchet rungs) and the other per-card *_debt integers, git_daily_debt counts ONLY defects derivable from the fak-git-daily/1 rows the daily tick appends. It never counts a scheduler fire: a deliberately skipped tick (ALREADY_RAN_TODAY, TICK_BUSY) writes no ledger row, so zero debt means every RECORDED run was healthy, not that nothing was skipped.


### renderGitSpawnReport (bench gitspawn single-run view)

renderGitSpawnReport writes ONE gitspawn measurement run to a writer: per-hot-path git process spawn counts, the window each count was taken over, the per-command table, and the calibration line (injected vs counted) that states this run's own undercount factor.

**Distinct from:** Not renderGitSpawnDelta, which needs two reports and prints movement; renderGitSpawnReport renders a single run's absolute counts and reads no baseline. Not RenderText/RenderContrast (agentdemo walkthrough, sessionobs contrast) -- this is the bench gitspawn spawn-count view.


### renderGitSpawnDelta (bench gitspawn baseline comparison)

renderGitSpawnDelta writes the movement between TWO gitspawn reports to a writer: for each hot path present in both, the baseline spawn count, the current count, and the change -- the view that answers whether a rung actually removed spawns.

**Distinct from:** Not renderGitSpawnReport, which renders one run's absolute counts and reads no baseline; renderGitSpawnDelta requires a loaded baseline report and prints only movement. Not RenderContrast (sessionobs value-vs-waste) -- this compares two runs of the same bench.


### guardCompactionWitness (durable per-session compaction-health row)

cmd/fak/guard_compaction_witness.go guardCompactionWitness: the durable per-session compaction-health row `fak manage` appends at session exit -- {schema, recorded_at, session, anchor_mode, fired, bailed, off, anchor_starved, solvency_forced, shed_tokens, budget, cache_read_at_fire, bail_reasons} folded from the one gateway.Server that guard constructs and tears down per launch, and pinned to the append-only JSONL .fak/nightrun/compaction-health.jsonl so 'did compaction fire for THAT session?' outlives the process that measured it.

**Distinct from:** The post-hoc WITNESS OF RECORD: keyed by session id and readable with no live gateway anywhere. NOT the LIVE in-session verdict (#3099 / observeCompaction, the in-process metrics recorder that dies with the process) and NOT the honest shed ACCOUNTING (#3095 / warmWitness, which prices shed tokens against observed cache_read). Also not CompactSessionReport, which reconstructs compaction health by parsing a rollout transcript file -- this row is folded from the gateway's own counters at exit and changes nothing about how they are measured.


### agentHookDelegate

agentHookDelegate is one registered child process for one agent-LIFECYCLE event (PreToolUse/PostToolUse/Stop): the compiled stand-in for a single hooks entry in .claude/settings.json, carrying the event it serves and an Argv resolver that reports whether the delegate is present on this box at all.

**Distinct from:** Not hooks.Gate or hooks.HygieneGate, which are COMMIT-boundary checks run in-process over a staged diff or tracked tree and whose could-not-run is exit 2; an agentHookDelegate is an out-of-process child on the tool-call path, where exit 2 is the harness BLOCK signal and could-not-run must therefore report as exit 1.


### repoguardArgv

repoguardArgv resolves the repo-guard PreToolUse delegate's child command for a repo root: the compiled tools/.bin/repoguard if present, else the tools/repo_guard.py source, else NOT-PRESENT. It answers only 'what should be executed here, and does it exist', never whether the guard allows the call.

**Distinct from:** Not repoguard itself (the separate cmd/repoguard binary that renders the permission decision on stdout), and not agentHookDelegate (the registry entry that OWNS this resolver alongside its event). repoguardArgv deliberately omits the settings.json wrapper's staleness probe, which blanked a stale binary and fell through to a source path it never confirmed existed -- silently running nothing.


### micro-context

A lightweight logical agent execution context containing only a task delta, bounded mutable state, capabilities, budget, continuation identity, and output contract over an immutable shared agent base.

**Distinct from:** A logical scheduling and isolation unit over one shared base; not a full harness process, not a provider context-window limit, and not context-MMU result-byte admission.


### FakWitnessArgKey

FakWitnessArgKey (internal/gateway/proxy_fill_witness.go) is the reserved wire key "_fak_witness": the external world-state token (git SHA / blob hash / etag / lease epoch) a proxy CLIENT declares on a tool_result (or its call args) to say what state it read at. A declared token is used VERBATIM as the vDSO admission witness for that fill, so an operator can retire every entry admitted under it out of band with fak_revoke using the same token they already know.

**Distinct from:** It is a CLIENT ASSERTION carried on the wire, not a fak-derived or fak-verified value: unlike syspromptmmu.witnessPrefix (a blob-sha256 label fak computes over content it holds) and unlike origin_witness (a taskmgr evidence AXIS naming which witness kind proved a claim), FakWitnessArgKey names the inbound field fak reads and trusts only for revocation identity - fak's own path-scoped refutation still applies on top of it, and a client that declares a constant token can only lose fills, never force a stale serve.


### aggregateAnswers

Typed exhaustive corpus-level gold facts and candidate outputs for state counts, label counts, and chronology top-k grading.

**Distinct from:** aggregateAnswers is the benchmark answer payload; guard-corpus is a policy-test corpus and grade-candidates are scorecard candidates, not expected benchmark facts.


### quantpolicy

Structural policy constraints over quantization capability metadata, including precision bounds, exact approved artifact formats, provenance requirements, and conversion permission.

**Distinct from:** Unlike the general capability floor, quantpolicy decides whether one declared quantized artifact operation satisfies caller-supplied constraints; it neither selects nor runs a quantizer, conversion, runtime, or model kernel.


### CompactionJoinKey

The event-join coordinate a compaction fire shares with the provider usage record for the turn it rewrote, so the fire's provider-side re-warm counters can be PROVEN against one usage row instead of pasted in by caller convention. The zero value is UNSTAMPED: a sample assembled without turn context, which the join passes through verbatim rather than counting as a failed join.

**Distinct from:** A correlation coordinate, not a budget or a threshold: CompactionBudget decides WHETHER a rewrite fires, while CompactionJoinKey only says WHICH provider usage row belongs to a fire that already happened. No verdict reads it -- it selects evidence, it never scores it.


### CompactionJoinResult

The outcome of attempting to bind one compaction fire to the provider usage record sharing its CompactionJoinKey: the joined sample plus whether the binding was PROVEN, left unstamped, or withdrawn because no single usage row matched. It reports the provenance of the provider counters, so an unproven join withdraws them rather than letting an unmatched number stand as evidence.

**Distinct from:** The verdict on the BINDING, not on the compaction: it says whether the provider half may be believed, while the compaction verdict says whether the rewrite paid. CompactionJoinKey is the coordinate looked up; CompactionJoinResult is what the lookup proved.


### FAK_RECALL_MMR

The environment gate that arms MMR redundancy suppression inside journal Recall's top-k selection (#3940). Fail-closed: anything that is not an explicit truthy value leaves Recall's committed provenance-recency-relevance-index ordering byte-identical to pre-#3940, so the suppressor can never silently change what a session recalls.

**Distinct from:** Arms the reranker; it does not tune it. FAK_RECALL_MMR_LAMBDA sets the relevance/diversity trade-off once armed, and cmdRecall is the operator verb that reads the index -- this knob only decides whether the diversity term participates at all. It cannot reorder across provenance tiers under any setting.


### FAK_RECALL_MMR_LAMBDA

The relevance/diversity trade-off weight for armed MMR recall reranking, in [0,1]: 1 is pure relevance (the rerank becomes a no-op reordering), 0 is pure novelty. Out-of-range values clamp and an unparseable one falls back to 0.7, which keeps relevance dominant so the diversity term breaks near-ties rather than dragging a weak-but-novel row past a strongly relevant one.

**Distinct from:** A weight, not a switch: with FAK_RECALL_MMR unset this value is never read at all. And no setting of it -- including 0, the most diversity-aggressive -- can promote a claim above a witnessed row, because the provenance boundary is structural rather than a competing term in the same sum.


### GateFiling

The idea-scout's CONVERSION decision: given the ledger of what the scout already filed and the declared untriaged_cap, GateFiling returns the FilingGate that says whether a live run may create issues at all today. It pauses on stock (more untriaged open filings than the cap) and, as a fail-closed backstop, on a filed-issue index big enough to matter that reports no state.

**Distinct from:** GateFiling decides about the scout's OWN downstream backlog, so it is not a dedup rung (PlanIssues / the filed-stamp index, which decide about one candidate's novelty) and not a threshold like MinScore or MaxIssues (which shape a single day's batch). It is the DECISION function; FilingGate is the record it returns.


### FilingGate

The RECORD GateFiling returns and the idea-scout run result carries as filing_gate: the cap in force, the untriaged stock it was measured against, whether filing is paused, and a reason plus an operator-actionable detail. It is what makes a run that filed nothing because of the backlog distinguishable from one that simply found nothing new.

**Distinct from:** FilingGate is the decision RECORD, not the decision function (GateFiling) and not the ledger the decision reads (BacklogStats). It also is not a dedup outcome: a candidate dropped by a dedup rung is reported in dropped/skipped, while a held FilingGate suppresses the whole day's filing however novel the candidates are.


### GateUntriagedCap

The FilingGate.Reason a paused idea-scout run carries when the scout's OWN untriaged open filings outnumber the declared untriaged_cap. It is a self-releasing brake: the same run files again as soon as the stock is triaged or closed back under the cap, so re-enablement needs no code change and no operator memory.

**Distinct from:** GateUntriagedCap is the STOCK reason -- the backlog was measured and found too large -- as opposed to GateIndexUnclassified, which fires when the backlog could not be measured at all. Neither is a refusal: the run still gathers, still reports its plan, and still exits 0.


### mmrPoolFactor

The multiple of the caller's k that bounds how many already-ranked recall candidates the MMR reranker considers (3x, the borrow's window). Greedy MMR is quadratic in similarity comparisons, so bounding the pool keeps the cost proportional to what the caller actually asked for; candidates past the window keep their baseline order and can only matter when the pool is the whole list.

**Distinct from:** A cost bound on a rerank window, not a set of resources handed out: gradedPool and seatpool name populations something is drawn FROM, while this names how far down an existing ranking the diversity term is allowed to look. Widening it can only change ordering within the window -- never across provenance tiers.


### GateIndexUnclassified

The FilingGate.Reason for the fail-closed arm of the idea-scout conversion gate: a filed-issue index larger than the cap that reports no state for any of its rows cannot be shown to be under the cap, so filing pauses rather than treating an unreadable ledger as an empty backlog.

**Distinct from:** GateIndexUnclassified is about the MEASUREMENT being blind, not about the stock being large (GateUntriagedCap). It is also not the scout-index saturation refusal, which exits 2 because the DEDUP guarantee is at risk; this one holds filing while the conversion evidence is missing and lifts by itself once gh returns state again.


### FP4ClaimRuntimeDelegated

The claim scope in which the checkpoint producer states that execution belongs to an external runtime rather than to whoever reads the metadata. It is a producer ASSERTION carried in the document, and reading it routes the artifact away from in-kernel execution even when every other field would license acceptance.

**Distinct from:** A scope the document declares, not a verdict fak reaches: FP4Delegate is the disposition the adjudicator returns, and this is one of the inputs that can force it. The other claim scopes (artifact, recipe, measured_hardware) say what the numbers describe; only this one reassigns who runs the model.


### FP4Delegate

The disposition meaning the FP4 document is readable and self-consistent but execution belongs to someone else -- because the producer said so, or because the declared hardware lacks native FP4 decode/GEMM. fak routes the artifact rather than claiming it can run it.

**Distinct from:** Distinct from a refusal and from an abstain: delegate asserts the metadata IS understood and valid, and only the executor is elsewhere. Refuse means fak read it and it is wrong; abstain means fak could not read it at all.


### runtime_delegated

The wire value of the runtime-delegated claim scope: the literal string a producer writes into an FP4 metadata document's claim_scope field to say that execution belongs to an external runtime. Being a wire value, it is part of the artifact's public contract and cannot be renamed without breaking documents already written.

**Distinct from:** The string on disk, not the Go constant that names it: FP4ClaimRuntimeDelegated is the identifier a fak build compiles against, and this is what a foreign producer -- which has never seen fak's source -- actually emits.


### FP4HardwareCapability

What the producer says the target device can do natively for FP4: the runtime and accelerator names plus separate native-decode and native-GEMM bits. The two bits are separate because a device can unpack FP4 into a wider type without owning an FP4 tensor-core GEMM, and only the PAIR licenses in-kernel execution.

**Distinct from:** A declared capability of hardware, not a measurement of it and not a permission: fak never probes the device here, it reads what the document claims. A claim of capability that fak cannot honor produces a delegate, not a refusal.


### AdjudicateFP4Metadata

The function that turns an already-parsed FP4 metadata document into one of four typed dispositions -- accept, delegate, abstain, refuse -- with a stable machine-readable reason. It adjudicates against the published NVFP4 and OCP-MXFP4 definitions rather than fak preferences, and there is no fifth implicit 'assume it is fine' path.

**Distinct from:** Adjudicates a document already read; ParseFP4Metadata is the strict reader that produces it, and a parse failure never reaches here. It also decides nothing about a RUN: it says whether these bytes are decodable and by whom, never that a kernel is fast or a speedup is real.


### FP4ReasonSupported

The Go constant naming the reason code carried by an accepting FP4 verdict: every field was readable, self-consistent, and named an envelope this build can decode in-kernel. Every verdict carries a reason so a caller never has to parse the prose detail.

**Distinct from:** Names the reason for an ACCEPT specifically, not the disposition itself: the disposition says what to do, the reason says why. Distinct from the delegation and unsupported-combination reasons, which accompany verdicts that decline to run the artifact here.


### FP4_SUPPORTED

The wire value of the accepting FP4 reason code -- the literal string a caller matches on to learn that a metadata document was fully readable and decodable in-kernel. As a stable machine-readable code it is contract surface: callers switch on it instead of parsing the human-readable detail.

**Distinct from:** The emitted string, not the Go constant: FP4ReasonSupported is what a fak build compiles against; this is what crosses a process or file boundary and must stay stable across builds.


### FP4ReasonUnsupportedCombination

The Go constant naming the reason for refusing an FP4 document whose field tuple contradicts the fixed definition of the format it names -- for example mxfp4 declared with 16-element blocks. It marks a REFUSAL that no future schema version can turn into an acceptance, because the contradiction is with the published format, not with this build.

**Distinct from:** A refusal reason, not an abstention: abstaining says fak could not READ the document (an unknown schema or vocabulary word, which may simply be newer than this binary), while this says fak read it fine and the combination it describes cannot exist.


### FP4_UNSUPPORTED_COMBINATION

The wire value of the unsupported-combination refusal: the literal reason string emitted when an FP4 document's tuple contradicts the published definition of the format it names. Callers match on this code to distinguish a permanently invalid artifact from one this build merely cannot read yet.

**Distinct from:** The emitted string rather than the Go constant FP4ReasonUnsupportedCombination, and distinct from the malformed code: malformed means the document contradicts ITSELF, while this means the document is coherent but describes a format combination that does not exist.


### session intent

A provider-neutral declaration of when a session may start, how much active or elapsed effort it should receive, what terminal evidence or limits stop it, and which bounded lifecycle reactions apply.

**Distinct from:** Unlike SessionBudget, session intent includes activation, completion, recurrence, and lifecycle policy; unlike NativeScheduler, it grants no authority and performs no scheduling or work.


### session stop decision

A deterministic verdict over session intent and measured progress: continue, eligible, complete, timeout, failed, or cancelled, with a receipt-ready reason.

**Distinct from:** Unlike a kernel authorization Decision, this evaluates lifecycle timing and completion only and grants no tool capability; unlike a scheduler Decision, it does not choose work placement.


### Qwen3.8 cache campaign

Versioned exact-model workflow-cache benchmark corpus and fold for the first-class Qwen3.8 default.

**Distinct from:** Unlike the generic cachevalue ledger, this binds Qwen3.8 checkpoint, tokenizer, template, backend, tools, policy, equivalence, invalidation, and per-mode measurements.


### guard disable (one-child break-glass launcher)

The fak guard/manage disable operator subcommand implemented by runGuardDisable: it launches exactly one raw repair child with loud warnings, child-scoped recovery variables, inherited guard-routing removal, and exit-status propagation, then leaves later launches guarded by default.

**Distinct from:** This is an attended one-child repair launcher, not guardDisabled, the GUARD_DISABLED dispatch switch that skips wrapping workers, and not fak guard allow, which edits a capability overlay while the guard remains active.


### ProviderSessionBoundary

A provider-reported replacement conversation that closes one fak trace and opens another inside the same guard process.

**Distinct from:** The explicit provider conversation boundary, not SessionReset/Recontinue, which preserves one logical session across an internal context reset.


### BeginProviderSessionAt

The session-table transition that terminalizes the current fak trace and creates the fresh provider-conversation trace while carrying cumulative envelopes.

**Distinct from:** The atomic mutation that applies ProviderSessionBoundary, not the boundary record itself and not SessionReset/Recontinue within one logical session.


### benchmark license posture

The benchmark environment contract field that states whether a named software license must be verified, must be absent, or is irrelevant before task launch.

**Distinct from:** A task-scoped SOFTWARE-LICENSE requirement, not the tool-admission posture and not a compute receipt's observation of installed software.


### benchmark capability missing

The stable benchmark preflight refusal emitted when a required environment capability is absent or incompatible with the task contract.

**Distinct from:** Absence or identity mismatch, not a present resource below its minimum and not a capability that the contract expressly forbids.


### benchmark capability insufficient

The stable benchmark preflight refusal emitted when a compute resource is present but its observed quantity is below the task minimum.

**Distinct from:** A numeric shortfall in an observed resource, not a wholly absent or incompatible capability and not an expressly forbidden capability.


### benchmark capability forbidden

The stable benchmark preflight refusal emitted when the provider environment exposes a capability the task contract requires to be absent.

**Distinct from:** A prohibited observed capability, not a missing requirement and not a present resource below its minimum.


### ObservationValidityDecision

Reconciliation receipt that binds a read-only child result to its observed state epoch and read set, then marks it current or stale from relevant post-start workspace changes.

**Distinct from:** Unlike StaleFactDecision, which evaluates one recalled memory fact, this decision validates a child observation against workspace-change evidence and carries the exact invalidating paths plus rerun-or-abstain guidance.


### supervision policy

Typed platform-neutral decision policy for fault-domain restart, reattach, hold, and escalation.

**Distinct from:** Unlike dispatch admission policy, it decides process recovery from role, generation, checkpoint, effect certainty, backoff, and restart intensity; it does not authorize execution.


### ExecutionEngine (campaign evidence)

The campaign evidence field naming the runtime that actually executed model math (fak-native or llama.cpp), which the validator uses to decide whether a result is eligible for promotion.

**Distinct from:** It records the model-math executor for evidence promotion, not GatewayURL or EndpointConfig HTTP endpoint identity, a planner or model route, gateway transport, or the compute.Backend hardware/device selected inside that runtime.


### ExecutionEngine values (fak-native / llama.cpp)

The closed qwen38 campaign values selecting either fak-native model math for promotion eligibility or the pinned llama.cpp comparison-only runtime.

**Distinct from:** These are the values carried by ExecutionEngine, not the ExecutionEngine evidence field itself, the generic engine dispatch interface, a backend/device choice, or permission to fall back between runtimes.


### full_context_tokens

Token count in the unscoped full-context counterfactual used as the conservation baseline.

**Distinct from:** A baseline count for one evaluator receipt, not a managed runtime context, context span ledger, or context-window limit.


### guardChildWaitEvent (supervision outcome envelope)

The tagged outcome returned by the guard child wait multiplexer, carrying exactly one completion, restart, time-budget, or resource-containment result back to the supervision loop.

**Distinct from:** It is the WAIT-MULTIPLEXER OUTCOME ENVELOPE, not guardRestartRelaunchCommand, which builds the next child command, and not guardRestartLimitStatus, which formats restart-budget exhaustion status.


### guardCrashRestartDelay (bounded child relaunch backoff)

The function that computes the bounded exponential pause before a child relaunch attempt, shared by generic crash recovery and child resource-containment recovery.

**Distinct from:** It is the RELAUNCH BACKOFF CALCULATOR, not guardRestartLimitStatus, which reports restart-budget state, and not guardRestartRelaunchCommand, which constructs the resumed child command.


### guardSameTraceRelaunchHop (restart lineage constructor)

The constructor for one restart-chain lineage hop whose source, destination, and child remain on the same guard trace, marking recognized resume handbacks engaged and unsupported agents orphaned.

**Distinct from:** It is the RESTART LINEAGE VALUE CONSTRUCTOR, not guardRestartRelaunchCommand, which builds the executable child command, and not guardCrashRestartDelay, which computes how long to wait before relaunch.


### guardEmitRestartHop (restart lineage persistence)

The supervision helper that persists an already-constructed restart-chain hop to the audit journal and reports the same lineage status to the operator surface.

**Distinct from:** It is the RESTART LINEAGE PERSISTENCE AND REPORTING step, not guardSameTraceRelaunchHop, which constructs the hop value, and not guardRestartRelaunchCommand, which builds the child command that the hop describes.


### NativeEngine (model descriptor execution identity)

The modeldescriptor field that pins an onboarding descriptor to fak-native execution before compatibility validation.

**Distinct from:** This descriptor compatibility field is not the campaign ExecutionEngine evidence axis, a backend/device selector, or permission to fall back to an external runtime.


### RenderAdapterReport (benchmark adapter layout)

internal/benchmarkdown RenderAdapterReport renders the shared headings, metadata order, summary table, task table, promotion section, and spacing for an already-projected benchmark adapter report.

**Distinct from:** Owns only the cross-benchmark adapter-report LAYOUT after callers preformat domain rows - NOT the package-specific RenderMarkdown functions that project measured domain reports, and NOT pre-run contract renderers.


### WeightLayout (newmodel native envelope)

newmodel.NativeHardwareEnvelope.WeightLayout names the checkpoint weight packing/storage contract that native obligation compilation must match before allocation, such as gguf-q4-k.

**Distinct from:** This is the checkpoint artifact packing admitted by the new-model compiler, not compute.Layout tensor element ordering and not a runtime KV-cache implementation.


### StateLayout (newmodel native envelope)

newmodel.NativeHardwareEnvelope.StateLayout names the physical organization of recurrent/KV state admitted by obligation compilation, such as contiguous, independent of state kind and residency.

**Distinct from:** It is native planning state-buffer organization, not checkpoint weight packing, not the model KV-cache algorithm/interface, and not ctxplan context layout.


### OpenViking REST adapter

The optional typed HTTP client that lets fak operators call an external OpenViking service through its public REST contract.

**Distinct from:** An interoperability boundary only: it transports health, retrieval, session capture, and commit calls; it is not fak-native context storage, contextq materialization, ctxplan optimization, or recall persistence.


### execViaKernel (agent-loop syscall adapter)

The agent-loop adapter that lowers one admitted model tool call into abi.ToolCall, invokes the fak kernel syscall, and converts its verdict/result into model-visible tool content.

**Distinct from:** The per-call ADAPTER at the agent loop boundary — not the kernel coordinator itself and not an inference engine backend.


### guardSessionStartInstall (guard SessionStart hook install receipt)

guardSessionStartInstall is the typed receipt returned while fak guard installs a provider-native SessionStart affordance hook; it records whether the hook was applied, how it is managed, and which settings/state paths own it.

**Distinct from:** The INSTALL RECEIPT describing one launch-time hook mutation, not guardsessionstart (the hook command that later emits the first-turn hint) and not a running session record.


### loadTUISessions (TUI session-list loader)

loadTUISessions is the cmd/fak input adapter that loads a gateway SessionListResponse either from an operator-provided JSON snapshot or from the live /v1/fak/sessions endpoint for TUI rendering and control selection.

**Distinct from:** The INPUT LOADER for the TUI session view, not tuiSessionReport (the derived render model), tuiSessionsSchema (its JSON schema tag), or the gateway session runtime itself.


### guardJSON (--guard-json TUI artifact inputs)

guardJSON is the repeatable cmd/fak flag binding that carries operator-supplied guard artifact JSON paths into the standalone guard pane or the overview guard card.

**Distinct from:** The INPUT PATH LIST for TUI rendering, not tuiGuardReport (the parsed pane model), tuiGuardSchema (its payload schema), or a live guard gate.


### sessionsJSON (--sessions-json TUI snapshot input)

sessionsJSON is the cmd/fak flag binding that carries an operator-supplied SessionListResponse snapshot path into the standalone sessions view or the overview sessions card.

**Distinct from:** The INPUT FILE PATH selecting a read-only session snapshot, not loadTUISessions (the adapter that reads it or calls the live gateway), tuiSessionReport (the derived render model), or tuiSessionsSchema (the output schema tag).


### InKernelPlannerConfig (native planner construction settings)

agent.InKernelPlannerConfig is the typed bundle of native planner and session settings fixed at construction, including Qwen Q4_K prefill chunking, Qwen3.5 Metal GDN sequencing, CPU expert offload, and the Q4_K gate/up output slab.

**Distinct from:** This is configuration supplied to a native planner, not InKernelPlanner, the runtime planner that owns request state, and not NewInKernelPlannerWithConfig, the constructor that consumes the configuration.


### NewInKernelPlannerWithConfig (typed native planner constructor)

agent.NewInKernelPlannerWithConfig constructs the local in-kernel planner from a loaded model plus an explicit InKernelPlannerConfig, fixing operator-selected native behavior before any request session is created.

**Distinct from:** This is the constructor that consumes explicit typed settings; unlike NewInKernelPlanner it is not the compatibility constructor with only CPU-offload input, and unlike InKernelPlannerConfig it performs construction rather than representing settings.


### Gateway in-kernel planner configuration binding

The serveNativePlannerConfig production seam binds explicit serve flags into agent.InKernelPlannerConfig and then into gateway.Config.InKernelPlanner before the gateway constructs its native planner.

**Distinct from:** This is the CLI-to-gateway binding for typed native settings, not agent.InKernelPlannerConfig, the reusable settings type, and not InKernelPlanner, the runtime request planner.


### Gateway configured in-kernel planner construction

newInKernelChatPlanner carries gateway.Config.InKernelPlanner into agent.NewInKernelPlannerWithConfig when the gateway selects its in-process native chat planner.

**Distinct from:** This is the gateway factory boundary that invokes the typed constructor; it is not NewInKernelPlannerWithConfig itself and not the resulting InKernelPlanner runtime.


### Q4_K gate/up output slab (session-owned Metal buffer)

Q4KGateUpOutputSlab is the explicit session setting that reuses one bounded Metal output buffer across eligible Q4_K gate and up MLP projections within that session.

**Distinct from:** This is a session-owned output-buffer reuse mechanism for two FFN projections, not gate_up_proj, the fused model weight, and not an adjudication or policy gate.


### Qwen Q4_K native prefill chunk setting

InKernelPlannerConfig.QwenQ4KPrefillChunkTokens is the explicit 128..8192-token bound used to partition Qwen Q4_K native prefill calls before decode; operator flags populate it at planner construction.

**Distinct from:** This is one bounded prefill tuning value, not the full InKernelPlannerConfig bundle and not InKernelPlanner, the runtime planner consuming that bundle. The former FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS ambient spelling is a retired alias, not a second authority.


### performance-RSI loop-turn scorer

ScoreLoopTurn evaluates one completed loop run's strict performance-RSI evidence and emits the nonfatal loop-turn receipt.

**Distinct from:** Unlike loopscore, which grades loop durability across a corpus, this scorer evaluates one current run's performance evidence after its child exits.


### routeGuardOperatorSubcommand (guard operator-command dispatcher)

routeGuardOperatorSubcommand is the cmd/fak control-flow seam that recognizes guard allow, deny, disable, resume, and sessions before the wrapped-agent FlagSet is constructed, then reports whether it consumed the invocation.

**Distinct from:** The OPERATOR-SUBCOMMAND DISPATCHER before guarded launch parsing, not cmdGuard (the guarded-agent launcher) and not guardSessionStartInstall (the launch-time SessionStart hook mutation receipt).


### inkernel_expert_spill.go (served graded expert-spill seam)

inkernel_expert_spill.go is the agent planner seam that resolves a loaded model’s graded MoE expert-spill placement once from operator intent and measured device budget, then applies that placement to each request session.

**Distinct from:** The SERVE-SIDE LIFECYCLE BRIDGE for graded expert spill, not InKernelPlanner (the whole native request planner) and not its Qwen prefill-chunk control.


### InKernelQwenQ4KPrefillChunkConfigError (deferred native-prefill bounds error)

InKernelQwenQ4KPrefillChunkConfigError is the typed error retained when an explicit resident-Qwen Q4_K prefill chunk size lies outside 128..8192; a targeted request returns it before tokenization or model execution.

**Distinct from:** The FAIL-CLOSED ERROR VALUE for an invalid bound, not QwenQ4KPrefillChunkTokens (the requested planner setting) and not InKernelPlanner (the planner retaining it).


### kvSpanEvict (planner KV-quarantine bridge gate)

kvSpanEvict is the planner-scoped enablement bit set only when FAK_INKERNEL_KVMMU opts in on the CPU model path; guarded eviction code checks it before rebuilding a session and evicting a quarantined tool-result K/V span.

**Distinct from:** The ENABLEMENT GATE for the live planner bridge, not KVSpanEvictor (the public quarantine-bridge interface) and not KVCache.Evict (the model cache mutation it eventually invokes).


### PerformanceRSIDebt (performance-RSI unresolved-dimension debt)

PerformanceRSIDebt is the perfrsiscore report metric counting canonical performance dimensions that remain BEHIND or UNKNOWN. Build derives it as Behind plus Unknown, uses zero as the clean condition, and passes the already-computed value to human and Markdown renderers.

**Distinct from:** Use PerformanceRSIDebt for the unresolved-dimension COUNT computed by the performance-RSI scorecard. Unlike performance-rsi-scorecard, it is the metric emitted by that evaluator; unlike the generic scorecard renderer, it is computed before presentation and RenderHuman or RenderMarkdown only project it.


### validateCandidate (study-adjacency candidate receipt validator)

internal/studyadjacency.validateCandidate validates one recorded study candidate: required identity and rationale, a vLLM mechanism link or frontier-changing contrast, nonempty repository links, declared repositories, duplicate-link rejection, and inclusion of the owning study member.

**Distinct from:** Use this package-local validateCandidate for STUDY-ADJACENCY receipt and repository-link integrity. Unlike issuecontract ReviewCandidate, it does not score dispatchability; unlike placementtax validateCandidate, it does not judge a native execution plan against topology, SLO, or provenance constraints.


### validateCandidate (placement-tax plan feasibility validator)

internal/placementtax.validateCandidate validates one PlanCandidate against its topology contract: nonblank identity and rationales, fak-native engine ownership, valid parallelism strategy, nonempty hierarchy, measured or estimated provenance, SLO validity, and consistent cross-domain provenance.

**Distinct from:** Use this package-local validateCandidate for PLACEMENT-TAX plan feasibility before scoring alternatives. Unlike studyadjacency validateCandidate, it does not validate study receipts or repository ownership; unlike placementCandidates, it validates one fully formed native plan rather than building an ungraded model pool.


### SetKVPreemptionPolicy (scheduler mutator)

SetKVPreemptionPolicy installs a normalized NativePreemptionPolicy on a NativeScheduler and initializes or reconfigures its paged-KV block capacity before execution.

**Distinct from:** It is the scheduler mutation that applies a preemption configuration, not the NativePreemptionPolicy value that describes swap or recompute behavior and not an authorization policy.


### NativeSessionRestored (readmitted-session lifecycle)

NativeSessionRestored is the NativeSessionLifecycle value assigned when the scheduler rebuilds a model session during swap readmission and imports the preserved KV state.

**Distinct from:** It identifies the restored member of the lifecycle enum, not the NativeSessionLifecycle type itself and not a fresh or recomputed session.


### guardCodexAuthManagementCommand

Recognizes the exact Codex CLI authentication-management commands that must execute without FAK provider or credential injection.

**Distinct from:** Unlike ordinary Codex launches or the one-child guard-disable break-glass launcher, this narrow command class manages Codex's own login state and bypasses only FAK auth/config injection; it does not disable guarding for other Codex commands.
