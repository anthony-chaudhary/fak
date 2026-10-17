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

## The rest of the glossary

The remaining families and the machine-positioned entries live in companion
pages so this front page stays a bounded read; every entry is preserved
verbatim and every anchor still resolves:

- [witness/evidence, session/scheduling, gateway/engine, policy/authorization, and context-management families](glossary-families-a.md)
- [scorecard/debt, eviction, decision, render/materialize, plan, pool, layout, loop, trajectory-control, dev-tier/operator, and cross-cluster families](glossary-families-b.md)
- [positioned concept entries, 1 of 4](glossary-positions-1.md) · [2 of 4](glossary-positions-2.md) · [3 of 4](glossary-positions-3.md) · [4 of 4](glossary-positions-4.md)

New entries appended by the positioning tools (`fak concept position`) continue
to land at the tail of THIS page, which is why late-positioned symbols appear
after the Read next section; periodically move landed entries down into the
companion positioned-entries pages.

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


## Entries retained on this page

Two positioned entries stay on this page verbatim because the managed-docs
classifier pins their exact lines to this path; everything else moved to the
companion pages indexed above:

### guard disable (one-child break-glass launcher)

The fak guard/manage disable operator subcommand implemented by runGuardDisable: it launches exactly one raw repair child with loud warnings, child-scoped recovery variables, inherited guard-routing removal, and exit-status propagation, then leaves later launches guarded by default.

**Distinct from:** This is an attended one-child repair launcher, not guardDisabled, the GUARD_DISABLED dispatch switch that skips wrapping workers, and not fak guard allow, which edits a capability overlay while the guard remains active.



### guardSessionStartInstall (guard SessionStart hook install receipt)

guardSessionStartInstall is the typed receipt returned while fak guard installs a provider-native SessionStart affordance hook; it records whether the hook was applied, how it is managed, and which settings/state paths own it.

**Distinct from:** The INSTALL RECEIPT describing one launch-time hook mutation, not guardsessionstart (the hook command that later emits the first-turn hint) and not a running session record.




### CompactionEvents (transcript content retention receipts)

Per-file Codex compacted event receipts measuring supported explicit transcript UTF8 content before and after replacement history.

**Distinct from:** Counts reconstructed transcript content bytes with incomplete-history coverage; does not measure provider prompt tokens, attribute the compaction engine, or stitch separate transcript fragments.


### MaxKernelBufferBindCount (Metal ICB descriptor)

MaxKernelBufferBindCount is the per-command Metal indirect-command-buffer descriptor limit for bound kernel buffer arguments.

**Distinct from:** It bounds buffer argument slots encoded by one Metal ICB command; kernel denotes the broader compute implementation, execution path, or subsystem.


### Qwen35 MTP spec-decode unsupported error

`Qwen35MTPSpecDecodeUnsupportedError` is the typed model-layer admission refusal returned before native Qwen3.8 MTP depth-N decoding mutates an unsupported target session.

**Distinct from:** It reports that the requested MTP target or draft envelope is unsupported; it does not claim general support and it is not `Qwen35MTPForwardError`, which reports failure while executing an admitted MTP forward.


### WithPolicySnapshot (agent run option)

The agent loop run option that pins one caller-supplied adjudicator policy value so Configure cannot replace the capability floor during that run, including native child-task execution.

**Distinct from:** It binds an already-compiled policy to one RunArm invocation; it neither compiles a manifest nor mutates the process-global adjudicator policy.


### refuseContextLength (native admission helper)

refuseContextLength is the native planner helper that compares an exact rendered prompt token count plus its reserved output tokens with the effective model context limit before prefill or decode.

**Distinct from:** This is the enforcement helper, not context-window (the hard budget it reads) and not native-context-refusal (the typed error value it returns when the request exceeds that budget).


### FAK_STREAM_INKERNEL_PER_TOKEN

Default-off environment switch that makes InKernelPlanner.CompleteStream forward committed token pieces during decode; disabled, the sink receives the completed content in one buffered delta.

**Distinct from:** This switch selects emission timing for the concrete in-kernel planner; StreamingPlanner is the generic capability interface implemented by multiple planners and does not enable this opt-in policy.


### FAK_ABLATE_UNCACHED_TRIM

Environment switch that supplies the default result-elision byte threshold when no positive explicit threshold is configured.

**Distinct from:** FeatureUncachedTrim names the ablation mechanism; this environment variable configures its default threshold and does not prove that a request used it.


### FeatureMockPlanner

Serve feature catalog identifier for the deterministic mock-planner configuration and its evaluated enablement state.

**Distinct from:** MockPlanner is the executable offline planner; FeatureMockPlanner is the catalog key describing whether that planner is configured or installed.


### strixPreflight

Serve runtime field retaining detected Strix appliance capabilities for gateway configuration and hardware-derived feature provenance.

**Distinct from:** DispatchPreflight admits a worker launch; strixPreflight records serving-host detection and does not grant dispatch or tool-call permission.


### chatPlanner

chatPlanner selects the planner stored in the admitted request binding, falling back to the gateway boot planner only when no binding exists.

**Distinct from:** DualPlanner selects between its local and proxy legs from a model alias; chatPlanner selects an already admitted roster-specific transport without making a new account choice.


### chatRouteFromContext

chatRouteFromContext retrieves the admitted chat account binding from the current request context, or nil when the request uses the boot planner unchanged.

**Distinct from:** principalFromContext extracts the authenticated tenant identity used to admit an account; chatRouteFromContext retrieves the resulting transport binding and does not authenticate or resolve an account.


### FeatureActivationTracker

The request-owned, concurrent-safe observation of which closed ServeFeature identifiers actually executed. It captures configured active/standby identifiers at ingress separately from accepted uses and freezes the used set at request completion.

**Distinct from:** FeatureCatalog describes evaluated configuration and provenance; this tracker records execution on one request. It neither enables a feature nor measures savings. LookupReceipt supplies ephemeral real-vDSO source identity, while FeatureProofReceipt records bounded quantitative evidence in the separate proof surface. None of these observations authorizes a tool or settles a financial claim.


### X-Fak-Features request observation headers and final trailer

The HTTP projection of one FeatureActivationTracker: X-Fak-Features-Enabled lists ingress configuration, X-Fak-Features-Used snapshots accepted use before the first final response commit, and the declared X-Fak-Features-Used-Final trailer reports completed execution after streaming finishes.

**Distinct from:** Initial headers cannot report later execution and remain immutable after commit. The final trailer extends observation without buffering live SSE; an incomplete worker does not receive a complete final-use value. These identifier lists carry neither retained FeatureProofReceipt measurements nor settlement receipts, and Enabled never implies Used.


### fak_gateway_feature_used_requests_total

The cumulative Prometheus counter of requests that actually used each closed ServeFeature identifier. Request finalization folds each used identifier once, even when the mechanism executes repeatedly within the request; the feature label is bounded by the catalog.

**Distinct from:** This counts requests with observed use, not configured features, number of individual executions, savings, or proof-verification outcomes. The gateway prefix identifies the HTTP serving subsystem and does not make this a security gate or a policy verdict.


### LookupReceipt (per-call real vDSO source identity)

An opaque, call-local receipt written only by successful real VDSO.Lookup execution and bound to the exact returned abi.Result pointer. WithLookupReceipt transports it through context, including production tier wrappers, and Matches checks that identity after the caller's acceptance gates. A later real lookup clears earlier evidence, including on a miss.

**Distinct from:** This short-lived identity check retains no payload serialization, historical measurement, or settlement authority. It differs from a world-state witness that binds cache validity, from FeatureCatalog configuration, from activation identifier lists, and from retained FeatureProofReceipt quantitative evidence. Matching the real result does not replace freshness, screening, or result-admission checks; arbitrary FastPath metadata cannot manufacture the receipt.


### NativeCompactionObservation

A bounded value summary of the actual native typed-message compaction cut, published to request observers only after successful native generation consumes the prepared prompt. It records exact typed-message JSON byte counts and hashes, explicitly estimated message tokens, and shared whole-message regions without retaining prompt or restore payloads.

**Distinct from:** FeatureActivationTracker retains used identifiers, while this producer summary supplies quantities to the separate FeatureProofReceipt collector. Its shared regions do not certify KV cache anchors, estimates are not tokenizer counts, and byte reduction is not measured time savings. EncodePrompt, idle cuts, and failed generation publish no observation; proof capture failure does not erase actual compaction use. LookupReceipt identifies a real vDSO result and has a different producer.


### FeatureProofReceipt

Bounded process-local observation of applied Anthropic or native typed compaction, result elision, or accepted real-vDSO serving, retaining quantitative digests or an explicit proof-generation failure.

**Distinct from:** Each receipt ties an executed feature to byte or lookup evidence. Catalog configuration and activation headers alone do not prove savings; the transient LookupReceipt establishes only exact-result provenance. This observation does not persist settlement evidence or authorize execution.


### FeatureCacheProof

Quantitative accepted-vDSO-hit evidence binding the actual gateway call identity digest to its served payload digest; measured resident entries also carry a signed historical-engine-minus-current-lookup estimate.

**Distinct from:** LookupReceipt authenticates an exact result transiently. This bounded proof records the accepted hit and explicit timing basis without claiming an internal cache key or a measured counterfactual saving. Unmeasured entries retain null timing quantities.


### FeatureRewriteWitness

Input to the pure Anthropic byte-rewrite verifier specifying protected prefix and suffix bounds, exact claimed byte delta and labeled token estimates.

**Distinct from:** The witness is verifier input rather than proof that execution occurred. The structural producer selects the real regions; the verifier checks their byte equality and measured delta before a FeatureProofReceipt can retain the result. Native typed-message observations use separately labeled representation evidence.


### FieldCompletionTiming

Opaque completion attestation minted by the kernel around an actual successful engine execution, binding a measured wall-clock duration to the exact call and result contents.

**Distinct from:** Gateway counters aggregate observations; this transient attestation authenticates one prior execution measurement. A cache-hit duration estimate additionally requires the exact resident entry and current lookup receipt; it does not measure an avoided counterfactual execution.


### ExpertCacheTier

The closed three-rung vocabulary (VRAM/L2/NVMe) naming where a routed expert's weights live, mapped onto the pkg/moecache tier names.

**Distinct from:** Names the TIER a routed expert resides on, not a residency plan (ExpertCachePlan) or a batch-admission result: it is the enum a policy transitions between, not a plan or a trace.


### ExpertTierPolicy

The opt-in policy that gates the two transitions between the expert waterfall's L2 and NVMe rungs: an NVMe->L2 fill on a checkpoint miss and its inverse L2->NVMe demotion that never drops a pinned expert.

**Distinct from:** Selects the PROMOTION/DEMOTION transition between the host L2 and backing-store rungs (fill-on-miss, demote-without-dropping-pins), not the ring's victim-eviction ranking (ExpertRingPolicy) nor the ctx-residency TierPolicy.


### ExpertTierStats

The session-level, tier-shaped fold of the device ring ledger, the checkpoint-tier ledger and the explicit L2 tier policy's transition counters, one counter block per rung.

**Distinct from:** FOLDS per-rung traffic and transition counters into one three-tier record; unlike ExpertRingStats or ExpertCheckpointStats it is not one rung's own ledger, and unlike ExpertCachePlan it carries no admission decision.


### ExpertTierLedger

The record of ONE expert tier transition: the from/to rungs, whether a promotion or demotion happened, how many candidates a demotion pass preserved because they were pinned, and why.

**Distinct from:** Records a SINGLE transition decision with its reason, whereas ExpertTierStats aggregates many into counters; it is an event, not a running ledger of the whole tier.


### ExpertNVMeGeometry

The aligned read geometry of one NVMe expert fault: the alignment granularity plus the start/end and byte length a fault's unaligned range rounds to, stated without issuing the read.

**Distinct from:** States the ALIGNED SPAN a fault would move (round-down start, round-up end) as pure geometry, rather than naming a page-cache entry or modeling the page cache that serves it.


### readExpertPageCache

The #1302 page-cache-aware expert read: reads one expert's stride through a mapped region, serving a warm fault zero-copy and issuing an aligned device ReadAt only on a cold fault.

**Distinct from:** Names the READ OPERATION (one expert's bytes, warm-from-mapping vs cold-aligned-device) rather than a page-cache entry, a slot count, or the byte floor a page cache must clear to pay for itself.


### CandidateKind

The closed promotion-candidate taxonomy a scratchpad artifact is classified into: unfinished-spine, blocked, discovered-edge-case, deferred-caveat, next-checkable-step. A CandidateKind is the deterministic Classify verdict over one parked artifact's filename/leading block.

**Distinct from:** It is the CLOSED taxonomy value for a parked scratchpad artifact that looks like an unfiled session flag, NOT the plan/spawn Candidate (a scored span the planner may keep resident) and NOT a generic kind enum.


### expertGateUp (V4.1 optional device gate/up callback)

The optional callback field v41ForwardState.expertGateUp (v41_forward.go), bound by Session.v41ExpertGateUpFunc when the session advertises a DeviceMemory backend; both the token-major and expert-major V4.1 routed contractions offer each routed row to it so the device gate/up + SwiGLU run on the backend while the host down contraction stays on the host.

**Distinct from:** The OPTIONAL per-row device gate/up + SwiGLU callback a V4.1 session installs (v41ForwardState.expertGateUp); a compute-engine SEAM that runs a routed expert's gate/up projections on the device, NOT the adjudication guard, NOT the FFN load-time gateUp locals, and NOT a routing gate.


### verifyTopologyCandidate (land-verify candidate checkout)

internal/workerworktree.verifyTopologyCandidate materializes the exact tree a land is about to publish as a detached verify-only git worktree named .fak-cand-validate-<pid>-<start>-<n> beside the repository (or in the system temp directory), runs the verify hook in it, and removes it afterwards.

**Distinct from:** A disposable VERIFY CHECKOUT of a land's prospective tree - NOT the planner Candidate (a scored context span), not a dispatch or issue candidate, and not the worker worktree whose diff is being landed.


### SweepTopologyCandidates (leaked verify-checkout collector)

internal/workerworktree.SweepTopologyCandidates classifies every .fak-cand-validate-* verify checkout in a parent directory and, under Apply, removes those whose owner process is gone, then prunes their git worktree registrations.

**Distinct from:** Collects leaked land-verify checkouts by owner liveness - NOT the owner-stamped worker-worktree GC and not the lease-based cold reap, which only see fak-worker-wt-* worktrees.


### CandidateSweepOptions (verify-checkout sweep knobs)

internal/workerworktree.CandidateSweepOptions carries the clock, legacy age floor, apply opt-in, removal limit, and process-liveness probes for SweepTopologyCandidates.

**Distinct from:** Options for the verify-checkout sweep only - NOT GCOptions (owner-stamped worker GC) and not a planner candidate configuration.


### LegacyCandidateMaxAge (verify-checkout legacy age floor)

internal/workerworktree.LegacyCandidateMaxAge (2h) is how long a verify checkout whose name carries no owner pid/start must sit untouched before the sweep collects it.

**Distinct from:** An age floor only for owner-less legacy verify checkouts - NOT DefaultColdAgeFloor (worker-worktree cold reap) and not a planner retry or budget window.


### TopologyCandidateParent (verify-checkout parent directory)

internal/workerworktree.TopologyCandidateParent returns where a repository's verify checkouts are created: beside the repository when its go.work escapes it, otherwise the system temp directory.

**Distinct from:** Resolves the directory verify checkouts live in - NOT the worker-worktree root (DefaultRoot) and not a path to one checkout.


### topologyCandidatePattern (owner-named verify-checkout name)

internal/workerworktree.topologyCandidatePattern is the os.MkdirTemp pattern .fak-cand-validate-<pid>-<start base36>-* that encodes the creating process identity in each verify checkout name.

**Distinct from:** The naming scheme that makes owner liveness provable after a kill - NOT the legacy bare .fak-cand-validate-* name and not a worker-worktree owner stamp sidecar.


### cleanupTopologyCandidate (verify-checkout deferred cleanup)

internal/workerworktree.cleanupTopologyCandidate is verifyTopologyCandidate's deferred cleanup: git worktree remove, then a retried directory removal, then prune only once the directory is gone.

**Distinct from:** The in-process cleanup of the checkout this land created - NOT the cross-process sweep that collects checkouts whose land was killed before this ran.


### sweepTopologyCandidatesBeforeCreate (land-path verify-checkout sweep)

internal/workerworktree.sweepTopologyCandidatesBeforeCreate is the rate-limited, two-removal-bounded sweep a land runs before creating its own verify checkout.

**Distinct from:** The bounded in-land trigger of SweepTopologyCandidates - NOT the unbounded operator verb fak worktree worker gc --candidates.


### MCP session bearer (guard launch capability)

The per-launch random credential written into the guarded child's MCP HTTP config and hash-validated by the gateway to prove that an MCP request came from that launched child; it grants no workspace lease ownership.

**Distinct from:** It proves guard-child launch provenance for MCP requests; it is not the gateway RequireKey/defaultGatewayBearerToken traffic credential and not the stored SHA-256 digest.


### sharedCache (requested worker Go build-cache path)

sharedCache is the requested Go build-cache path read by internal/workerworktree.WorktreeEnv from an explicit caller-map FAK_SHARED_GOCACHE value, otherwise the process environment. An explicit empty value suppresses ambient opt-in. After trimming, an absolute path without NUL selects GOCACHE; invalid or disabled input retains the worktree-local default.

**Distinct from:** This is the worker opt-in path selection value, not GOCACHE itself, a runtime KV or provider cache, a warm-seeding copy, or cache-retention authority. GOTMPDIR remains worktree-private and the worktree reaper does not own an external shared cache.


### SharedGoCacheEnv (Go constant naming worker shared-cache opt-in)

SharedGoCacheEnv is the exported workerworktree Go constant whose value is FAK_SHARED_GOCACHE. Callers use it as the explicit WorktreeEnv map key for a requested shared Go build-cache path; it names the control rather than storing the selected directory.

**Distinct from:** The constant is the Go API spelling of the same control named FAK_SHARED_GOCACHE in an environment or caller map. It is distinct from sharedCache, the requested path value that WorktreeEnv validates, and does not introduce separate cache ownership.


### FAK_SHARED_GOCACHE (worker shared-cache opt-in environment key)

FAK_SHARED_GOCACHE is the environment-key spelling of the worker shared Go build-cache opt-in, also exposed to Go callers as SharedGoCacheEnv. WorktreeEnv reads an explicit caller-map value before the ambient environment; an explicit empty value disables ambient opt-in, and only a trimmed absolute non-NUL path selects the child GOCACHE.

**Distinct from:** This is the external key spelling of the same control named by SharedGoCacheEnv, not a second cache. It is distinct from sharedCache, the requested directory value, and from the selected GOCACHE destination. The external cache remains operator-owned and GOTMPDIR remains worktree-local.


### PrefillAttention

metalgemm's device prefill attention: a tiled simdgroup-MMA causal GQA kernel that computes softmax(scale*q.k^T)v for a P-row prefill panel against the layer's whole f32 KV cache on the GPU, one threadgroup per 8 query rows and GQA group (fak#13695).

**Distinct from:** attnPrefillInto is the host CPU loop and parity oracle for the same math; PrefillAttention is its GPU twin, and attnPrefillDispatch chooses between them.


### attnPrefillDispatch

The Session-level prefill attention route chooser: on a Metal prefill path it runs metalgemm.PrefillAttention and otherwise, or on any case the device kernel cannot reproduce exactly (observer, soft-cap, unsupported geometry, tiny panel, device failure), runs the host attnPrefillInto.

**Distinct from:** It computes nothing itself: attnPrefillInto (host) and PrefillAttention (device) are the two kernels it routes between.


### fak_engine_kernel_*

Prometheus family prefix for the sub-kernel seam: per-GEMM call counts, per-call durations, and the kernel presence and overflow bits, all rendered by internal/stepobs from computetrace events.

**Distinct from:** Covers the KERNEL sub-seam only, inside one GEMM. The fak_engine_* phase families are the planner cycle (admission, prefill, decode, sample) and fak_engine_planner_step_* is the per-leg step seam, so a kernel series is never a phase.


### fak_engine_planner_step_*

Prometheus family prefix for the sub-planner step seam: per-leg durations and leg counts over the closed MicroSpanKind vocabulary, plus the planner presence bit, all rendered by internal/stepobs.

**Distinct from:** Covers one LEG of a planner step. The fak_engine_* phase families time a whole request cycle and fak_engine_kernel_* times inside a GEMM, so this prefix sits between them and must not be summed with either.


### kernelRegistry

The bounded fold inside internal/stepobs that turns computetrace.Event values into per-(kernel, backend, timer_domain) call counters and histograms, capped at MaxKernelKeys with a single fixed overflow key.

**Distinct from:** The KERNEL half of the stepobs Recorder, which also owns a planner-step half; internal/computetrace.Recorder is a separate trace ARTIFACT store with a ring bound and activation samples, not a metric aggregate.


### MetricPlannerStepSeconds

Name of the fak_engine_planner_step_seconds histogram: the duration of one planner-step leg over the closed StepKind vocabulary.

**Distinct from:** The PLANNER-LEG histogram. fak_engine_phase_seconds times a whole serving cycle and fak_engine_kernel_seconds times one GEMM, so this is the only one of the three whose label is a step KIND rather than a phase or a kernel.


### MetricPlannerStepEvents

Name of the fak_engine_planner_step_events_total counter: how many planner-step legs of each closed kind have been observed.

**Distinct from:** The COUNT companion to the leg duration histogram. fak_engine_kernel_calls_total counts GEMM invocations, not planner legs, so a reader must not divide one series by the other.


### MetricPlannerStepKindOverflowTotal

Name of the fak_engine_planner_step_kind_overflow_total counter: planner-step legs dropped because their kind fell outside the closed StepKind vocabulary.

**Distinct from:** The KIND-bound overflow for the planner seam. fak_engine_kernel_keys_capped and fak_engine_kernel_event_overflow_total bound the KERNEL label set by count; this bounds the planner label set by vocabulary, because the planner side is closed and never grows.


### ObservePlannerStep

Recorder method that folds one planner-step leg of a closed StepKind and its duration into the planner-step registry, counting and dropping any out-of-vocabulary kind.

**Distinct from:** The PLANNER ingress. ObserveKernel is the compute-side ingress, and ObserveEnginePhase is the typed adapter that projects a native serving phase onto the closed kind vocabulary; this method takes the kind already resolved.


### plannerRegistry

The closed-vocabulary fold inside internal/stepobs that turns planner-step legs into per-kind histograms and counters, dropping any kind outside the closed StepKind set.

**Distinct from:** The PLANNER half of the stepobs Recorder, whose other half is the kernel registry; because its vocabulary is closed it pre-allocates every series, so a cold render emits all of them at a real zero instead of omitting them.


### cachedImmutableWeight (borrowed device weight staging)

Session.cachedImmutableWeight reuses a session handle or borrows a model-owned immutable device weight and stages it only on a miss.

**Distinct from:** It reuses immutable model weights whose borrowed handles survive Session.Close until Model.CloseWeights; KV cache stores mutable per-token attention state.


### SupportsQ3KMatMul (packed Q3_K pipeline availability)

The optional backend predicate reports whether the loaded Vulkan bundle has the packed Q3_K matrix-multiply pipeline needed before device selection.

**Distinct from:** It reports Q3_K pipeline availability; SupportsQ6KMatMul reports a different packed dtype, and neither predicate qualifies full-model execution.


### SupportsQ5KMatMul (packed Q5_K pipeline availability)

The optional backend predicate reports whether the loaded Vulkan bundle has the packed Q5_K matrix-multiply pipeline needed before device selection.

**Distinct from:** It reports Q5_K pipeline availability; SupportsQ3KMatMul and SupportsQ6KMatMul report different packed dtypes, and none qualifies full-model execution.


### V41AttentionRoleReader (V4.1 shared-source consumer role)

The V4.1 decoder role assigned to a layer that consumes the nearest preceding declared KV source's published compressed rows. Its resolved plan also identifies the index source whose published selection it consumes unless the layer owns that index selection. A ratio-greater-than-one reader still advances its own compressor and private history while contracting the resolved shared KV stream.

**Distinct from:** This value identifies ownership and reuse of published KV rows and index selections across decoder layers. Attention (softmax token-mixer) identifies the mathematical query-key softmax and value contraction operation; the reader role resolves its inputs rather than defining that operation.


### parallelQuantLoadContextBudget (ggufload hierarchical quant load)

The internal/ggufload WeightSource method that runs the production hierarchical, cancellable quantized tensor load: an outer tensor fan-out with a per-tensor inner worker budget, applying each computed tensor serially under the caller's context.

**Distinct from:** Names the ctx-aware parallel load scheduler, not the expert-parallel memory plan or the host-load-peak admission estimate that decides whether a load may start.


### QuantModelQ4KProfileOptionsContext (cancellable Q4_K model build)

The internal/ggufload WeightSource entry point that builds a Q4_K quantized model from a GGUF under a caller context, a load profile, and functional load options such as streamed experts; the context-aware form the plain QuantModelQ4K wrappers delegate to.

**Distinct from:** Names the cancellable, option-carrying Q4_K model constructor, not the parallel tensor scheduler it drives or the generic command context.


### v41AttentionQKRoPE

Owns BF16 query and key-value publication before and after the full V4.1 forward tail rotary operation, whose arithmetic remains F32.

**Distinct from:** Unlike attnPrefillDispatch, this owner stages rotary operands and results; it does not select or compute an attention kernel.


### v41AttentionOutputForProjection

Owns full V4.1 BF16 sparse-attention output staging before inverse F32 tail rotary arithmetic and BF16 publication to grouped output projection.

**Distinct from:** Unlike v41AttentionQKRoPE, this owner operates on attention output after sparse attention and applies inverse rotation before output projection.


### ExecutionDeadlineAdmissionSupported

Reports whether the normal request-owned prefix acquisition route can perform deadline admission using usable cached-prefix credit; speculative routes remain cold-admitted.

**Distinct from:** KVPrefixReuseSupported concerns whether a KV cache contains complete reusable session state; this predicate concerns whether this planner route has the owned-prefix execution boundary required to grant deadline credit.


### inKernelDeviceGate

Owns the planner-owned single native forward slot shared by ordinary requests and decode cohorts, supporting cancellation-aware waiters without detached lock-acquisition goroutines.

**Distinct from:** ExecutionDeadlineAdmissionSupported identifies deadline-credit capability at a prefix boundary; this gate controls exclusive concurrent access to the device forward slot and does not estimate deadline feasibility.


### deadlineCacheTicket

Immutable fingerprint-chain and estimated-token snapshot of the original request, carried to an accepted buffered completion for historical cache-usage observation before final socket delivery.

**Distinct from:** FeatureCacheProof binds an accepted vDSO hit to the actual call and served payload. This snapshot only describes historical request input; it neither witnesses current residency nor grants route-bound deadline credit.


### withDeadlineCacheObservation

Attaches a fresh original-request deadlineCacheTicket to a derived request context for later accepted buffered-completion observation, without querying historical credit or reserving admission.

**Distinct from:** Unlike deadlineCacheTicket, which stores the snapshot, this function defines the request-context attachment boundary; it does not alter deadline admission or prove delivery.


### DeadlineCacheCreditStats

Exported compatibility snapshot of historical credited-admission, credited-token, overrun-denial and misprediction counters; observational-only served routes do not increment credited admissions or tokens.

**Distinct from:** Unlike deadlineCacheTicket, which describes one request for historical recording, this aggregate metric schema is monitoring output and carries no current-residency proof or deadline-credit authority.


### VulkanReceiptBuildPlan (receipt verification policy plan)

VulkanReceiptBuildPlan fixes the independently trusted package, output package, and smoke policy used to verify a Vulkan binary receipt.

**Distinct from:** It is the verifier's expected build-policy input, not vulkan_plan command-buffer scheduling and not the generic BuildPlan action planner.


### forwardCoverage (per-token native traversal observation)

forwardCoverage is the internal request-local observation emitted only after a logits-producing native V4.1 forward commits, recording configured and executed layers, head dimension, completion, and full versus incremental logits source for one output token.

**Distinct from:** This is the in-process observation of one logits-producing traversal; the serialized NativeForwardCoverage receipt is its external projection, while generic coverage measures breadth across a declared universe.


### NativeForwardCoverage (serialized per-token traversal proof)

NativeForwardCoverage is the versioned receipt row attached one-for-one to generated output tokens, projecting the native model type, configured and executed layer counts, head dimension, full-depth/completeness flags, and full or incremental logits source.

**Distinct from:** This is the serialized receipt projection consumed by evidence validators; internal forwardCoverage is captured at decode time, while generic coverage does not prove one token traversed the configured native depth.


### v41CandidateSource

The identity of one V4.1 sparse-attention candidate-mask producer: producer layer, compressed-key source layer, compression ratio, and absolute query interval.

**Distinct from:** It identifies a compressed-key mask publication for one query interval; CandidateTree holds speculative continuation-token branches for target-model verification. This identity contains neither masks nor continuation tokens.


### v41CandidatePublication

An immutable session-local V4.1 sparse-attention publication retaining one owned compressed-key candidate mask and causal visible extent per absolute query, together with source identity and key width.

**Distinct from:** v41CandidateSource is the producer and query-range identity; this publication owns the per-query masks bound to that identity. Its masks restrict later query-owning indexers, while ordinary attention readers reuse final selected IDs. It is not yet stored in V41AttentionState or wired into serving.


### v41SelectCandidateSource

The V4.1 source-selection operation that consumes already computed per-query compressed-key scores and atomically returns a candidate-mask publication plus unmasked source top-k ID rows, or neither output on error.

**Distinct from:** v41CandidateSource supplies identity and geometry, and v41CandidatePublication retains immutable masks. This operation constructs that publication without projections or rescoring; later query-owning indexers apply its masks to their own scores, while source top-k remains unmasked.


### renderDeepSeekV41Chat (DeepSeek V4.1 GGUF-style prompt)

Renders the implemented DeepSeek V4.1 GGUF-style chat grammar: role boundaries, tool definitions and DSML tool-call history, merged tool results, and response-format instructions in the in-kernel prompt path.

**Distinct from:** Uses DeepSeek V4.1 message and DSML tool grammar rather than the ChatML role framing and Qwen thinking-prefix seeding of renderInKernelChatMLTools. This implementation record does not establish official-profile selection, checkpoint parity, or output-schema enforcement.


### FakExpertRingPoolFractionEnv (auto expert-ring host-memory fraction)

FakExpertRingPoolFractionEnv names FAK_EXPERT_RING_POOL_FRACTION. For auto activated-expert ring sizing with known host MemTotal, the fraction contributes a sizing bound before the activated-layer minimum is applied. The default is 0.10, accepted values satisfy 0 < f <= 0.5, and invalid values are refused. Explicit byte sizes are unaffected; the activated-layer minimum can exceed the fraction-derived bound.

**Distinct from:** This is a host-memory sizing parameter for the activated-expert ring, distinct from the worker-worktree disk-pool capacity setting and from the PagedKVPool block allocator. It does not create a separate allocator or guarantee a hard memory ceiling.


### ManifestPreflight

Header-only aggregate validation report from GGUF metadata and tensor descriptors, optional HF-config parity, loader key/tensor census and host-kernel availability; it reads no tensor payload.

**Distinct from:** Unlike serveexecutionpolicy, this diagnoses header configuration and coverage gaps; it neither authorizes execution nor proves runtime, device or model readiness.


### V41SharedAttentionPlain (V4.1 softmax mode)

V41SharedAttentionPlain is the V4.1 shared-attention native/shader ABI mode value 0. It selects a negative-infinity softmax maximum seed.

**Distinct from:** V41SharedAttentionCompressed selects negative MaxFloat32 and skips when that sentinel survives. Plain can consume selected compressed-history rows; it does not mean uncompressed storage. Unlike softmaxAttentionScores, this is a numerical mode selector, not the score-normalization function. Unlike v41SharedAttentionRows, the mode does not describe borrowed row storage.


### v41SharedAttentionRows (borrowed combined KV descriptor)

v41SharedAttentionRows is an internal model request descriptor containing borrowed window and compressed row slices, logically concatenated in that order, valid only during the synchronous attention callback. It does not own or cache the float data.

**Distinct from:** Unlike V41SharedAttentionPlain, a numerical softmax mode selector, this type describes input row storage. Admission validates every original row before selection; resident staging materializes only missing upload runs and no-row-log fallback packs selected rows.


### fusedVulkanSlotBytesOKLocked

Vulkan fused wrapper helper checking all selected weight-slot pointers have sufficient nominal resident buffer bytes while their live owners and command-stream lock are retained.

**Distinct from:** These slots are expert weight bindings, unlike the scheduler Slot free/busy signal. The helper checks nominal bytes only and does not validate stale aliases, device generations, or dispatch-grid limits.


### rebuildCacheGeometryOwned

Private Session cache-pool rebuild body invoked after its public boundary acquires the optional backend recording-domain owner; replaces expert, KV, recurrent and window pools while preserving maintenance locking and rollback.

**Distinct from:** The already-owned operation body performs the rebuild; cacheGeometryMu is the separate session maintenance mutex, not device-domain ownership. This body never reacquires the outer backend owner.


### evictKVOwned

Private Session store-aware KV-span eviction body used inside an already-owned backend operation; coordinates host or HAL eviction and token-lineage indices without reacquiring the recording-domain mutex.

**Distinct from:** This performs an index span eviction under an existing operation scope; EvictKV speculative rollback is the rejected-position count supplied to eviction, not the operation or ownership boundary.


### v41BatchIndexerGuard

Cohort fault-boundary adapter that invokes the serial selected-indexer panic guard for only the recorded failing Session and recovers the resulting primary value after cohort rollback.

**Distinct from:** This preserves selected-indexer failure and teardown semantics for one faulting lane without retiring surviving peers. evictKVOwned is an ordinary store-aware KV span mutation inside an owned operation, not a selected-failure panic or Session retirement boundary.


### CachePolicyStream (offline expert access replay input)

A flat deterministic sequence of unit-sized expert groups consumed by offline cache-policy replay.

**Distinct from:** Offline replay input data; it neither selects serving eviction policy nor owns a runtime cache or route-trace emitter.


### v41ColdAttentionResidentOwner

Pure cold-layer admission that returns the seeded attention state only when existing mirror windows fit and repeated-window upload savings cover the compressed bank; it performs no backend operation.

**Distinct from:** This decides upload-volume eligibility for a whole cold layer; v41AttentionOutputForProjection produces the attention result for projection. Eligibility is not a hardware speed guarantee and does not replace callback capability checks.


### v41ColdAttentionResidentView

Constructs a borrowed resident KV view using a successfully seeded layer attention state, logical position, window row count and compressed row count; original row streams remain authoritative.

**Distinct from:** This describes the already admitted state for one query without owning a temporary identity or validating row data; v41ColdAttentionResidentOwner performs the separate whole-layer upload-volume admission.


### inKernelDecodeV41Lanes

Concrete adapter to result-bearing Model.StepV41Batch with untouched unselected scalar fallback and owner-busy wait.

**Distinct from:** Executes prepared-lane decoding rather than reporting coordinator cleanup; does not activate serving.


### inKernelV41DecodeReceipt

Private driver aggregate of completed shared-head panels and rows returned by model operations.

**Distinct from:** Completed-operation counts rather than the concrete decode adapter; no physical throughput claim.


### inKernelV41DecodeOps

Private callbacks for batch attempt, acquiring scalar fallback and cancellation-aware busy wait.

**Distinct from:** Operation bundle rather than completed-operation receipt or capability declaration.


### inKernelDecodeV41LanesWithOps

Stable-ID decode state machine retaining sampled token, emission and penalty/RNG state across busy retries and scattering outcomes without selected replay.

**Distinct from:** Execution algorithm rather than callback bundle.


### inKernelV41Pending

One stable lane slot retains Session identity, chosen token and pending-forward state across retries and compaction.

**Distinct from:** Uncommitted request progress rather than completed-operation aggregate.


### inKernelV41LaneContext

Selects a decode lane request context, falling back to driver context.

**Distinct from:** Cancellation authority selection rather than retained token progress; creates no context and does not cancel healthy peers.


### inKernelV41ValidateResult

Post-operation check of unique exact lane-ID coverage and untouched unselected results before fallback.

**Distinct from:** No-replay protocol validation rather than executing concrete model wiring.


### inKernelV41ApplyRow

Applies one validated stable-ID outcome, updating history only on commit and preserving committed errors and actual forward kind.

**Distinct from:** Commit-aware mutation rather than pre-scatter validation.


### inKernelCoalesceKind

Decode-algorithm discriminator and Qwen/V41 values separating unlike queued requests into contiguous FIFO cohorts.

**Distinct from:** Queue grouping and Flash lane bounds rather than pending token progress or backend capability.


### inKernelV41RequestState

Associates a request with its prepared decode lane and unknown panic published by closure completion.

**Distinct from:** Prepared-request lifecycle rather than pending sampled-token state; creates no Session lock.


### errInKernelV41CohortAborted

Terminal no-replay outcome for unfinished or unstarted requests in an aborted selected Flash cohort.

**Distinct from:** Selected-cohort termination rather than retained request state or untouched fallback eligibility.


### v41ColdPlainAttentionInput

Chooses a bounded borrowed contiguous KV-row window for valid admitted cold plain attention, otherwise preserves original flat packing.

**Distinct from:** Host materialization choice rather than whole-layer resident-owner upload admission.


### v41ColdPlainAttentionCanBorrow

Pure shape, finite-value and index predicate for safe contiguous cold plain KV-row borrowing.

**Distinct from:** Eligibility predicate rather than packing or dispatch; questionable inputs retain original flat validation precedence.


### v41ColdPlainAttentionSink

Synchronous cold plain attention dispatch through borrowed rows with original flat reconstruction when dispatch guards change.

**Distinct from:** Timed dispatch and guard recheck rather than pre-dispatch borrowing predicate.


### Qwen coalesced decode kind

Existing Qwen algorithm selector for contiguous queue grouping.

**Distinct from:** Selects Qwen execution rather than the Flash four-lane grouping variant.


### Flash coalesced decode kind

Flash algorithm selector for bounded four-lane contiguous FIFO cohorts.

**Distinct from:** Selects the Flash member of the decode-kind discriminator, distinct from Qwen; neither grouping value declares backend capability.


### Flash lane-local cancellation context

Selects each decode lane request cancellation/deadline context with driver-context fallback.

**Distinct from:** Context-family view of the same helper: cancellation authority rather than gateway decode flow or retained token progress; creates no new context.


### AdmitCachedBudget

The deadline admission estimator entrypoint that preserves caller-declared budget separately from remaining budget and returns bounded numeric input evidence with the same feasibility verdict and retry policy as AdmitCached.

**Distinct from:** AdmitCached accepts only the remaining budget; AdmitCachedBudget also projects the declared budget and mutex snapshot. The supplied cache count is caller-supplied resident-prefix credit; callers validate route proof separately from prefix identity or cache allocation.


### V41Q2KTokenPanelPlan

Pure bounded Q2_K token-panel geometry carrying output/input widths, exactly 2/4/8 rows, validated byte extents and the workgroup count for the three-int32 shader push ABI.

**Distinct from:** An immutable launch geometry result, not a recorded Vulkan command buffer or a device qualification claim.


### PlanV41Q2KTokenPanel

Pure planner that validates Q2_K matrix geometry, block divisibility, signed 32-bit byte bounds and exactly 2/4/8 token rows before returning V41Q2KTokenPanelPlan.

**Distinct from:** Performs host arithmetic admission only; the returned plan holds geometry while provider capability and opt-in decide whether native execution is available.


### V41Q3KTokenPanelPlan

Pure bounded Q3_K token-panel geometry carrying output/input widths, exactly 2/4/8 rows, 110-byte quantized block extents padded to four bytes, and workgroup count for the three-int32 shader push ABI.

**Distinct from:** An immutable launch geometry result for Q3_K storage rather than Q2_K geometry or a recorded Vulkan command buffer; does not qualify native device execution.


### PlanV41Q3KTokenPanel

Pure planner validating Q3_K matrix geometry, block divisibility, padded 110-byte block storage, signed 32-bit byte bounds and exactly 2/4/8 token rows before returning V41Q3KTokenPanelPlan.

**Distinct from:** Host arithmetic admission for Q3_K storage; the result contains geometry while provider capability and opt-in independently control execution. Distinct from the Q2_K planner.


### PrepareForCache

PrefixSnapshot cache-admission method that seals independently owned V4.1 publication rows into immutable shared prefix storage before publishing an admitted snapshot, retaining private mutable continuation fields.

**Distinct from:** A snapshot publication preparation step invoked by the serialized cache owner after admission succeeds; not the RadixKV index, cache identity, lookup operation or a native performance qualification.


### v41IndexerKeyCache

Session-owned resident F32 indexer-key matrices keyed by immutable publication ownership, source and geometry identity, with live handles charged across callback rebinding. Admission requires an explicit positive index-key byte allowance that defaults to zero; unchanged history reuses its exact buffer, append stages only new rows while copying the old device prefix, and invalidated identity reseeds.

**Distinct from:** Separate from attention KV state and model-owned immutable weight staging. It owns indexer score inputs, charges retained plus replacement/staging peak, and retires replaced storage only after checked score read; append still incurs O(history) device copies. No production default or hardware-performance qualification.


### V4.1 EP panel transport layout

Checked per-owner request word counts, aggregate payload words, reply words, and explicit live-byte allowance for one V4.1 expert-parallel prefill panel exchange.

**Distinct from:** Transient wire-exchange geometry derived from routed rows and existing TPPlan owner bands, not tensor element ordering or context-region layout. Named live transport allocations do not cover RSS, device scratch, borrowed normalized input, or all-history Prefill outputs.


### V4.1 EP panel geometry admission

Validates the existing owner partition, routed picks, token/width bounds, every owner's encoded request size, response size, and explicit live transport allowance before creating v41EPPanelLayout.

**Distinct from:** Bounds one exchange before allocation or I/O; does not select TPPlan partitions or estimate complete host/device model memory, and does not certify hardware execution or full-history memory use.


### V4.1 EP owner build invocation plan

Pure value descriptor fixing the supported owner-build profile, helper-digest linker symbol/value, build argv, working directory, and complete replacement environment derived from validated configuration values.

**Distinct from:** A proposed invocation, not VulkanReceiptBuildPlan verification expectations or BuildPlan action diffs. It performs no filesystem observation or execution and proves neither fresh output, approved source contents, nor actual linkage of the requested helper pin.


### V4.1 EP owner build-session declaration

Supplied invocation, receipt digest, finished/joined flags, exit code, and output identity in an optional owner-build supplement declaration; the pure checker requires exact supplied counterparts and a finished, joined, zero-exit declaration before reporting value agreement.

**Distinct from:** A build-invocation declaration, not a live served Session or scheduler model-session origin. Parsing/equality do not observe a process, authenticate an issuer, prove historical build causality or binary linkage, or grant native admission.


### V4.1 EP session construction ownership result

Reports one authenticated EP loader invocation's returned Session/coordinator, error, separately flagged panic payload, ownership transfer, and construction-owned retirement observation.

**Distinct from:** A construction outcome, not a runtime Session or scheduler fresh/restored origin. Successful transfer leaves factory retirement unavailable/not applicable and ownership with the caller. Unsuccessful opaque constructors leave hidden cleanup unknown unless an observed outer-release failure proves incomplete retirement. The additive result API returns this evidence; the compatibility wrapper rethrows the original primary panic. Group/backend remain borrowed; no field proves device execution.


### v41SharedAttentionBinding

Private immutable identity container pairing one ordinary shared-attention callback with optional canonical Session, Model, and Backend provenance. Reconstructing or wrapping the callback creates an ordinary-only binding, so function identity does not transfer canonical provenance.

**Distinct from:** Unlike v41SharedAttentionRows, which describes borrowed KV input slices, this binding records callback origin and identity and selects no numerical mode. Forward state and scratch preserve the binding identity; any ordinary callback replacement creates a fresh ordinary-only binding. The binding alone proves neither current ownership nor row validation: only a revalidated v41SharedAttentionContinuation together with a one-use v41SharedAttentionValidation permits the private core to omit the duplicate row-finiteness scan.


### v41SharedAttentionContinuation

Private call-local eligibility record for one canonical shared-attention operation during a serial single-token continuation. It captures binding, Session, Model, Backend, forward-state, scratch, execution-owner, KV-owner, position, cache epoch, geometry and role-plan identities, and rechecks that they remain live before consumption.

**Distinct from:** Unlike v41SharedAttentionBinding, this record establishes current execution-context eligibility rather than callback origin alone. Unlike v41SharedAttentionRows, it owns no borrowed row storage and proves no row finiteness. Unlike V41SharedAttentionPlain, it selects no numerical mode. Its one-use validation witness is created separately only after the composer checks all original rows.


### v41SharedAttentionValidation

Private one-use, exact-request witness created after the combined-attention composer validates every original window and compressed row and selection ID before masking. It is tied to the live continuation, exact request and row-descriptor identities, is consumed once by that canonical private core, and permits omission only of the duplicate row-finiteness scan.

**Distinct from:** Unlike v41SharedAttentionContinuation, this witness records that the original inputs were validated for this exact invocation, not merely that the execution context is eligible. Unlike v41SharedAttentionBinding, it is call-local and single-use, not reusable provenance. Unlike v41SharedAttentionRows, it owns no KV data; unlike V41SharedAttentionPlain, it selects no numerical mode. It never travels through an ordinary callback request, and it does not replace geometry, shape, ID, Q, sink, selected-error or lifetime checks.


### v41RoPETableCache

Session-owned bounded exact F32 rotary sin/cos device tables. One query slot or two compressed-publication slots reuse only matching values, model, epoch, backend, geometry and live session handles; replacement retains one transient upload.

**Distinct from:** Caches uploaded rotary coefficients, not inverse-frequency host memoization, attention KV state or resident indexer-key history. Session closure detaches all rotary owners before fallible teardown and schedules each backend release attempt; physical retirement remains backend-owned.


### newV41RoPETableCache

Constructs the bounded rotary-table owner metadata for one or two fixed session keys, copying the key list and capturing the backend. It does not upload device tables; lookup and installation occur through the returned owner.

**Distinct from:** Constructor operation rather than the retained rotary-table ownership state. v41RoPETableCache holds exact host values and live session tensor identity and participates in table replacement and session retirement.


### v41SharedAttentionResult

Private canonical shared-attention host output ownership carrier. A successful private core invocation returns a fresh zero row or a validated readback clone allocated before tensor retirement; failure returns no output ownership.

**Distinct from:** Unlike v41SharedAttentionBinding this value owns one completed host output rather than callback provenance. Unlike v41SharedAttentionContinuation and v41SharedAttentionValidation it grants no input validation or execution eligibility. Unlike v41SharedAttentionRows it is fresh output, not borrowed KV descriptors. Ordinary callbacks and borrowed resident outputs cannot acquire this carrier; final composer width and finite checks remain mandatory.


### TryV41PoolNormPanel

Bounded canonical Vulkan FFN normalization operation that atomically borrows already-free exact-size native buffers and existing staging capacity, sharing scalar-required gain preparation and refusing before side effects when capacity is unavailable.

**Distinct from:** An exclusive temporary allocator-buffer execution loan with selected-failure no-replay semantics; distinct from Pool (session), which owns bounded LRU session state. This operation creates neither a new persistent pool nor a device-memory budget grant.


### RequestedContextTokens

RequestedContextTokens preserves the caller requested native context token count in a header-derived memory projection, including unknown and refused results.

**Distinct from:** This is the caller input before memory resolution; native-context-resolution derives the fitting runtime token count.


### ResolvedContextTokens

ResolvedContextTokens is the native context token count derived by the runtime context resolver for the activated-ring memory projection snapshot; zero with UNKNOWN means unresolved.

**Distinct from:** This is the resolved model window after sizing, while RequestedContextTokens preserves the caller input even when resolution cannot run.


### serveGGUFMemoryPlanForArm

serveGGUFMemoryPlanForArm derives classed weight, scratch and context memory demands from a GGUF header for the selected native serve load arm.

**Distinct from:** This function produces byte demands for a chosen arm; native-context-resolution derives a token window from sizing inputs and the activated-ring fallback may replace the resulting full-residency plan.


### serveFullPlanFitTooBig

serveFullPlanFitTooBig recognizes a typed compute FitError whose verdict is FitTooBig, enabling the activated-expert fallback only after a full-residency capacity refusal.

**Distinct from:** This predicate recognizes an existing typed refusal and produces no memory plan; serveGGUFMemoryPlanForArm produces the demands that a fit guard judges.


### VulkanMemoryGuardSnapshot

Bounded current-state projection of the Vulkan explicit-memory reservation cap: configured limit, live and successful-allocation peak bytes, refusal count, and configuration/accounting validity across every memory heap.

**Distinct from:** Observes the allocator reservation ledger; it is neither an admission decision nor readiness, physical residency, driver-overhead accounting, or proof of target cgroup fit. C ABI, Go value and debug label project this same state.


### V41Q2KTokenPairPlan

Immutable paired Q2_K launch geometry embedding validated single-panel geometry and a doubled X workgroup range for two independent output banks using the same three-int32 push ABI.

**Distinct from:** Describes paired launch arithmetic only, unlike the single-output V41Q2KTokenPanelPlan or the constructor that validates and creates this result. It does not record commands or qualify device execution.


### PlanV41Q2KTokenPair

Pure constructor reusing Q2_K token-panel shape, row-count and signed byte-bound validation before returning paired geometry with twice the single-panel X workgroup count.

**Distinct from:** Constructs host geometry rather than executing a contraction or checking actual device limits; the paired capability query separately determines native availability.


### SupportsV41Q2KTokenPair

Effect-free optional backend capability query for two same-shape Q2_K contractions sharing one input token panel and producing two independent output panels with unchanged per-workgroup arithmetic.

**Distinct from:** Checks selected provider availability and bounded geometry before effects; unlike pure plan construction, it reflects native opt-in, exact module availability and device dispatch limits. It grants no residency, memory budget or speed guarantee.


### DeviceTransientPoolStats

Optional backend observation of request-transient buffers currently parked for reuse: backend name, current parked buffer count and bytes, and parked-byte high-water mark since backend start. Unsupported planners report absence; the counters do not grant memory capacity.

**Distinct from:** Observes reusable backend allocation storage after live tensor ownership ends; unlike pool-session it does not hold session state, and unlike an allocation reservation it neither admits nor owns future memory.


### cpuRefScratchPool

The CPU reference backend's sync.Pool of fixed 256-element float32 dequantization panels, borrowed by quantized MatMul/BatchedMatMul row-dot work and returned after use to avoid a fresh panel allocation per call.

**Distinct from:** Reuses small host arithmetic scratch, unlike pool-session's bounded LRU session container. It is not device storage, model weight residency, or a hard memory reservation; sync.Pool may discard cached panels.


### fak_gateway_go_memory_bytes

Gateway process-memory gauges labeled by kind: Go heap objects/unused/free/released, heap/OS stacks, runtime-mapped bytes, and whole-process RSS. Values come from the runtime observation receipt; unavailable axes are omitted.

**Distinct from:** Measures memory quantities, unlike the gateway operation-latency histogram. Its kinds are overlapping accounting views, not additive independent budgets; RSS minus runtime-mapped bytes is only an approximation of outside-runtime mappings.


### fak_gateway_go_gc_heap_bytes

Gateway gauges for the Go GC heap: kind=live is the heap marked live by the last completed collection, and kind=goal is the heap size target for the next collection. Unavailable kinds are omitted.

**Distinct from:** Reports GC live/goal byte quantities rather than operation latency, total process RSS, or the configured soft memory limit. A goal is not an enforced allocation grant.


### fak_gateway_go_memory_limit_bytes

Gateway gauge exposing the Go runtime's configured GOMEMLIMIT soft memory limit in bytes, omitted when the runtime reports it as unlimited or unavailable.

**Distinct from:** A configured soft runtime target, unlike gateway operation latency or observed resident memory. It is neither an OS hard limit nor a whole-process/device memory guarantee.


### fak_gateway_go_gc_cycles_total

Gateway counter of completed Go garbage-collection cycles since process start, taken from the runtime observation receipt and omitted when unavailable.

**Distinct from:** Counts collection events; unlike gateway operation latency and GC pause duration it carries no timing or memory-size claim.


### fak_gateway_go_gc_pause_p99_seconds

Gateway gauge of the process-lifetime 99th-percentile stop-the-world GC pause, estimated from the runtime pause histogram using the selected bucket's upper bound; omitted when the quantile cannot be read.

**Distinct from:** A lifetime GC-pause percentile estimate, unlike the gateway per-operation latency histogram. It is not a recent-window request-latency percentile or an exact individual pause duration.


### fak_gateway_go_goroutines

Gateway gauge of live Go goroutines in the gateway process from the runtime scheduler observation; omitted when unavailable.

**Distinct from:** Counts runtime goroutines, unlike gateway operation latency or request/session counts. It does not imply parallel CPU execution or available worker capacity.


### fak_gateway_device_transient_pool_bytes

Gateway gauges labeled by backend and kind for request-transient device buffers parked for reuse: kind=pooled is current parked bytes, kind=peak is the high-water mark since backend start. Backends without the optional pool reporter emit no series.

**Distinct from:** Measures reusable device allocation bytes rather than gateway operation latency, live tensor bytes, or host RSS. The peak is historical, not an additional allocation, and the observation is not a reservation grant.


### fak_gateway_device_transient_pool_buffers

Gateway gauge labeled by backend for the current count of request-transient device buffers parked for reuse. It is emitted only when the optional backend pool reporter supplies an observation.

**Distinct from:** Counts parked buffers rather than gateway operation latency or bytes; unlike fak_gateway_device_transient_pool_bytes it does not describe buffer size and cannot be used as a memory budget.


### v41AttentionInputPanel

Private call-scoped loan of freshly normalized cold-prefill activation chunks to the Q and KV projections. It captures canonical Session, Model, Backend, binding, execution-scope and cache-epoch identities; the attentionInputPanel scratch field carries the same loan only until synchronous QKV completion.

**Distinct from:** Unlike v41SharedAttentionRows, this holds normalized projection inputs rather than borrowed attention KV rows. Unlike v41SharedAttentionContinuation, it governs a multi-row cold-prefill input lifetime rather than single-token attention eligibility. It neither owns persistent KV state nor grants general zero-copy or device authority.


### v41PlainAttentionContinuation

Private one-lane suspended plain-attention operation between contraction/rotation and grouped output projection. Its shared per-layer lifecycle permits one resume or abandon cleanup, and successful resume transfers resident cleanup to the returned layer work.

**Distinct from:** Unlike v41SharedAttentionContinuation, this owns suspended computation and resource transfer rather than proving live execution eligibility. Unlike v41AttentionInputPanel, it carries one lane's output-projection input rather than a multi-row cold-prefill normalized-input loan. It owns neither the transaction journal nor persistent KV state and grants no batching or device capability.


### v41SparseAttentionSinkChecked

Synchronous sparse sink contraction over an already validated flat or split-row host view, preserving arithmetic order without flattening the cold KV bank.

**Distinct from:** Unlike v41ColdPlainAttentionSink, which chooses and dispatches a contiguous plain-attention window, this contracts selected sparse rows only after its caller validates every original row and canonical index.


### suffixPrefillSession

Modelbench interface for owned prefix prefill, suffix continuation, next-token step, between-call sequence position and terminal close in checkpoint replay comparisons.

**Distinct from:** This is the narrow benchmark adapter contract, not the model Session implementation, transport conversation, or concurrent state snapshot; implementations retain their own checkpoint backing lifetime.


### kernel_timestamp_delta

Optional serialized modelbench receipt field containing the backend KernelProfile counter and timestamp delta across host-delimited prefix, suffix or next-step phases.

**Distinct from:** This profile delta is process-global and may be partial or contaminated by concurrent work. It is not GPU utilization, device wall time or isolated kernel-active time; Vulkan timestamp profiling serializes diagnostic batches and includes barriers and copies.


### ArtifactCoverage

The modelbench suffix receipt field and JSON alias describing the declared extent of checkpoint artifact identity verification, currently input-file-only.

**Distinct from:** This states an identity-check boundary, not test or model coverage. It does not establish a complete sharded checkpoint set, correctness, GPU execution or production readiness; identity hashing can warm the page cache outside timing.


### artifact_identity_coverage

Serialized JSON field in the suffix-prefill receipt that exposes the ArtifactCoverage value for checkpoint identity verification; currently input-file-only.

**Distinct from:** This is the wire key for the declared identity boundary, unlike the Go ArtifactCoverage field or general test coverage. It does not verify a complete shard set, GPU execution or model correctness.


### loadBenchOwnedQ4KContext

Modelbench Q4K loader adapter that accepts cancellation context and retains the loader-owned checkpoint backing for the resulting model session until terminal cleanup.

**Distinct from:** Unlike the hierarchical quant loader cancellation budget, this benchmark adapter governs retained backing ownership and cleanup across successful execution, load errors and panic; it does not add a new context scheduler or change device timing.
