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


