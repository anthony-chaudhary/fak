---
title: "fak concept glossary — Concept families — scorecards, eviction, decisions, plans, loops, and operator surfaces"
description: "The scorecard/debt, eviction, decision, render/materialize, plan, pool, layout, loop, trajectory-control, dev-tier/operator, and cross-cluster concept families of the fak concept glossary."
---

# Concept families — scorecards, eviction, decisions, plans, loops, and operator surfaces

Split out of [the concept glossary](concept-glossary.md); the routing table and the
cache and guard/gate families remain on that page.

## Reader orientation

**For:** readers comparing fak's measurement, planning, loop, and operator-surface vocabulary. **TL;DR:** start here when two operational terms sound interchangeable, then use each entry's "distinct from" boundary to choose the right one.

List the operational families covered by this page:

```bash
git grep -n '^## ' -- docs/fak/glossary-families-b.md
```

Use the matching line number as the next reading stop, then follow its local links when the entry names a deeper contract.

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

