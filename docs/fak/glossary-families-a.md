---
title: "fak concept glossary — Concept families — witness, session, gateway, policy, and context"
description: "The witness/evidence, session/scheduling, gateway/engine, policy/authorization, and context-management concept families of the fak concept glossary."
---

# Concept families — witness, session, gateway, policy, and context

Split out of [the concept glossary](concept-glossary.md); the routing table and the
cache and guard/gate families remain on that page.

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

