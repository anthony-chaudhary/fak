---
title: "fak concept glossary — Positioned concept entries (2 of 3)"
description: "Machine-positioned glossary entries (second third), split out of docs/fak/concept-glossary.md with anchors and text preserved verbatim."
---

# Positioned concept entries (2 of 3)

Machine-positioned entries, split out of [the concept glossary](concept-glossary.md).

## Reader orientation

**For:** readers resolving a term found in a receipt, refusal, or implementation comment. **TL;DR:** this is the second alphabetically positioned shard; scan stable headings here rather than reading it front to back.

List its entries from a repository checkout:

```bash
git grep -n '^### ' -- docs/fak/glossary-positions-2.md
```

Match the exact term from your artifact to a heading, then read its distinction and grounding notes before treating neighboring vocabulary as equivalent.

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


