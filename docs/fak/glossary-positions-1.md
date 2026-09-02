---
title: "fak concept glossary — Positioned concept entries (1 of 3)"
description: "Machine-positioned glossary entries (first third), split out of docs/fak/concept-glossary.md with anchors and text preserved verbatim."
---

# Positioned concept entries (1 of 3)

Machine-positioned entries, split out of [the concept glossary](concept-glossary.md)
(which stays the landing surface for new `fak concept position` appends).

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


