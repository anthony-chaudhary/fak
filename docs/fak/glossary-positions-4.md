---
title: "fak concept glossary — Positioned concept entries (4 of 4)"
description: "Machine-positioned glossary entries (final part), split out of docs/fak/concept-glossary.md with anchors and text preserved verbatim."
---

# Positioned concept entries (4 of 4)

Machine-positioned entries, split out of [the concept glossary](concept-glossary.md).

## Reader orientation

**For:** readers tracing the newest positioned concepts, including TUI and operator-facing adapters. **TL;DR:** use the heading index to locate the named concept, then verify its stated package or command boundary in the repository.

List this shard's stable headings:

```bash
git grep -n '^### ' -- docs/fak/glossary-positions-4.md
```

The matching heading is the checkpoint: its definition should explain what the concept owns and what adjacent layer it deliberately does not own.

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


### Decision (incident trigger receipt)

Closed COLLECT, ADMIT, or SUPPRESS outcome recorded by the versioned incident trigger receipt.

**Distinct from:** Incident debounce receipt outcome, not the incident handler disposition or kernel, scheduler, witness, or shared-task Decision types.
