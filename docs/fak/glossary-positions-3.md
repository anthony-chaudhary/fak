---
title: "fak concept glossary — Positioned concept entries (3 of 3)"
description: "Machine-positioned glossary entries (final third), split out of docs/fak/concept-glossary.md with anchors and text preserved verbatim."
---

# Positioned concept entries (3 of 3)

Machine-positioned entries, split out of [the concept glossary](concept-glossary.md).

## Reader orientation

**For:** readers looking up later positioned concepts while debugging provider, session, or control behavior. **TL;DR:** treat this page as an indexed reference shard, not a tutorial sequence.

List the available concept headings:

```bash
git grep -n '^### ' -- docs/fak/glossary-positions-3.md
```

Select the exact heading named by the code or receipt, then use the entry's "distinct from" text to rule out the nearest false match.

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


