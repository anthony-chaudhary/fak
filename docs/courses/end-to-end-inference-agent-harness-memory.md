---
title: "End-to-end inference, agent harness, and memory course"
description: "A bounded course following one fak request through native inference, agent tools, policy, context admission, memory recall, and verification."
---

# End-to-End Inference, Agent Harness, and Memory

> **Flagship bounded course.** Follow one fak request end to end. Trace runtime admission, fak-native model execution, and agent tool use. Inspect policy checks and local fast paths. Then examine context admission, memory recall, and durable verification. This page is a syllabus and lab guide; the linked canonical documents remain authoritative.

## Course contract

### Audience

This course is for engineers who can read Go and operate a command line and who want to build, integrate, review, or run a modern fak deployment. It is suitable for agent-harness authors, inference engineers, security and policy reviewers, and operators responsible for evidence-backed releases.

It is not an introduction to transformers, Go syntax, Git, or shell basics. It also does not replace the subsystem references, benchmark authority, or private lab runbooks.

### Prerequisites and background

Before starting, you should be able to:

- Build and test a Go module.
- Read JSON receipts and distinguish a claim from an observed artifact.
- Explain prompts, tokens, and model weights at a basic level. Also understand prefill, decode, and a KV cache.
- Explain why untrusted tool output must not automatically become trusted model context.
- Use explicit paths in a peer-dirty checkout.

Start with [Getting Started](../../GETTING-STARTED.md), the [reproduction packet](../repro-packet.md), and the [architecture overview](../../ARCHITECTURE.md). Keep the [CLI reference](../cli-reference.md) open throughout the course.

### Environment markers

Every lab carries one of these markers:

- **LOCAL / OFFLINE** — no API key, model, or GPU is required.
- **LOCAL MODEL (ENVELOPE-DEPENDENT)** — a supported local fak-native backend and compatible model artifact are required. Inspect admission before loading a payload.
- **FLEET GPU / PRIVATE LAB** — accelerator evidence must run on a sanctioned compute node. The workstation serves as the control point rather than the compute boundary. Follow the public [fleet compute node map](../fleet-compute-nodes.md) and enter private infrastructure only through the [private communications channel](../private-comms-channel.md). Never copy credentials, private hostnames, private paths, or raw internal logs into public course work.

Absence of local GPU hardware is not a terminal result. Dispatch the GPU lab or record the exact sanctioned handoff; do not substitute llama.cpp and call the result fak-native.

### Measurable outcomes

By the end, you will be able to:

1. Distinguish executable, control-plane, and model-execution readiness from a `fak-runtime-capabilities/1` or execution-mode receipt.
2. Prove that native/performance evidence names `engine: "fak-native"`, a current Qwen3.8 model, a backend, and an explicit operating envelope.
3. Trace a typed tool call through the harness and syscall layer. Follow adjudication, dispatch or vDSO, result admission, and model context.
4. Explain policy precedence, structural denial, and why policy success does not prove model execution.
5. Distinguish provider prompt cache, radix-prefix reuse, and model KV cache. Compare those with vCache and tool-result vDSO reuse.
6. Describe the context-MMU write-time lifecycle: admit, quarantine/page out, restore under a witness, and evict stale or poisoned state.
7. Construct and explain a bounded `memq` recall/render/compact plan. Include trust gates and proposal-only mutation defaults.
8. Produce a compact evidence packet. Separate delivery, quality, and performance claims from security and operating-envelope claims.
9. Run the scope-correct local verification workflow without mixing peer WIP into the result.

## Conceptual system map

```text
operator / application
        |
        v
agent harness + model loop
        |  typed ToolCall
        v
fak Syscall / Submit-Reap seam
        |
        +--> preflight grammar + policy/adjudicator fold --> DENY / DEFER / ALLOW
        |
        +--> local vDSO/tool-result fast path -----------> local result, when valid
        |
        +--> registered tool/engine dispatch -----------> external or in-process effect
                                                               |
                                                               v
context-MMU result admission <--- provenance / witness / taint / revocation
        |
        +--> admitted result enters bounded model context
        +--> quarantined bytes page out behind a small pointer
        |
        v
fak-native Qwen3.8 execution within declared quality + resource envelope
        |
        +--> shared prefix / provider cache / radix reuse / model KV cache
        +--> context lifecycle: retain, shed, compact, reset, restore
        +--> memq memory: scan -> filter -> rank -> budget -> render
                              mutation steps propose by default
        |
        v
journals + receipts + traces + captured witnesses -> independent verification
```

Three distinctions govern the whole course:

- **Control is not execution.** A policy verdict can be real while no model weights were loaded.
- **Context is not memory.** Context is the model-visible working set; durable memory is stored, selected, provenance-bearing state. Read [Context Is Not Memory](../CONTEXT-IS-NOT-MEMORY.md).
- **A cache name is not a cache identity.** Use the taxonomy in [Cache concept disambiguation](../concepts/disambiguation-cache.md) before reasoning about hits, savings, or invalidation.

## Module 1 — Establish the local control-plane proof

### Read

- [Getting Started](../../GETTING-STARTED.md)
- [The 60-second reproduction packet](../repro-packet.md)
- [`runtime-capabilities` in the CLI reference](../cli-reference.md#runtime-capabilities-inspect-the-deployable-runtime-before-payload-load)

The first objective is deliberately model-free. Establish that the binary runs and that policy can structurally deny one tool while allowing another. Confirm that the offline agent completes useful work. Then inspect runtime capability without confusing that inspection with payload execution.

### Lab — LOCAL / OFFLINE

From the repository root, use a temporary output path rather than writing an in-tree binary:

```powershell
go build -o "$env:TEMP/fak-course.exe" ./cmd/fak
& "$env:TEMP/fak-course.exe" preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
& "$env:TEMP/fak-course.exe" preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
& "$env:TEMP/fak-course.exe" agent --offline
& "$env:TEMP/fak-course.exe" runtime-capabilities
```

Capture the commands, exit status, and machine-readable output. Label each observation as binary, control plane, or model execution evidence.

### Checkpoint

Submit a table with one row per command and answer:

- Which command proves a structural denial?
- Which command proves fak is not a blanket blocker?
- Which output, if any, proves that model weights executed?
- Why is a runtime-capability projection not itself a live inference receipt?

Proceed only if you do not claim model execution from the offline proof.

## Module 2 — Admit fak-native Qwen3.8 execution and its quality envelope

### Read

- [Native inference goal and non-negotiable contract](../native-inference-goal.md)
- [Benchmark authority](../../BENCHMARK-AUTHORITY.md)
- [Native performance observability contract](../observability/native-performance-contract.md)
- [Native performance artifact guide](../observability/native-performance-artifacts.md)

Modern native/performance work prefers Qwen3.8. llama.cpp is used only for explicit benchmarks, reference comparisons, or borrowing. It is never an automatic recovery or convenience engine. A valid native receipt names fak ownership and the exact model/backend. It records the quality constraint, workload, and resource boundary. It also includes all setup, recovery, and verification overhead.

### Lab — LOCAL MODEL (ENVELOPE-DEPENDENT), then FLEET GPU / PRIVATE LAB when required

1. Inspect the available execution envelope before loading a payload:

   ```powershell
   fak runtime-capabilities --receipt-schema fak-execution-mode-receipt/1
   ```

2. If the task names a backend, inspect it exactly with `--backend NAME`; do not accept substitution.
3. For a native/performance run, use the canonical command and receipt named by the relevant Qwen3.8 plan or benchmark runbook. Dispatch accelerator work through the [fleet compute node map](../fleet-compute-nodes.md).
4. Record the exact engine, Qwen3.8 model/artifact, and backend/device. Also note the prompt/workload, concurrency, and context length. Include the quality gate, memory limit, warm/cold state, and overhead accounting.
5. If the envelope cannot be admitted, preserve the structured refusal or handoff. Do not silently run llama.cpp, Ollama, or another provider. Do not run `cpu-ref` or another model and relabel it native.

### Checkpoint

A reviewer must be able to answer yes to all of these from your receipt alone:

- Does `engine` equal `fak-native`?
- Is the Qwen3.8 model and artifact identity explicit?
- Is the backend/device explicit rather than inferred?
- Is quality constrained and measured beside performance?
- Are context and batch/concurrency in the envelope? Are memory, setup, recovery, and verification costs included?
- Is any degraded or remote placement explicit and authorized?

Otherwise mark the execution or performance claim as not yet witnessed.

## Module 3 — Follow the agent harness and syscall path

### Read

- [Architecture and extension model](../../ARCHITECTURE.md).
- [MCP integration](../integrations/mcp.md).
- [Harness composition](../integrations/harness-composition.md).
- [Harness verification runs](../integrations/harness-verify-run.md).
- [MCP stdio example and verifier](../../examples/mcp/README.md). This demonstrates schema-light bootstrap discovery, deferred `fak_admit` discovery, benign result admission, and allow/deny adjudication over the real editor transport.

The harness owns the model loop and exposes tools. fak owns the typed call boundary. Synchronous `Syscall` is defined over asynchronous `Submit`/`Reap`. Registries provide adjudicators, fast paths, and engines. They also supply result admitters and observers without making every new feature part of the hot path.

### Lab — LOCAL / OFFLINE

1. Run the offline harness from Module 1.
2. Read the `ToolCall`, syscall, and result structures named in the architecture document. Locate their current definitions under `internal/abi`, `internal/gateway`, and `internal/agent`.
3. Draw a sequence diagram for one allowed call and one denied call. Include the caller, harness, syscall boundary, and adjudicator fold. Show the vDSO lookup and dispatch engine. Also include the result admitter, context-MMU, observer, and model loop.
4. Run `python examples/mcp/verify.py --no-color`. Confirm all six checks pass. These include the handshake, schema-light `tools/list`, and deferred `fak_admit` discovery through `fak_tools_search`. They also cover benign result admission with a typed DEFER/OK envelope, `git_push` denial, and `git_status` allowance.

### Checkpoint

Explain, without implementation narration:

- Where a tool call becomes typed kernel input.
- Where a denial stops side effects.
- Why `Syscall` over `Submit`/`Reap` preserves one policy path.
- Where result bytes can be stopped even after a tool was allowed.
- Which components are harness responsibilities and which are kernel responsibilities.

## Module 4 — Policy, grammar, and adjudication

### Read

- [Policy reference](../../POLICY.md)
- [Customer-support policy example](../../examples/customer-support-readonly-policy.json)
- [Architecture: adjudicator registration and fold](../../ARCHITECTURE.md#how-a-new-idea-bakes-in-the-only-mechanism)

Policy serves as a deployable capability floor rather than a prompt suggestion. The preflight ladder combines tool registration, argument grammar, capability checks, and ordered adjudicators. Provable refusals deny. An adjudicator that cannot prove its case defers rather than inventing authority.

### Lab — LOCAL / OFFLINE

Repeat the two preflight calls from Module 1. Then add one malformed or unregistered call of your own using only flags documented under `fak preflight` in the CLI reference. Preserve each typed reason and identify the deciding rung.

Do not execute a real destructive tool for this exercise. Preflight exists to prove the refusal before dispatch.

### Checkpoint

For each case, provide:

- The normalized tool and arguments.
- The deciding rung and structured reason.
- The `ALLOW`, `DENY`, or defer behavior.
- An assessment of whether a side effect could have occurred.
- The distinction between a policy verdict and context admission of a later result.

## Module 5 — Shared prefix, vDSO, and the cache stack

### Read

- [Cache concept disambiguation](../concepts/disambiguation-cache.md)
- [Cache explainer](../explainers/cache.md)
- [Long sessions keep the cache hit](../explainers/long-sessions-keep-the-cache-hit.md)
- [Tool vDSO three-tier fast path claim](../claims/tool-vdso-3-tier-local-fast-path.md)
- [Addressable KV cache in five minutes](../explainers/addressable-kv-cache-in-5-min.md)

Shared prefixes reduce repeated provider or prefill work only when byte identity and the relevant cache contract hold. Tool-result vDSO reuse is a different mechanism from provider prompt caching, radix-prefix snapshots, model attention KV, and vCache. Invalidation, provenance, and economic accounting differ across them.

### Lab — LOCAL / OFFLINE

Build a cache ledger for one hypothetical two-turn agent session. For each reusable item, name:

- Its cache identity.
- Its owner and trust boundary.
- Its key or prefix identity.
- Its hit eligibility.
- Its invalidation or revocation condition.
- What is saved: network/tool work, provider input processing, prefill compute, or decode state.
- The evidence that would prove a hit rather than infer one.

Then inspect `internal/vdso` tests and run:

```powershell
go test ./internal/vdso
```

### Checkpoint

You are given five observations: provider cached-input tokens, a radix-prefix match, and a model KV page. You also observe a vCache entry and a local tool-result hit. Classify each observation without using the generic word “cache” alone. Reject any savings claim that lacks the corresponding hit evidence and denominator.

## Module 6 — Context-MMU lifecycle and trust-preserving context control

### Read

- [Context-MMU write-time result admission](../claims/context-mmu-write-time-result-admission.md)
- [Managed context continuous usage](../managed-context-continuous-usage.md)
- [Context shedding](../explainers/context-shedding.md)
- [Context and ctx disambiguation](../concepts/disambiguation-context-ctx.md)

The context-MMU screens results before they enter model-visible context. Safe bytes may be admitted. Harmful patterns like secrets, injections, poison, or repeats can be quarantined and replaced by a small content-addressed pointer. Restore is witness- and trust-gated. Revocation must evict causally dependent state rather than leaving poisoned reuse behind.

### Lab — LOCAL / OFFLINE

1. Read the context-MMU claim’s reproduction command and run the current `internal/ctxmmu` package tests:

   ```powershell
   go test ./internal/ctxmmu
   ```

2. Trace these states for both a safe result and a poison-shaped result. Follow produced, screened, and admitted or quarantined stages. Then follow paged out, pointer rendered, and restore attempted. Finally trace restored or refused, followed by revoked or evicted.
3. Explain what remains durable when content leaves the active context and what does not automatically become memory.

### Checkpoint

Your state diagram must show:

- A write-time trust decision before model visibility.
- A bounded pointer replacing quarantined bytes.
- A witness requirement for page-in.
- A refusal path for sealed, tombstoned, stale, or revoked content.
- Separation between context pressure management and durable memory selection.

## Module 7 — memq recall, render, and compact under trust gates

### Read

- [Memory engineering](../explainers/memory-engineering.md)
- [Agent memory integration](../integrations/agent-memory.md)
- [Context Is Not Memory](../CONTEXT-IS-NOT-MEMORY.md)
- Current implementation and tests under `internal/memq` and `internal/recall`

`memq` is a composable query algebra rather than a magical memory oracle. A plan can scan, filter, rank, and limit. It can also budget, deduplicate, and render items. Operators can tombstone, consolidate, reclassify, or prune records. Recall and render are bounded projections through provenance and trust gates. Mutating operations are proposal-only by default. Sealed spans are never rendered, and negative-only/storage mutations require explicit application.

### Lab — LOCAL / OFFLINE

1. Run the package tests:

   ```powershell
   go test ./internal/memq ./internal/recall
   ```

2. Using the current memory driver/tool schema documented in [Agent memory integration](../integrations/agent-memory.md), explain a built-in `recall` plan before running it.
3. Run a bounded recall against the demo corpus with a concrete intent, small `k`, and byte budget.
4. Explain a `compact` plan with `apply=false`. Identify every effect step and why it does or does not mutate durable state.
5. Do not set `apply=true` for the course. The learning objective focuses on proposed effects and trust refusals without altering a learner's memory store.

### Checkpoint

Submit the plan trace and rendered set, then answer:

- Which operator selected each item?
- Where was the byte budget enforced?
- Which provenance/trust state prevented rendering?
- Which compact effects were merely proposed?
- Why is a tombstone preferable to an unaudited hard delete?
- What evidence would be required before a stored memory is presented as fresh fact?

## Module 8 — Durable evidence, observability, and verification workflow

### Read

- [Observability map](../observability/README.md)
- [Durable artifacts](../observability/durable-artifacts.md)
- [Development tooling and build profiles](../dev-tooling.md)
- [Benchmark authority](../../BENCHMARK-AUTHORITY.md)
- [Claims registry](../../CLAIMS.md) and [status matrix](../../STATUS.md)

Observability is useful only when its records survive the run and preserve provenance. A trace helps diagnose issues. A receipt binds a claim to an operating envelope. A captured witness lets an independent reviewer reproduce or reject the claim. “Tests pass,” “fast,” and “native” are separate claims requiring different evidence.

### Lab — LOCAL / OFFLINE

Create an evidence matrix for Modules 1–7 with these columns:

- Claim.
- Controlling subsystem.
- Artifact or command.
- Observed versus self-reported status.
- Environment and envelope.
- Expected failure or refusal state.
- Independent verification step.

Run scope-correct checks for course-relevant code or docs without treating the peer-dirty working tree as your change:

```powershell
fak-dev ci-preflight
fak validate --mine docs/courses/end-to-end-inference-agent-harness-memory.md docs/_witnesses/issue-10424-learning-before.json
```

If `fak validate --mine` is unavailable in the current bootstrap state, record that honestly and use the documented isolated build/test primitive rather than a broad in-place build.

### Operations lesson — update without trusting the binary being replaced

The shipped `cmd/fak-selfupdate` surface is a recovery-sized executable over the same implementation as `fak self-update`. Use the ordinary verb when the installed `fak` is healthy. Keep the standalone entry point available when the main command is stale or is being replaced. Both emit the versioned `fak.self-update.receipt/v1` JSON contract.

The default native path builds from a repository, runs the green gate, and installs only after passing. `--check` provides a non-mutating inspection path, and even `--force` does not bypass the green gate. Optional signed-manifest selection adds channel, cohort, and authenticated-cache controls. In addition, `--offline` refuses network access and accepts only a valid authenticated cache.

On Windows, the explicit MSIX path verifies signed package provenance and requires an explicit opt-in for downgrade. Differential delivery falls back only to the declared full package. These bounded alternatives provide safety without accepting unsigned artifacts or skipping verification. A scheduled updater can also pin the executable path and must refuse provenance drift.

From the repository root, prove the recovery surface is present without replacing anything:

```powershell
go run ./cmd/fak-selfupdate --help
go run ./cmd/fak-selfupdate --check --target (Get-Command fak).Source --json
```

The first command exposes the standalone surface. The second inspects the installed target's embedded Go build metadata without executing that target. Read the exact flags and receipt fields in the [CLI reference](../cli-reference.md). Then consult the [self-update fast-path design note](../notes/CONCEPT-SELF-UPDATE-FAST-PATHS-2026-08-29.md) when reasoning about manifest, delta/full-fallback, MSIX, or handoff boundaries.

### Checkpoint

A peer should be able to reconstruct what ran and where it ran. They should identify the engine, model, and policy. They should also verify the envelope boundaries and the observed result—all without trusting your prose. Treat any missing dimension as an explicit limitation instead of an inferred success.

## Capstone — One governed Qwen3.8 agent turn, end to end

### Brief

Design and execute, or dispatch, one bounded agent task that reads from an allowed knowledge tool and proposes—but does not perform—a sensitive action. The run must connect all eight modules.

### Required path

1. **Admit the runtime.** Capture `runtime-capabilities`; distinguish control-only readiness from model execution.
2. **Pin native execution.** Use Qwen3.8 on a fak-native backend. If acceleration is required, dispatch to the sanctioned fleet/private-lab path. Keep native/performance execution fak-native. Select llama.cpp only explicitly for benchmarks, parity/reference diagnosis, study, or borrowing. Never use it as a silent recovery path. Likewise, never silently substitute an external provider, a CPU path, or a different model.
3. **Declare the envelope.** Record the model/artifact, backend/device, and context length. Note the concurrency, memory budget, and workload. Finally, record the quality criterion and overhead accounting.
4. **Compose the harness.** Show the model loop, exposed tool schema, and typed syscall boundary.
5. **Prove policy.** Preflight the allowed read and the denied sensitive action; retain typed verdicts.
6. **Account for reuse.** Identify the shared stable prefix. Distinguish vDSO, provider, and radix reuse. Also separate vCache and model-KV evidence without conflating them.
7. **Admit results.** Show the context-MMU decision for the returned data, including a quarantine case if the fixture contains poison-shaped text.
8. **Use memory safely.** Run a bounded `memq` recall/render plan and a proposal-only compact plan; preserve trust refusals.
9. **Capture evidence.** Produce a manifest that points to commands, receipts, and traces. Include quality results, policy verdicts, and context decisions. Link memory plan traces and verification output as well.
10. **Verify independently.** Have a reviewer reproduce the local control-plane slice and inspect the native receipt and private-lab/public evidence boundary.

### Capstone acceptance rubric

The capstone passes only when all conditions are met:

- The task completes or reaches a typed refusal without bypassing policy.
- Native model evidence explicitly names fak-native Qwen3.8 and the backend.
- Quality and performance are reported in the same declared envelope.
- The harness-to-syscall path and result-admission path are both evidenced.
- Every claimed cache hit names its cache identity and witness.
- Quarantined content does not appear in model-visible context.
- Memory output is bounded, provenance-bearing, and trust-gated.
- Private infrastructure details remain private while the public receipt remains reproducible.
- Local verification uses isolated, scope-correct commands.
- Limitations are stated as limitations rather than converted into success.

A control-only demonstration is a valid partial artifact but not a completed native-inference capstone. A llama.cpp reference run may be attached as an explicitly labeled comparison, but it cannot satisfy the fak-native execution requirement.

## Next routes

Choose the route that matches the next responsibility:

- **Operate or integrate a harness:** [integration index](../integrations/README.md), [adopter playbook](../integrations/adopter-playbook.md), and [harness acceptance checklist](../integrations/harness-acceptance-checklist.md).
- **Extend the kernel:** [Extending fak](../../EXTENDING.md) and the registry seams in [Architecture](../../ARCHITECTURE.md).
- **Optimize native inference:** [native inference goal](../native-inference-goal.md), [SOTA index](../sota/README.md), and [benchmark authority](../../BENCHMARK-AUTHORITY.md). Keep Qwen3.8 and fak-native ownership explicit.
- **Deepen cache work:** [cache explainer](../explainers/cache.md), [managed cache](../explainers/what-is-managed-cache.md), and [cache value rollup](../cache-value-rollup.md).
- **Deepen context and memory:** [managed context glossary](../managed-context-glossary.md), [memory engineering](../explainers/memory-engineering.md), and [agent memory integration](../integrations/agent-memory.md).
- **Build durable observability:** [observability map](../observability/README.md), [trajectory observability](../observability/trajectory.md), and [durable artifacts](../observability/durable-artifacts.md).
- **Run hardware-gated work:** [fleet compute nodes](../fleet-compute-nodes.md) and the [private communications channel](../private-comms-channel.md). Keep the public/private boundary intact.

## Completion record

A learner's final record should contain:

- The eight module checkpoints.
- The capstone manifest and acceptance review.
- An explicit list of labs completed locally, with a local model, or on a fleet/private-lab node.
- All refusals and missing witnesses preserved verbatim.
- No copied canonical documentation beyond the minimum command snippets needed to perform the labs.

The course is complete when another engineer can follow the evidence from operator intent to verified outcome. They must be able to identify where fak controlled, accelerated, or refused the run. They must also see where it quarantined, recalled, or merely observed behavior.


