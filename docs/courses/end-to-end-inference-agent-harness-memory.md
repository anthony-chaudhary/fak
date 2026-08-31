# End-to-End Inference, Agent Harness, and Memory

> **Flagship bounded course.** Follow one fak request from runtime admission through fak-native model execution, agent tool use, policy, local fast paths, context admission, memory recall, and durable verification. This page is a syllabus and lab guide; the linked canonical documents remain authoritative.

## Course contract

### Audience

This course is for engineers who can read Go and operate a command line and who want to build, integrate, review, or run a modern fak deployment. It is suitable for agent-harness authors, inference engineers, security and policy reviewers, and operators responsible for evidence-backed releases.

It is not an introduction to transformers, Go syntax, Git, or shell basics. It also does not replace the subsystem references, benchmark authority, or private lab runbooks.

### Prerequisites and background

Before starting, you should be able to:

- build and test a Go module;
- read JSON receipts and distinguish a claim from an observed artifact;
- explain prompts, tokens, model weights, prefill, decode, and a KV cache at a basic level;
- explain why untrusted tool output must not automatically become trusted model context;
- use explicit paths in a peer-dirty checkout.

Start with [Getting Started](../../GETTING-STARTED.md), the [reproduction packet](../repro-packet.md), and the [architecture overview](../../ARCHITECTURE.md). Keep the [CLI reference](../cli-reference.md) open throughout the course.

### Environment markers

Every lab carries one of these markers:

- **LOCAL / OFFLINE** — no API key, model, or GPU is required.
- **LOCAL MODEL (ENVELOPE-DEPENDENT)** — a supported local fak-native backend and compatible model artifact are required. Inspect admission before loading a payload.
- **FLEET GPU / PRIVATE LAB** — accelerator evidence must run on a sanctioned compute node. The workstation is the control point, not the compute boundary. Follow the public [fleet compute node map](../fleet-compute-nodes.md) and enter private infrastructure only through the [private communications channel](../private-comms-channel.md). Never copy credentials, private hostnames, private paths, or raw internal logs into public course work.

Absence of local GPU hardware is not a terminal result. Dispatch the GPU lab or record the exact sanctioned handoff; do not substitute llama.cpp and call the result fak-native.

### Measurable outcomes

By the end, you will be able to:

1. distinguish executable, control-plane, and model-execution readiness from a `fak-runtime-capabilities/1` or execution-mode receipt;
2. prove that native/performance evidence names `engine: "fak-native"`, a current Qwen3.8 model, a backend, and an explicit operating envelope;
3. trace a typed tool call through harness, syscall, adjudication, dispatch or vDSO, result admission, and model context;
4. explain policy precedence, structural denial, and why policy success does not prove model execution;
5. distinguish provider prompt cache, radix-prefix reuse, model KV cache, vCache, and tool-result vDSO reuse;
6. describe the context-MMU write-time lifecycle: admit, quarantine/page out, restore under a witness, and evict stale or poisoned state;
7. construct and explain a bounded `memq` recall/render/compact plan, including trust gates and proposal-only mutation defaults;
8. produce a compact evidence packet that separates delivery, quality, performance, security, and operating-envelope claims;
9. run the scope-correct local verification workflow without mixing peer WIP into the result.

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

The first objective is deliberately model-free: establish that the binary runs, policy can structurally deny one tool and allow another, and the offline agent still completes useful work. Then inspect runtime capability without confusing that inspection with payload execution.

### Lab — LOCAL / OFFLINE

From the repository root, use a temporary output path rather than writing an in-tree binary:

```powershell
go build -o "$env:TEMP/fak-course.exe" ./cmd/fak
& "$env:TEMP/fak-course.exe" preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
& "$env:TEMP/fak-course.exe" preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
& "$env:TEMP/fak-course.exe" agent --offline
& "$env:TEMP/fak-course.exe" runtime-capabilities
```

Capture the commands, exit status, and machine-readable output. Label each observation as **binary**, **control plane**, or **model execution** evidence.

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

Modern native/performance work prefers Qwen3.8. llama.cpp is an explicit benchmark, parity, reference, migration, interoperability, or borrowing choice only; it is never an automatic recovery or convenience engine. A valid native receipt names fak ownership, the exact model/backend, quality constraint, workload, resource boundary, and all setup, recovery, and verification overhead included in the claim.

### Lab — LOCAL MODEL (ENVELOPE-DEPENDENT), then FLEET GPU / PRIVATE LAB when required

1. Inspect the available execution envelope before loading a payload:

   ```powershell
   fak runtime-capabilities --receipt-schema fak-execution-mode-receipt/1
   ```

2. If the task names a backend, inspect it exactly with `--backend NAME`; do not accept substitution.
3. For a native/performance run, use the canonical command and receipt named by the relevant Qwen3.8 plan or benchmark runbook. Dispatch accelerator work through the [fleet compute node map](../fleet-compute-nodes.md).
4. Record the exact engine, Qwen3.8 model/artifact, backend/device, prompt/workload, concurrency, context length, quality gate, memory limit, warm/cold state, and overhead accounting.
5. If the envelope cannot be admitted, preserve the structured refusal or handoff. Do not silently run llama.cpp, Ollama, another provider, `cpu-ref`, or a different model and relabel it native.

### Checkpoint

A reviewer must be able to answer **yes** to all of these from your receipt alone:

- Does `engine` equal `fak-native`?
- Is the Qwen3.8 model and artifact identity explicit?
- Is the backend/device explicit rather than inferred?
- Is quality constrained and measured beside performance?
- Are context, batch/concurrency, memory, setup, recovery, and verification costs in the envelope?
- Is any degraded or remote placement explicit and authorized?

Otherwise mark the execution or performance claim **not yet witnessed**.

## Module 3 — Follow the agent harness and syscall path

### Read

- [Architecture and extension model](../../ARCHITECTURE.md)
- [MCP integration](../integrations/mcp.md)
- [Harness composition](../integrations/harness-composition.md)
- [Harness verification runs](../integrations/harness-verify-run.md)
- [MCP example and known verifier drift](../../examples/mcp/README.md) — the example explains the intended contract, but its checked-in verifier currently fails because `tools/list` omits `fak_admit`; track the repair in [issue #10449](https://github.com/anthony-chaudhary/fak/issues/10449).

The harness owns the model loop and exposes tools. fak owns the typed call boundary: synchronous `Syscall` is defined over asynchronous `Submit`/`Reap`; registries provide adjudicators, fast paths, engines, result admitters, and observers without making every new feature part of the hot path.

### Lab — LOCAL / OFFLINE

1. Run the offline harness from Module 1.
2. Read the `ToolCall`, syscall, and result structures named in the architecture document and locate their current definitions under `internal/abi`, `internal/gateway`, and `internal/agent`.
3. Draw a sequence diagram for one allowed call and one denied call. Include caller, harness, syscall boundary, adjudicator fold, vDSO lookup, dispatch, result admitter, context-MMU, observer, and model loop.
4. Do **not** treat the checked-in MCP verifier as a passing lab today: `python examples/mcp/verify.py` currently fails at the `fak_admit` surface check. Inspect the failure as a contract-drift case and follow [issue #10449](https://github.com/anthony-chaudhary/fak/issues/10449) for the repair; rerun it only after that issue lands.

### Checkpoint

Explain, without implementation narration:

- where a tool call becomes typed kernel input;
- where a denial stops side effects;
- why `Syscall` over `Submit`/`Reap` preserves one policy path;
- where result bytes can be stopped even after a tool was allowed;
- which components are harness responsibilities and which are kernel responsibilities.

## Module 4 — Policy, grammar, and adjudication

### Read

- [Policy reference](../../POLICY.md)
- [Customer-support policy example](../../examples/customer-support-readonly-policy.json)
- [Architecture: adjudicator registration and fold](../../ARCHITECTURE.md#how-a-new-idea-bakes-in-the-only-mechanism)

Policy is a deployable capability floor, not a prompt suggestion. The preflight ladder combines tool registration, argument grammar, capability and policy checks, and ordered adjudicators. Provable refusals deny; an adjudicator that cannot prove its case defers rather than inventing authority.

### Lab — LOCAL / OFFLINE

Repeat the two preflight calls from Module 1, then add one malformed or unregistered call of your own using only flags documented under `fak preflight` in the CLI reference. Preserve each typed reason and identify the rung that decided it.

Do not execute a real destructive tool for this exercise. Preflight exists to prove the refusal before dispatch.

### Checkpoint

For each case, provide:

- normalized tool and arguments;
- deciding rung and structured reason;
- `ALLOW`, `DENY`, or defer behavior;
- whether a side effect could have occurred;
- the distinction between a policy verdict and context admission of a later result.

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

- its cache identity;
- its owner and trust boundary;
- its key or prefix identity;
- hit eligibility;
- invalidation or revocation condition;
- what is saved: network/tool work, provider input processing, prefill compute, or decode state;
- the evidence that would prove a hit rather than infer one.

Then inspect `internal/vdso` tests and run:

```powershell
go test ./internal/vdso
```

### Checkpoint

Given five observations—provider cached-input tokens, a radix-prefix match, a model KV page, a vCache entry, and a local tool-result hit—classify each without using the generic word “cache” alone. Reject any savings claim that lacks the corresponding hit evidence and denominator.

## Module 6 — Context-MMU lifecycle and trust-preserving context control

### Read

- [Context-MMU write-time result admission](../claims/context-mmu-write-time-result-admission.md)
- [Managed context continuous usage](../managed-context-continuous-usage.md)
- [Context shedding](../explainers/context-shedding.md)
- [Context and ctx disambiguation](../concepts/disambiguation-context-ctx.md)

The context-MMU screens results before they enter model-visible context. Safe bytes may be admitted; secret-, injection-, poison-, or repeat-shaped bytes can be quarantined and replaced by a small content-addressed pointer. Restore is witness- and trust-gated. Revocation must evict causally dependent state rather than leaving poisoned reuse behind.

### Lab — LOCAL / OFFLINE

1. Read the context-MMU claim’s reproduction command and run the current `internal/ctxmmu` package tests:

   ```powershell
   go test ./internal/ctxmmu
   ```

2. For a safe result and a poison-shaped result, trace these states: produced, screened, admitted or quarantined, paged out, pointer rendered, restore attempted, restored or refused, and revoked/evicted.
3. Explain what remains durable when content leaves the active context and what does not automatically become memory.

### Checkpoint

Your state diagram must show:

- a write-time trust decision before model visibility;
- a bounded pointer replacing quarantined bytes;
- a witness requirement for page-in;
- a refusal path for sealed, tombstoned, stale, or revoked content;
- separation between context pressure management and durable memory selection.

## Module 7 — memq recall, render, and compact under trust gates

### Read

- [Memory engineering](../explainers/memory-engineering.md)
- [Agent memory integration](../integrations/agent-memory.md)
- [Context Is Not Memory](../CONTEXT-IS-NOT-MEMORY.md)
- Current implementation and tests under `internal/memq` and `internal/recall`

`memq` is a composable query algebra, not a magical memory oracle. A plan can scan, filter, rank, limit, budget, deduplicate, render, tombstone, consolidate, reclassify, or prune. Recall and render are bounded projections through provenance and trust gates. Mutating operations are proposal-only by default; sealed spans are never rendered, and negative-only/storage mutations require explicit application.

### Lab — LOCAL / OFFLINE

1. Run the package tests:

   ```powershell
   go test ./internal/memq ./internal/recall
   ```

2. Using the current memory driver/tool schema documented in [Agent memory integration](../integrations/agent-memory.md), explain a built-in `recall` plan before running it.
3. Run a bounded recall against the demo corpus with a concrete intent, small `k`, and byte budget.
4. Explain a `compact` plan with `apply=false`. Identify every effect step and why it does or does not mutate durable state.
5. Do not set `apply=true` for the course. The learning objective is to reason about the proposed effects and trust refusals, not to mutate a learner’s memory store.

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

Observability is useful only when its records survive the run and preserve provenance. A trace helps diagnose; a receipt binds a claim to an operating envelope; a captured witness lets an independent reviewer reproduce or reject it. “Tests pass,” “fast,” and “native” are separate claims requiring different evidence.

### Lab — LOCAL / OFFLINE

Create an evidence matrix for Modules 1–7 with these columns:

- claim;
- controlling subsystem;
- artifact or command;
- observed versus self-reported;
- environment and envelope;
- expected failure/refusal state;
- independent verification step.

Run scope-correct checks for course-relevant code or docs without treating the peer-dirty working tree as your change:

```powershell
fak-dev ci-preflight
fak validate --mine docs/courses/end-to-end-inference-agent-harness-memory.md docs/_witnesses/issue-10424-learning-before.json
```

If `fak validate --mine` is unavailable in the current bootstrap state, record that honestly and use the documented isolated build/test primitive rather than a broad in-place build.

### Checkpoint

A peer should be able to reconstruct what ran, where, with which engine/model/policy, inside which envelope, and with what result—without trusting your prose. Any missing dimension becomes an explicit limitation, not an inferred success.

## Capstone — One governed Qwen3.8 agent turn, end to end

### Brief

Design and execute, or dispatch, one bounded agent task that reads from an allowed knowledge tool and proposes—but does not perform—a sensitive action. The run must connect all eight modules.

### Required path

1. **Admit the runtime.** Capture `runtime-capabilities`; distinguish control-only readiness from model execution.
2. **Pin native execution.** Use Qwen3.8 on a fak-native backend. If acceleration is required, dispatch to the sanctioned fleet/private-lab path. Keep native/performance execution fak-native: select llama.cpp only explicitly for benchmark, parity/reference diagnosis, interoperability/migration, study, or borrowing, and never as silent recovery; likewise, do not silently substitute a provider, CPU path, or different model.
3. **Declare the envelope.** Record model/artifact, backend/device, context length, concurrency, memory, workload, quality criterion, and overhead accounting.
4. **Compose the harness.** Show the model loop, exposed tool schema, and typed syscall boundary.
5. **Prove policy.** Preflight the allowed read and the denied sensitive action; retain typed verdicts.
6. **Account for reuse.** Identify the shared stable prefix and any vDSO, provider, radix, vCache, or model-KV evidence without conflating them.
7. **Admit results.** Show the context-MMU decision for the returned data, including a quarantine case if the fixture contains poison-shaped text.
8. **Use memory safely.** Run a bounded `memq` recall/render plan and a proposal-only compact plan; preserve trust refusals.
9. **Capture evidence.** Produce a manifest that points to commands, receipts, traces, quality results, policy verdicts, context decisions, memory plan traces, and verification output.
10. **Verify independently.** Have a reviewer reproduce the local control-plane slice and inspect the native receipt and private-lab/public evidence boundary.

### Capstone acceptance rubric

The capstone passes only when all are true:

- the task completes or reaches a typed refusal without bypassing policy;
- native model evidence explicitly names fak-native Qwen3.8 and the backend;
- quality and performance are reported in the same declared envelope;
- the harness-to-syscall path and result-admission path are both evidenced;
- every claimed cache hit names its cache identity and witness;
- quarantined content does not appear in model-visible context;
- memory output is bounded, provenance-bearing, and trust-gated;
- private infrastructure details remain private while the public receipt remains reproducible;
- local verification uses isolated, scope-correct commands;
- limitations are stated as limitations, not converted into success.

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

A learner’s final record should contain:

- the eight module checkpoints;
- the capstone manifest and acceptance review;
- an explicit list of labs completed locally, with a local model, or on a fleet/private-lab node;
- all refusals and missing witnesses preserved verbatim;
- no copied canonical documentation beyond the minimum command snippets needed to perform the labs.

The course is complete when another engineer can follow the evidence from operator intent to verified outcome and can identify exactly where fak controlled, accelerated, refused, quarantined, recalled, or merely observed the run.


