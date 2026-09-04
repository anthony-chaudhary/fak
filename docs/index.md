---
title: "fak — the agent runtime | cheaper long sessions, the right model per call, audited tool-call control"
description: "fak is an agent runtime: one operator-controlled boundary for cache and context, model routing, tool authority, memory, observability, and native inference. Its Fused Agent Kernel architecture ships as one Go binary."
---

# fak documentation

**Primary audience:** people entering the documentation who need to choose a current route by job. **fak is an agent runtime: the operator-controlled boundary for cache and context, model routing, tool authority, memory, observability, and native inference.** Its technical architecture is the **Fused Agent Kernel**—an **agent kernel** shipped as one Go binary. Each operating mode exposes a documented subset of that boundary; use the claims ledger rather than the category label to decide what is shipped.

FAK coordinates the whole agent path, not isolated components. Start with the [canonical architecture map](architecture.md#whole-path-coordination-target) for the five-layer observation → constrained-plan → typed-effect contract, then use the [glossary](glossary.md#whole-path-decision-terms) to keep coordination, orchestration, scheduling, routing, and serving distinct.

**Default next action:** run the deterministic offline proof in [the reproducibility packet](repro-packet.md) with `fak agent --offline`. It needs no API key, model, or GPU and ends with checkable task-completion and blocked-operation results.

## Choose your route

| You are… | Start here | Use this route to… |
|---|---|---|
| Evaluating fak | [Reproducibility packet](repro-packet.md) | Identify the product, run the offline proof, then inspect its evidence. |
| Learning the whole system | [8-module flagship course](courses/end-to-end-inference-agent-harness-memory.md) | Follow one request across native inference, the agent harness, policy, context, memory, observability, and proof; use the [99-course learning path](../LEARNING-PATH.md) for prerequisite ordering and deeper study. |
| Building or integrating an agent or client | [Agent runtime](explainers/agent-runtime.md) | Understand the category and ownership boundary, choose an interface, and follow the proposal-to-continuation flow. |
| Improving or comparing local inference | [Fak-native inference doctrine](native-inference-goal.md) | Keep native product work inside fak, classify explicit llama.cpp uses, and apply the matched-envelope rule. |
| Deploying or operating | [Deployment guide](fak/deployment-guide.md) | Choose an operating envelope, then configure and observe the service. |
| Contributing | [Contributor guide](../CONTRIBUTING.md) and [developer tooling](dev-tooling.md) | Find the owning document first, then build, test, change, and prove the repository under its current contracts. |
| Researching design or history | [Notes archive](notes/) | Find rationale and dated evidence, then check current code and tests before relying on it. |

For a human role map, use [`START-HERE.md`](../START-HERE.md). Agents that need a compact authority map should use [`llms.txt`](../llms.txt). The exhaustive audience, task, and lifecycle catalog is [`INDEX.md`](../INDEX.md).

## Authority and lifecycle

This page is the **current, public documentation landing page** for the current generation. Runtime behavior is authoritative in code and tests; operational pages link to their owning commands and proofs. Pages marked experimental, simulated, stubbed, superseded, or historical describe a narrower lifecycle and do not override current authorities. Dated material under [`notes/`](notes/) provides provenance rather than the default product contract.

This landing page supports route selection for the offline proof, managed runtime, integrations, HTTP service, policy floor, deployment, and contribution workflow. It does not establish availability for accelerator hardware or private control channels; those environment-specific routes state their own prerequisites and support boundary. Begin with the offline proof unless your task requires one of those envelopes.

## Choose the smallest efficiency layer

- [Less context, less code: where fak fits beside Caveman and Ponytail](explainers/less-context-less-code.md) — answer-first guide to concise output, YAGNI/minimal-code guidance, and fak runtime cache, context, reuse, recovery, and policy.

## What fak does

The everyday wins first — the reasons most people put `fak` in front of an agent:

- **Cheaper long sessions.** A long conversation re-sends its whole transcript every
  turn, and the provider only discounts it while the cached prefix stays byte-for-byte
  the same. `fak` sheds the un-cacheable middle turns by splicing on the original bytes
  (a memcpy, never a re-marshal), so the prompt-cache discount survives instead of
  breaking. It guarantees prefix byte-identity, and relays the provider's cache number
  rather than claiming it.
- **The right model per call.** `fak route` routes an *aspect* (a tool call, a
  reasoning step, a stage) to a different model, with first-class ensembles
  (`vote`, `best_of`). An easy read goes to a cheap model; a write-shaped call goes to
  a careful one.
- **Fewer wasted turns.** A repeated read is served locally, a malformed call is
  repaired in place, and a dead-end branch is refused before the agent spends a turn on
  it. Shared work is computed once because the KV cache is a kernel object, not a rented
  one.
- **A trail you can audit.** Every decision is a plain verdict (`ALLOW`, `DENY`,
  `TRANSFORM`, or `QUARANTINE`) in JSON logs, an optional hash-chained journal, and
  Prometheus metrics.

And the tool-call control floor, for teams that need one (more in
[Tool-call controls](#tool-call-controls)):

- **Stops prompt injection and tool poisoning by structure.** Suspicious tool
  *results* are quarantined out of the model's context entirely; dangerous tools are
  never on the allow-list. Two independent gates, not one evadable classifier.
  Addresses the OWASP Agentic Top-10 and the MCP Top-10 (Tool Poisoning, Memory
  Poisoning).
- **Default-deny capability security.** The permission policy runs *inside* the
  kernel, on the same call path as the tool call. It fails **closed**, not open.
- **Addressable, bit-exact KV cache.** Evict one span from the middle of a kept
  model run — a poisoned result, an expired secret — and leave the cache
  bit-for-bit identical to a run that never saw it (`max|Δ| = 0`). No shipped serving
  engine offers mid-run causal eviction.
- **Cache-efficient agent fleets.** ~4× fewer tokens than a tuned warm-cache stack
  on a 50-turn × 5-agent run; 8.8–9.7× modeled prefill elimination vs the naive
  floor over the real WebVoyager web-agent set (1.0–1.1× vs a tuned per-agent KV).

## See each win in one example

Each idea shrinks to a single worked example. The numbers trace to the
[benchmark authority](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md);
the live versions run on the [demos page](demos.html). Or watch the worked examples as a
[~25-second reveal](showcase.html#by-example).

- **A poisoned turn, removed mid-run.** Quarantine evicts a tool result's K/V from the *middle* of the
  kept run and re-seats every survivor, leaving the cache bit-identical to one that never saw it
  (`max|Δ| = 0`). → [Watch a turn vanish](explainers/addressable-kv-cache.md#a-worked-example-watch-one-turn-vanish-bit-for-bit)
- **More tool calls, more turns saved.** On one 14-call agent trace a naive loop is forced into 9 extra
  model round-trips and a tuned 2026 framework into 5; the kernel resolves them in-syscall, for 0. →
  [The turn that never fires](concepts-and-story.md#three-worked-examples-more-turns-more-agents-more-tool-calls)
- **Pay the shared prefix once.** 5 agents × 50 turns is 250 chances to re-read the setup: naive pays
  250×, a tuned warm cache 5×, fak once: 4.1× vs tuned, 62.0× fewer prefill tokens. →
  [The setup-payments table](concepts-and-story.md#three-worked-examples-more-turns-more-agents-more-tool-calls)
- **More hooks, sooner.** Four checks across 1,000 tool calls is ~28 s of gate latency if you spawn a
  hook per check, or ~10 ms in-process, which is what makes fail-closed the default. →
  [The cost of checking everything](explainers/policy-in-the-kernel.md#a-worked-example-the-cost-of-checking-everything-every-time)

## What fak is not

`fak serve` in proxy/gateway mode is **not** a claim that fak authored the upstream token
engine. An explicitly selected vLLM, SGLang, llama.cpp, or hosted-provider route remains
external inference while fak owns the agent boundary: which effects are allowed, which
results may enter memory, when reuse is legal, what gets audited, and what survives a
session boundary.

That gateway boundary does not turn the native engine into a permanent reference-only path.
For local inference, [fak-native is the product and performance path](native-inference-goal.md),
intended to beat llama.cpp in matched, quality-constrained envelopes while retaining ownership
of kernels, memory, scheduling, cache, adaptation, and operations. Current broad serving-speed
claims still need a benchmark-authority row; the doctrine is the direction, not a substitute for
evidence.

## Tool-call controls

If a hard capability floor is *why* you're here — not just a nice-to-have — this is the
load-bearing idea.

**Treat the model like an untrusted program, and the tool call like a syscall: the
model proposes, the kernel disposes.** Most agent security tries to recognize bad text.
Recognizers help; they are not the floor. Prompt injection is a text game, and attackers
get turns too. `fak` moves the load-bearing decision to the capability floor: a dangerous
tool outside the allow-list cannot be called, no matter what the model was told.

Two independent gates matter:

- **Call-side gate:** tool names and selected arguments are checked before dispatch, on
  the same call path as the tool call (one address space, no IPC, `default-deny`). A
  denied call never reaches the tool runner, and a check that crashes or times out fails
  **closed**.
- **Result-side gate:** tool output is screened before it enters context. A poisoned or
  secret-bearing result is paged out or quarantined instead of being handed back to the
  model as trusted text. The detector is treated as evadable by design, a bonus rather
  than the floor; the floor is the dangerous lever simply not existing.

The capability floor is the guarantee. Irreversible effects are unwired by default;
untrusted bytes have to pass a gate before they become model context. Read
[Policy in the kernel](explainers/policy-in-the-kernel.md),
[POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md), and
[the security model](https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/security.md).

## Try it in 2 minutes (no key, no model, no GPU)

Get the binary — no clone, no Go toolchain. The installer detects your OS/arch,
downloads the prebuilt static binary for the latest release, verifies its checksum, and
drops `fak` on your PATH:

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak version          # prints the installed version, e.g. 0.34.0
```

Now prove the floor from the bare binary — these need no clone and no `examples/` dir:

```bash
fak preflight --tool refund_payment --args "{}"   # -> DENY  (DEFAULT_DENY): unknown tool, fail-closed
fak preflight --tool search_kb      --args "{}"   # -> ALLOW: a read-shaped name is not blanket-blocked
fak agent --offline                               # runs one task twice — tools wired directly vs. behind fak — and prints the before/after
```

The dangerous action is refused by structure, before any model interpretation matters.
Then wrap the agent you already run — one command, no rewrite, no key to start:

```bash
fak manage claude           # short: fak m claude; or: fak manage --provider openai -- opencode
```

> **Have the source already?** From a clone you can skip the install and run the same
> proof against a named example floor, where the deny is by *argument value*:
> `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"`.
> Full paths in [INSTALL.md](https://github.com/anthony-chaudhary/fak/blob/main/INSTALL.md)
> (one-line installer · manual download · Docker · build-from-source · Windows).

## Learn more

- **Token-efficiency field map:** [Awesome Token Efficiency](awesome-token-efficiency.md) — prompt caching, context engineering, KV-cache reduction, serving, and agent-layer methods with loss/fidelity and fak-status labels.
| If you want… | Read |
|---|---|
| **Codex UserPromptSubmit modes, capability floor, and installer/runtime verbs** | [OpenAI Codex integration — UserPromptSubmit modes](integrations/openai-codex.md#userpromptsubmit-modes) |
| **The principles fak is built to satisfy** | [Charter](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/CHARTER.md) |
| **What changed recently** | [Witnessed recent changes](whats-new.md) — generated from authoritative commits, issues, claims, and module versions; freshness-bounded and grouped for humans |
| **Structured-output decoding SOTA + fak's ride-mode surface (#907)** | [Research note](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/RESEARCH-structured-output-decoding-2026-06-26.md) |
| **Prior art + threat model for a centrally-administered org policy plane (epic #5315)** | [Research note](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/RESEARCH-org-policy-plane-prior-art-2026-07-20.md) |
| **Org-policy precedence lattice: compiled-in FROZEN floor > central > operator > agent-self (R3 / #5318)** | [Research note](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/RESEARCH-org-policy-precedence-2026-07-20.md) |
| **Keeping a stable core as models × backends × features multiply** | [Combinatorial-growth epic](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/COMBINATORIAL-GROWTH-EPIC-2026-06-27.md) |
| **Current performance borrow map for agentic and model-runtime sources** | [Research note](notes/CONCEPT-PERFORMANCE-BORROW-MAP-2026-08-25.md) |
| **Related-system inventory contract for deep `study-repo` passes** | [Research note](notes/CONCEPT-RELATED-SYSTEM-INVENTORY-2026-08-25.md) |
| **Qwen4 experimental support rollback watch and cutover evidence** | [Operational note](notes/QWEN4EXP-SUPPORT-ROLLBACK-WATCH-2026-08-26.md) |
| **Choosing repository indexes for exhaustive study inventories** | [Decision matrix](notes/REPO-INDEX-BACKEND-DECISION-2026-08-25.md) |
| **Constructing many on-demand "views" of the token history at marginal cost (attention/KV side-cars, re-attend tiers)** | [Research note](https://github.com/anthony-chaudhary/fak/blob/main/docs/notes/CONTEXT-VIEWS-AT-MARGINAL-COST-2026-07-04.md) |
| **The quick answers** | [FAQ](FAQ.md) |
| **A guided first run** | [Tutorial](fak/tutorial.md) |
| **What the words mean** (preflight vs inflight vs prefill; cache rebate / net saving) | [Glossary](glossary.md) |
| **How shared state is split** | [Shared state ladder](shared-state-ladder.md) |
| **A collaborative task state contract** | [Shared task record contract](shared-task-record-contract.md) |
| **When managed context should append or reconstruct the task** | [Query, not chat](query-not-chat.md) — originating-task pin and checkable reseed-versus-append rule |
| **How to construct model-visible directives** | [Positive-state construction](positive-state-construction.md) — broadcast the target state instead of a negation operand |
| **How negframe and managed context form one pipeline** | [Shared-workspace positive state](shared-workspace-positive-state.md) — exact gateway emit seam, wired surfaces, and current limits |
| **Why positive workspace management beats punitive default-deny** | [Positive workspace management](positive-workspace-management.md) — immutable FROZEN safety floor vs permissive convenience surface, avoiding capability laundering and doom-loops |
| **How new work becomes readable and dispatchable** (outcome, leaf, attempt, witness; explicit scope, dependencies, acceptance, and placement) | [Shift-left task organization](shift-left-task-organization.md) |
| **How every new unit of work is scoped and shipped** (applied spine first, then exhaustive proof, measured optimization, and backlog fan-out) | [Spine-first + fan-out defaults](spine-first-defaults.md) |
| **How agents discover fak features and memory tools** | [Self-feature query spine](notes/SELF-FEATURE-QUERY-SPINE-2026-06-30.md) |
| **The two core ideas** | [Policy in the kernel](explainers/policy-in-the-kernel.md) · [Addressable KV cache](explainers/addressable-kv-cache.md) |
| **How named context is loaded, filtered, cached, and backed by call snapshots** | [Context as a variable](explainers/context-as-a-variable.md) |
| **Why a cache-hit % isn't the whole story** | [Context signal-to-noise](explainers/context-signal-to-noise.md) |
| **How fak runs the agent as nested loops** | [Engineering is building loops](explainers/engineering-is-building-loops.md) |
| **How agent lifecycle, model, and fleet scale differ** | [Agent scale hierarchy](concepts/agent-scale-hierarchy.md) — macro, baseline, sub-agent, and micro |
| **What a micro agent is, and when to use one** | [Micro agents](concepts/micro-agents.md) — definition, lifecycle, limits, and a no-key example |
| **How agent fleets coordinate workers safely** | [Fleet concepts](concepts/fleet.md) — workers, lanes, leases, seats, monitoring, and independent witnesses |
| **Why a loader can pass every shape/dtype check and still be wrong** | [Semantic transform contracts](explainers/loader-semantic-transform-contracts.md) — the tensor-meaning defect class and the contract that catches it |
| **Every benchmark number** | [Benchmark authority](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md) |
| **Every per-run benchmark sheet** (results · runbooks · pending/gated) | [docs/benchmarks index](benchmarks/README.md) |
| **Everything fak supports** | [What fak supports](supported/README.md) — models · features · clouds · APIs/MCP · harnesses · engines |
| **Every machine fak runs on** | [Hardware matrix](HARDWARE-MATRIX.md) (4 platforms · 2 CPU ISAs · 4 GPU backends) |
| **How fak serves at scale** | [Serving plans](serving/README.md) — dual-track · poly-model · hardware-aware & regenerable KV |
| **What's real, what's not** | [Claims ledger](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) |
| **The leadership snapshot** (wins · live goal · risks · the one decision) | [Executive roll-up](EXECUTIVE-ROLLUP.md) |
| **How fak maps to what enterprises are buying** (runtime enforcement · prove-it · cost kill-switch · NHI · tamper-evident audit · air-gap) — every stat sourced, every claim fenced shipped/ticketed | [Enterprise positioning](enterprise-positioning.md) |
| **A machine-readable map (for LLMs)** | [llms.txt](https://github.com/anthony-chaudhary/fak/blob/main/llms.txt) |

## Additional documentation routes

- **Documentation maintenance:** [Indexed document sets](document-sets.md) defines the bounded-page and reciprocal-index contract for maintained long-form Markdown.

- **Session control and trajectory:** [Child-agent registration and lineage](session-child-registration.md), [Session lifecycle reconciliation](session-lifecycle-reconciliation.md), [Trajectory assurance receipt](trajectory-assurance.md), and [Workflow concepts: the operator's middle layer](workflow-concepts.md).
- **Local application runtime:** [Local-app compute layer](local-app-compute-layer.md) explains the browser-to-daemon boundary, loopback security, offline behavior, and accelerator ownership; [job-apply migration runbook](local-app-job-apply-runbook.md) gives the shortest supported signed-desktop-app integration path.
- **Tool-result and work accounting:** [Operate tool-result budgets safely](tool-result-budget-operations.md), [Work-accounting coverage](work-accounting-coverage.md), [Work delivery: recording is not readiness](work-delivery.md), [Work-done baselines](work-done-baselines.md), [Work-done history](work-done-history.md), [Work-done query contract](work-done-query.md), and [Work-done source provenance](work-done-sources.md).
- **Captured read-backs and witnesses:** [Current task queue read-back](witnesses/task-queue-current.md) and [Issue 9020 — owned Metal session profile](_witnesses/issue-9020-metal-profile.md).

---

<sub>License: Apache-2.0 · [Report a vulnerability](https://github.com/anthony-chaudhary/fak/blob/main/SECURITY.md) · Keywords: fak agent runtime, Fused
Agent Kernel, fak agent kernel, fak manage, fak serve, fak-certified, agent kernel,
AI agent runtime boundary, long-session prompt cache, model routing for agents, MCP
tool-call boundary, local GGUF, KV cache, addressable KV cache, self-hosted LLM,
LLM agent fleet, agentic AI, Go.</sub>




## Claude usage guides

- [Claude usage limits](claude-usage-limits.md)
- [Claude Code usage limits](claude-code-usage-limits.md)
- [Claude extra usage](claude-extra-usage.md)

<!-- BREADCRUMB-JSONLD:BEGIN (generated by tools/gen_structured_data.py — do not edit by hand) -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "https://anthony-chaudhary.github.io/"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "fak documentation",
      "item": "https://anthony-chaudhary.github.io/fak/"
    }
  ]
}
</script>
<!-- BREADCRUMB-JSONLD:END -->

- [KV capacity normalization](kv-capacity-normalization.md) — compare block-oriented and direct KV metrics in tokens, bytes, and occupancy without inventing unavailable values.

# Scoreboard debt discovery index

These links expose captured evidence and maintained scorecard/research clusters to the documentation crawler without changing their content or scorecard coverage.


# Scoreboard debt discovery index

These links expose captured evidence and maintained scorecard/research clusters to the documentation crawler without changing their content or scorecard coverage.

- [armbench-caveman-native](_witnesses/armbench-caveman-native/README.md)
- [armbench-caveman-passthrough](_witnesses/armbench-caveman-passthrough/README.md)
- [caveman-pairwise-judge-v2](_witnesses/caveman-pairwise-judge-v2/README.md)
- [issue-8308-qwen38-bf16](_witnesses/issue-8308-qwen38-bf16/README.md)
- [issue-8311-qwen38-q5km](_witnesses/issue-8311-qwen38-q5km/README.md)
- [issue-8360-qwen38-mac-metal](_witnesses/issue-8360-qwen38-mac-metal/README.md)
- [issue-8504-temp-artifacts](_witnesses/issue-8504-temp-artifacts/README.md)
- [issue-8544-open-witnessed-closure](_witnesses/issue-8544-open-witnessed-closure/README.md)
- [issue-8621-qwen35-0.8b](_witnesses/issue-8621-qwen35-0.8b/README.md)
- [issue-8622-qwen35-27b](_witnesses/issue-8622-qwen35-27b/README.md)
- [issue-8629-qwen35-0.8b-valid-smoke](_witnesses/issue-8629-qwen35-0.8b-valid-smoke/README.md)
- [issue-8630-qwen35-9b](_witnesses/issue-8630-qwen35-9b/README.md)
- [issue-8819-qwen38-a100-roofline](_witnesses/issue-8819-qwen38-a100-roofline/README.md)
- [issue-8968-qwen38-metal-control](_witnesses/issue-8968-qwen38-metal-control/README.md)
- [issue-9044-q8-metal-residency](_witnesses/issue-9044-q8-metal-residency/README.md)
- [issue-CHILD-qwen38-startup-bisect](_witnesses/issue-CHILD-qwen38-startup-bisect/README.md)
- [qwen38-27b-2026-08-19](_witnesses/qwen38-27b-2026-08-19/README.md)
- [lightgap-scorecard](lightgap-scorecard/README.md)
- [ceilings](lightgap-scorecard/ceilings.md)
- [dents](lightgap-scorecard/dents.md)
- [model](lightgap-scorecard/model.md)
- [segment-fleet-operator](lightgap-scorecard/segment-fleet-operator.md)
- [segment-framework-builder](lightgap-scorecard/segment-framework-builder.md)
- [segment-local-first](lightgap-scorecard/segment-local-first.md)
- [segment-platform-team](lightgap-scorecard/segment-platform-team.md)
- [segment-regulated](lightgap-scorecard/segment-regulated.md)
- [segment-researcher](lightgap-scorecard/segment-researcher.md)
- [segment-solo-max](lightgap-scorecard/segment-solo-max.md)
- [unrun](lightgap-scorecard/unrun.md)
- [contract](portability/contract/README.md)
- [inventory](research/inventory/README.md)
- [langchain-ai-open-swe](research/inventory/langchain-ai-open-swe.md)
- [obra-superpowers](research/inventory/obra-superpowers.md)
