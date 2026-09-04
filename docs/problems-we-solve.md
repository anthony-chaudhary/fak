---
title: "The problems fak exists to solve"
description: "FAK's P1-P4 checklist for managed context, net-true efficiency, bounded adaptation, and integrated operations, plus its centrality model."
---

# The problems fak exists to solve

fak exists to make long-running agent work **cheaper, faster, safer, and more operable**.
Those outcomes meet at one kernel seam: every tool call crosses a checkpoint where fak can
preserve useful context, avoid unnecessary work, adapt execution, and enforce the operating
contract.

This page gives development work two different instruments. Do not collapse them into one score:

1. **The P1-P4 problem checklist** is a design and review checklist. Every change considers
   every row. The rows are not competing priorities and an author does not pick a favorite one.
2. **Problem centrality** is a portfolio signal. It says how directly a unit of work advances
   the connected problem cluster below. More-central work normally outranks less-related work,
   after hard obligations and dependencies are accounted for.

A P1 implementation still has to be net-true (P2), adaptable rather than brittle (P3), and
operable through the real system (P4). Conversely, release or maintenance work may directly
implement none of the four yet still be necessary stewardship. The checklist governs **how we
do the work**; centrality helps decide **which work to do next**.

## The connected problem cluster

The labels are stable handles for related user problems, not product silos. A real change can
advance several at once.

### P1 — Managed context

**Problem:** Agent sessions repeatedly reconstruct shared setup, lose useful state as histories
grow, and send context that could have been retained or served locally.

**Direction:** Preserve reusable prefixes, serve safe repeats locally, compact old turns
deliberately, and keep provider caches useful across long-running sessions.

**Witness:** less repeated input or setup work at equal task quality; longer useful sessions;
a cache, replay, or compaction effect read back from the real path.

### P2 — Net-true efficiency

**Problem:** An optimization can look cheaper in isolation while routing, verification, quality
loss, retries, or operator burden makes the real workflow worse.

**Direction:** Use the least expensive execution that still meets the contract, count all added
costs, and compare with the tuned next-best alternative rather than a naive baseline.

**Witness:** an end-to-end latency, cost, quality, or completed-work gain that survives the
[`net-true-value`](standards/net-true-value.md) accounting.

### P3 — Fast, bounded adaptation

**Problem:** Agents and workloads change faster than static infrastructure. Broad rewrites are
too slow; unbounded adaptation is unsafe and hard to trust.

**Direction:** Make routing, policy, model, and memory behavior changeable at small seams, with
explicit bounds, rollback, and evidence.

**Witness:** a new workload or policy supported through a small, reversible change with captured
before/after behavior.

### P4 — Integrated operations

**Problem:** A good component does not help if operators cannot discover, run, observe, recover,
and govern it through the actual agent/tool path.

**Direction:** Put performance, security, observability, recovery, and lifecycle control on the
same checkpoint rather than in disconnected side systems.

**Witness:** the real end-to-end path works and exposes enough state to diagnose, recover, and
enforce its contract.

## Cross-cutting outcome: portable intent

The four rows above also support a broader user outcome: **portable intent**. A person should be
able to state the behavior they want at a named product layer without first learning which
plugin, skill, prompt fragment, model option, or native implementation happens to provide it.
For example, a conversation preference should name the desired style and its constraints; it
should not require the user to know that one harness calls a possible implementation
“Caveman.” A team can then review and share the semantic declaration instead of persuading
every recipient to install the same mechanism.

Portable intent separates two identities:

- the **intent** is a versioned, typed value owned by its layer, such as a conversation-style
  preference with required fidelity and an authority ceiling;
- a **binding** is the replaceable, provenance-carrying mechanism that realizes that intent in
  one environment.

This is not a fifth P-number or a universal ontology. Each layer owns a bounded vocabulary and
composition rules while a common envelope carries scope, compatibility, provenance, resolution,
and receipts. The outcome connects all four checks: compact semantic declarations can reduce
repeated setup (P1); any gain must beat direct configuration net of resolver and verification
cost (P2); bindings make implementation churn bounded and reversible (P3); and the real path
must resolve, apply, explain, refuse semantic loss, and read back the effect (P4).

The roadmap is tracked in [#6877](https://github.com/anthony-chaudhary/fak/issues/6877). The
required first proof is [#6878](https://github.com/anthony-chaudhary/fak/issues/6878): one
implementation-independent conversation profile running through two different bindings, with
an unsupported required meaning refused before launch. Until that witness lands, portable
intent is a direction and contract, **not a shipped runtime claim**.

## The all-work problem checklist

Run this checklist at **framing, spine design, implementation, witness, and review**. “All must
pass” means every row receives an honest answer; it does not mean every change produces a new
feature for every P-number.

| Check | Ask on every unit of work | Pass condition |
|---|---|---|
| **P1 · Context** | What context or repeated work does this add, remove, preserve, or invalidate? | It improves managed context, or does not create avoidable repetition, cache damage, or context growth. |
| **P2 · Net value** | Is this better than the real alternative after implementation, runtime, verification, quality, retry, and operator costs? | A proportionate witness supports the gain, or it makes no gain claim and names its necessary obligation. |
| **P3 · Adaptation** | What is likely to change next, and is the seam bounded, reversible, and small enough to adapt? | It avoids gratuitous lock-in, states its bounds, and has a risk-appropriate rollback or safe failure path. |
| **P4 · Operations** | Can the real system discover, run, observe, secure, and recover this behavior? | The end-to-end seam and operating contract are witnessed; component proof is not used for a system claim. |

A row may be **advanced**, **preserved**, or **not applicable with a concrete reason**. “N/A” is
not a shortcut: a typo fix can state that it changes no runtime context, gain claim, adaptation
seam, or operating path; a broad feature cannot plausibly dismiss all four. A row **fails** when
the change creates an unmitigated regression, makes an unwitnessed claim, or supplies only a
label. Failed rows block normal landing. A time-critical obligation must name the accepted
tradeoff, owner, and follow-up rather than silently converting failure to N/A.

### Use it through the development loop

1. **Frame:** State `For / Problem / Today / Better because / Witness` in plain language.
2. **Classify centrality:** Assign one class and one sentence of evidence using the rubric below.
3. **Design the spine:** Choose the smallest real end-to-end path and record P1-P4 risks.
4. **Implement:** Keep context accounting, net cost, bounded change, and operations at the seam;
   do not bolt them on after the code is “done.”
5. **Witness:** Capture the working spine first, then its operating envelope and optimization.
6. **Review and close:** Re-run all four rows against the diff and evidence; file follow-ons
   rather than hiding a failed row in prose.

This complements the [spine-first defaults](spine-first-defaults.md), the
[Feynman-simple value frame](shift-left-task-organization.md), architectural leaf boundaries,
capability policy, and net-true evidence standard. Those are delivery disciplines; P1-P4 are
the persistent problem lens applied through them.

## Default priority hierarchy

Alongside problem centrality and the P1-P4 checklist, the repository prioritizes work across four explicit operating tiers:

1. **fak all in one (serving and harness + memory — the "one touch" thing):**
   The primary product focus: a single-binary turnkey deployment (`fak up`) uniting high-performance model serving, the agent harness with capability-floor security, and durable context memory into one frictionless developer experience.
2. **fak serving only:**
   High-performance model inference, disaggregated KV-cache routing, and OpenAI/MCP gateway endpoints (`fak serve`).
3. **fak harness only:**
   Standalone agent harness, capability floor enforcement, and tool-call interception wrapping external frontier models (`fak guard`).
4. **other things:**
   Peripheral utilities, benchmark harnesses, standalone tooling, and secondary integrations.

## Problem centrality: the portfolio signal

Centrality asks:

> If this work succeeds, how directly does it relieve the connected user problems above or
> unblock evidence that they are relieved?

Use qualitative classes; do not manufacture numeric precision the evidence cannot support.

| Class | Meaning | Evidence and default treatment |
|---|---|---|
| **Core** | Directly changes a user-visible P1-P4 outcome on fak's kernel path. | Witness the end-to-end effect; prefer it when evidence, urgency, and readiness are comparable. |
| **Enabling** | Removes a concrete blocker or supplies required measurement for named Core work. | Name the blocked outcome; sequence with it and reclassify if that outcome closes or changes. |
| **Stewardship** | Maintains reliability, security, compatibility, release health, or developer throughput without directly moving a P1-P4 outcome. | Name the obligation or risk; schedule by risk, deadline, and recurring cost. |
| **Peripheral** | Has no evidenced path to the problem cluster, enabling dependency, or current obligation. | Defer, reshape, or decline unless an explicit external obligation overrides. |

Centrality is **not the whole priority decision**. First honor security incidents, data-loss
risks, broken trunk/release obligations, and contractual deadlines. Then satisfy dependencies
and compare ready work by centrality, user evidence, expected net value, urgency, effort, and
reversibility. Centrality is the directional tie-breaker that prevents an attractive but
unrelated backlog from displacing fak's reason to exist.

Re-evaluate centrality when scope, dependencies, or the user outcome changes; never inherit it
mechanically from a parent epic. Classify the **effect**, not its directory, label, or mechanism.
For example, observability is Peripheral as a speculative dashboard, Enabling when it measures
a named cache spine, and Core when diagnosis/recovery is itself the broken P4 path.

### Examples

- **Core:** preserve a provider-cache-compatible prefix across compaction and capture reduced
  repeated input on the real path. It still answers P2, P3, and P4.
- **Enabling:** add counters required to decide whether that named compaction spine saves work.
  Generic telemetry without a named outcome is not enough.
- **Stewardship:** update a supported Go version after a security deadline or repair a flaky
  release gate. Urgent stewardship can outrank Core work without a fictional product claim.
- **Peripheral:** add an unrelated management surface with no kernel-path user, dependency, or
  obligation. Tie it to witnessed P1-P4 pain or spend capacity on more central work.

## Required issue and plan frame

```text
For: <specific user/operator>
Problem: <observable pain today>
Today: <real next-best alternative>
Better because: <expected outcome, net of added cost>
Witness: <artifact or read-back that can prove it>
Centrality: Core | Enabling(<named Core outcome>) | Stewardship(<obligation>) | Peripheral
P1 Context: advanced | preserved | N/A — <reason>
P2 Net value: advanced | preserved | N/A — <reason>
P3 Adaptation: advanced | preserved | N/A — <reason>
P4 Operations: advanced | preserved | N/A — <reason>
```

Keep this proportionate. The block replaces the old “choose one primary P-ID” convention.
Multiple rows may be advanced because the problems are a cluster, not competing buckets.

## What this model refuses

- Picking P2 and ignoring P4: a benchmark-only speedup is not an integrated outcome.
- Four ceremonial checkmarks: identify an effect, preserved invariant, or concrete reason.
- Calling every prerequisite Core: enabling work names the outcome it actually unblocks.
- Using centrality to skip maintenance: urgent stewardship can outrank product work.
- Using priority to excuse bad design: urgent work still passes a proportionate checklist.
- Mistaking a leaf for a user problem: centrality follows the witnessed effect.

## One-sentence test

> We apply context, net-value, adaptation, and operations checks to every change; when choosing
> among changes, we prefer the strongest evidenced path to fak's connected user problems,
> subject to real obligations and risk.
