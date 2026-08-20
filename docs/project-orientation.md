---
title: "Project orientation: the agent-kernel center"
description: "The active orientation for fak: preserve managed context, choose capable low-cost execution, enforce authority, and keep agent turns operable."
---

# Project orientation: the agent-kernel center

**Decision date:** 2026-08-15

**Status:** active orientation spine for [#6899](https://github.com/anthony-chaudhary/fak/issues/6899)

## Verdict

fak's center is the **kernel-mediated agent turn**:

> preserve useful managed context and cache-compatible shared work, choose the least-cost capable execution path, enforce a fail-closed capability floor at the same tool boundary, and keep that path operable through real agent harnesses.

This is one connected outcome, not four competing products. Managed context (P1), net-true efficiency (P2), bounded adaptation (P3), and integrated operations (P4) are checks on every change. Problem centrality says how directly a change relieves that outcome; it does not replace urgency, dependency order, risk, or stewardship obligations.

## Core is temporal, but identity is not fashion

“Core” must not answer four different questions at once:

1. **Enduring promise (constitutional, multi-cycle):** turn agent intent into economical, bounded, portable, and observable execution over long-running work.
2. **Owned seam:** mediate between agent intent and model/tool execution, including the state needed to preserve that contract across turns and harnesses.
3. **Current strategic emphasis (roughly 2–4 quarters):** invest most where the highest evidenced, controllable bottleneck limits that promise now.
4. **Change centrality:** classify a proposed effect as Core, Enabling, Stewardship, or Peripheral relative to the promise and current evidence.

A mechanism can move between headline wedge, active bottleneck, table stakes, reference implementation, optional substrate, and retired machinery without changing the enduring promise. De-emphasis never licenses regression of the contract the mechanism established.

### Why performance is emphasized now—and how it can recede

Performance means avoided total work, turns, latency, and cost **after** fak’s overhead, not a permanent allegiance to a particular cache implementation. It is a buying wedge now because repeated model setup, prompt replay, avoidable turns, and context reconstruction remain expensive and controllable. Cache stability and managed context are therefore strategic Core work today.

Performance emphasis should fall when representative tuned provider-native paths make fak’s marginal savings immaterial across two review cycles and no recovery or control advantage depends on the mechanism. At that point, “do not cause avoidable repeated work; report gains only when net-true” remains a constitutional invariant, while specialized cache machinery becomes Enabling, optional, or retired. If inference becomes nearly free but human attention, reliability, or effect risk dominates, investment moves to those bottlenecks.

### Why harness ownership is emphasized now—and how it can recede

A kernel that cannot preserve its semantics through a harness is not an integrated product. Harness and binding work is strategically central now because external seams are fragmented: they often lose context, recovery, capability, or operator-control semantics. fak needs at least one owned end-to-end path to prove the full contract and enough bindings to prove portability.

The first-party harness itself is not constitutionally Core. It can become a reference implementation when two independent external bindings pass the same portable-intent, capability, session, recovery, and operator-control witnesses without harness-specific behavior. Binding conformance remains central; UI breadth and harness-specific convenience then become Enabling or Peripheral.

### Transition discipline

- **Increase emphasis** only from a measured user-path bottleneck, required obligation, or evidence gap blocking a named decision.
- **Decrease emphasis** only when a tuned alternative satisfies the retained contract over representative paths—not because a mechanism became unfashionable.
- **Promote an option** when a named user path cannot meet privacy, availability, economics, or control requirements through the next-best substrate.
- **Demote or retire machinery** when usage and avoided cost no longer exceed maintenance, integration, and cognitive load.
- **Review before stale:** the canonical snapshot has an explicit review date; staleness requests judgment but never auto-rewrites strategy.

The machine-readable authority is `internal/orientation/orientation.json`; inspect it with `fak orientation` or `fak orientation --json`. It records current role, horizon, evidence state, retained contract, and increase/decrease triggers for each capability family.

The original fail-closed policy seam remains central, but it is not the whole product. Concurrent cache reuse exposed the larger user problem: agents repeatedly pay to reconstruct context and execution setup. Routing, compaction, repeat serving, recovery, and policy belong together when they improve one real mediated turn. Project automation does not become product Core merely because fak's maintainers use it heavily.

## Capability portfolio

| Capability family | Current centrality | Why | Required witness |
|---|---|---|---|
| Tool-call mediation, capability policy, adjudication | **Core** | The kernel must handle each effect before execution and fail closed without model persuasion. | Structural deny and non-blanket allow on the real preflight path. |
| Cache-stable traffic, shared-prefix/KV reuse, local repeat serving | **Core** | Avoiding repeated shared work is the principal performance outcome. | Net-true repeated-input, latency, or compute reduction against the tuned real alternative. |
| Managed context, compaction, recall, crash/session recovery | **Core** | Long-running agents need useful state preserved without repeatedly rebuilding the prompt. | Task continuity plus provider-cache-compatible prefix or measured reconstruction reduction. |
| Model routing and bounded adaptation | **Core when on the mediated turn** | Choosing a capable cheaper path is part of net-true execution; generic model experimentation is not. | Correctness non-inferiority and end-to-end cost/latency on the real call path. |
| Agent/harness integration and portable intent | **Core at the binding seam** | A kernel unused by a real harness is not an integrated outcome. Harness-specific convenience beyond the binding is Enabling or Peripheral. | One intent/profile runs through real bindings without capability widening or semantic drift. |
| Gateway, local inference, compute backends | **Enabling(the kernel-mediated turn)** | They provide execution substrates; backend breadth is not an end in itself. | A named Core path gains availability, cost, or control with full added operating cost stated. |
| Benchmarks, evals, claims, observability | **Enabling(a named Core claim)** | Evidence selects and verifies central product work. Generic measurement inventory is insufficient. | A decision or shipped claim changes because of the captured evidence. |
| Fleet, dispatch, shared-tree, release and CI machinery | **Stewardship(reliable development and release)** by default | These are real obligations for this repository, not automatic external product value. A leaf may be Enabling when tied to a named product witness. | Reduced delivery/recovery cost or a required release/build invariant. |
| Issue gardening, scoring, agent-process automation | **Stewardship(bounded portfolio operation)** by default | The 1,600+ issue portfolio needs control, but self-management must not consume the product it serves. | A bounded decision or retired queue cost, including the automation's own upkeep. |
| Unrelated management surfaces or research without a kernel-path user | **Peripheral** | Nearby usefulness does not establish fak product centrality. | Promotion requires a named user, real alternative, P1-P4 effect, and captured kernel-path witness. |

Classification attaches to a **change's witnessed effect**, not permanently to a directory. A gateway fix can be Core when the mediated turn is broken, Enabling when it adds a substrate, or Stewardship when it fulfills a compatibility obligation.

## Investment boundary

1. Prefer the smallest end-to-end improvement to the mediated turn over subsystem breadth.
2. An Enabling item must name the Core outcome it unblocks and the decision/witness that consumes it.
3. Stewardship stays explicit and can outrank Core work when deadlines, security, reliability, or repository integrity require it.
4. Peripheral work is visible, not disparaged; it receives capacity only after central outcomes and obligations, or after new evidence promotes it.
5. No benchmark-only speedup is a shipped gain without net-true end-to-end evidence.
6. No dogfood, fleet, scoring, or process verb becomes Core solely because it is implemented inside fak.
7. New breadth must either strengthen the real kernel path in the same spine or be filed as a separate, classified follow-on.

## Current evidence and uncertainty

The repository is 54 days old at this decision, with 11,629 commits, 646 `internal/` packages, 1,881 Go files under `cmd/fak`, and 1,631 open issues observed on 2026-08-15. The live centrality audit is the reproducible portfolio witness; raw counts are context, not proof that a capability is unnecessary.

This decision does **not** yet prove which individual legacy issues should close. Most predate the doctrine, so `unknown` is the honest state until sampled or migrated. The canonical 2026-08-15 audit found 5/1,631 explicitly classified, 1,626 unknown, and zero complete canonical P1-P4 frames; these live counts will change and should be regenerated rather than copied as current truth. The immediate operating goal is coverage and decision quality, not ceremonial mass-editing.

## Repeat the audit

```bash
fak orientation
fak orientation --json
fak centrality-audit --repo anthony-chaudhary/fak
fak centrality-audit --repo anthony-chaudhary/fak --json
fak centrality-audit --repo anthony-chaudhary/fak --sample > active-sample-ledger.json
fak centrality-audit --input open-issues.json --selections selected-migrations.json > migration-preview.json
```

The command reports explicit Core, Enabling, Stewardship, Peripheral, unknown, and complete P1-P4 counts. Each issue is typed `valid`, `invalid`, or `unclassified` with canonical reason and repair tokens. It does not score or reorder issues. `--sample` deterministically includes every open `priority/P0`, every open `gen/now`, and the most recently updated not-already-selected milestone-less issue from each declared capability-family label (lowest issue number breaks timestamp ties). The ledger records selection reasons and a conservative `retain`, `reframe`, or `unknown-with-missing-evidence` decision; labels choose strata but never infer centrality. `--selections` remains no-write: it accepts only explicitly numbered, evidence-backed classifications and emits every original and replacement body. It refuses blank evidence, duplicates, issues outside the audited input, invalid frames, and issues that already declare a frame; there is no blanket rewrite or metadata inference. Applying a reviewed patch and independently reading it back from GitHub remain separate explicit operator actions. [#6543](https://github.com/anthony-chaudhary/fak/issues/6543) owns canonical intake parsing; [#6544](https://github.com/anthony-chaudhary/fak/issues/6544) owns selection-surface propagation.

## Next bounded wave

1. Land this decision and the repeatable coverage audit.
2. Complete #6543 so new contracts cannot silently omit the frame.
3. Complete #6544 so centrality is visible beside urgency, readiness, dependency, and obligation.
4. Sample active `gen/now` and milestone-less work by capability family; retain, reframe, merge, defer, or close with evidence.
5. Revisit this record after that sample and after the next measured kernel-path spine; change classifications when evidence changes.
