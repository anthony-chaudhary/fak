---
title: "Astra formal subagents: bounded admission and operator contract"
description: "Current contract for routing complete, read-only formal packets to GPT-6 Astra without claiming measured quality superiority."
---

# Astra formal subagents

This is the operator contract for [#12090](https://github.com/anthony-chaudhary/fak/issues/12090). Treat the implementation as shipped only when its resolving commit is reachable from the repository's authoritative branch. The contract controls where Astra is used; it does not establish that Astra produces better results than another model. That requires a matched, accepted-outcome comparison that has not yet been reported here.

## The admission rule

Automatic Astra routing is for a complete, read-only formal packet with `work_class` set to `rigor`. The packet may contain only these four closed task kinds:

- `formal_proof`
- `invariant_audit`
- `state_machine_audit`
- `numerical_correctness_derivation`

Every packet must state its definitions and assumptions, exact proposition, required output form, deterministic witness, and one to three distinct bounded repository surfaces. The schema identity must be exactly `fak-formal-packet/1`.

Astra is not automatically selected for exploration, implementation, test execution, general review, issue triage, or documentation. Difficulty, model-related keywords, labels, and ordinary task prose do not confer eligibility. A declared effect-mode worker also makes the packet ineligible: the automatic formal route is an analysis-only route, not write authority.

## Plan a formal packet

Save a task such as this as `formal-task.json`:

```json
{
  "schema": "fak-orchestration-task/1",
  "id": "lease-holder-invariant",
  "work_class": "rigor",
  "formal_packet": {
    "schema": "fak-formal-packet/1",
    "task_kinds": [
      "invariant_audit"
    ],
    "definitions_and_assumptions": "Every positive lease generation denotes one acquired holder epoch; generation zero is the only anonymous legacy state.",
    "exact_proposition": "Prove that every accepted positive-generation fence token has matching non-empty presented and current holder identities, or give the smallest counterexample.",
    "required_output_form": "DEFINITIONS, INVARIANT, CASE TABLE, PROOF OR COUNTEREXAMPLE, MINIMAL FIX OBLIGATION, WITNESS",
    "deterministic_witness": "go test ./internal/leaseref -run TestFenceRefusesMissingHolderIdentity -count=1",
    "surfaces": [
      "internal/leaseref/**"
    ]
  }
}
```

Resolve it without launching work:

```powershell
fak orchestration plan --profile auto --task formal-task.json --json --selfcheck
```

The command emits stable plan JSON and `SELFCHECK PASS ... launched=0`. Inspect `resolved.astra_route`, `resolved.sol_route`, and `overrides` before launching. `--selfcheck` proves schema round-trip and route resolution only; it is not a model-quality or completed-task witness.

## Read the receipt

`resolved.astra_route.eligible` answers whether the packet satisfies the automatic formal-admission contract. `selected` answers whether the final executable child route uses an Astra-family model after profile and pin precedence. They intentionally differ in several cases:

| Situation | `eligible` | `selected` | Meaning |
|---|---:|---:|---|
| Complete packet, `auto`, no pin | `true` | `true` | Automatic worker route is `gpt-6-astra` at `xhigh`. |
| Complete packet, `off` or another direct/no-child resolution | `true` | `false` | The packet is valid, but there is no child route to select. |
| Complete packet, non-Astra environment, task, or CLI worker-model control | `true` | `false` | The higher-precedence model control wins over automatic selection. |
| Complete packet, Astra environment, task, or CLI worker-model control | `true` | `true` when a supported child route exists | The effective model is Astra, but the receipt source identifies the winning control rather than automatic admission. |
| Incomplete, excluded-kind, unbounded, or effect-mode packet | `false` | `false` unless explicitly pinned | The formal packet cannot automatically spend Astra capacity. A pin is explicit operator control outside the automatic-admission claim. |

Model and effort precedence is:

1. CLI `--worker-model` and `--worker-effort` controls;
2. task `pins.model` and `pins.effort` controls;
3. `FAK_ORCHESTRATION_WORKER_MODEL` and `FAK_ORCHESTRATION_WORKER_EFFORT` environment controls;
4. the eligible formal-packet default, `gpt-6-astra` and `xhigh`;
5. the ordinary orchestration default.

Environment controls are resolved before the stable plan and launch receipt. They appear with source `environment`, never silently override a task or CLI pin, and an invalid `FAK_ORCHESTRATION_WORKER_EFFORT` value fails before launch. `astra_route.source` identifies the effective model source (`formal-packet`, `environment`, `task.pin`, or `operator-pin`). `reasoning_effort_source` tracks effort independently. The matching `overrides` fields are `sol_route.worker_model` and `sol_route.worker_reasoning_effort`; a formal task pin must not masquerade as a `fast.*` decision.

Formal admission never grants write access. Default and explicitly declared observe workers remain read-only. Declaring any effect-mode `worker_access` adds `ASTRA_FORMAL_PACKET_ANALYSIS_ONLY` and prevents automatic Astra admission. Access compilation and enforcement remain separate controls even when an operator explicitly pins a model.

## Fail-closed refusal classes

A declared but invalid packet remains visible in `astra_route` with `eligible=false`, `selected=false` absent an explicit model pin, and ordered reason codes:

| Reason | Refusal |
|---|---|
| `ASTRA_FORMAL_PACKET_SCHEMA_INVALID` | Schema is missing or not the exact supported identity. |
| `ASTRA_FORMAL_PACKET_WORK_CLASS_REQUIRED` | Task is not `rigor`. |
| `ASTRA_FORMAL_PACKET_ANALYSIS_ONLY` | At least one declared worker requests effect mode. |
| `ASTRA_FORMAL_PACKET_TASK_KIND_REQUIRED` | No non-empty task kind is present. |
| `ASTRA_FORMAL_PACKET_TASK_KIND_DUPLICATE` | Canonically equivalent kinds repeat. |
| `ASTRA_FORMAL_PACKET_TASK_KIND_EXCLUDED` | A known non-formal kind is mixed into the packet. |
| `ASTRA_FORMAL_PACKET_TASK_KIND_UNKNOWN` | A kind is outside the closed vocabulary. |
| `ASTRA_FORMAL_PACKET_DEFINITIONS_AND_ASSUMPTIONS_REQUIRED` | Definitions/assumptions are blank or placeholder text. |
| `ASTRA_FORMAL_PACKET_EXACT_PROPOSITION_REQUIRED` | The proposition is blank or placeholder text. |
| `ASTRA_FORMAL_PACKET_REQUIRED_OUTPUT_FORM_REQUIRED` | The output contract is blank or placeholder text. |
| `ASTRA_FORMAL_PACKET_DETERMINISTIC_WITNESS_REQUIRED` | The witness is blank or placeholder text. |
| `ASTRA_FORMAL_PACKET_SURFACE_COUNT_INVALID` | Surface count is outside one through three. |
| `ASTRA_FORMAL_PACKET_SURFACE_UNBOUNDED` | A surface is rooted, drive-qualified, parent-escaping, wildcard-ambiguous, or otherwise unbounded. |
| `ASTRA_FORMAL_PACKET_SURFACE_DUPLICATE` | Two surfaces canonicalize to the same bounded region. |

Fix every returned reason before relying on automatic admission. Do not work around a refusal by embedding formal-sounding prose in `--task-text`; task text cannot invent a formal packet.

## Launch assignment binding

On launch, the complete packet is lowered into the worker's exact read-only assignment. The launch receipt and every worker row are expected to carry the same non-empty `assignment_digest`:

```text
sha256:<64 lowercase hexadecimal digits>
```

The digest is SHA-256 over the exact trimmed assignment text sent to workers. A different digest means a different assignment and must not be reconciled as the planned formal task. The digest proves launch-time binding to bytes, not that a worker understood the proposition, completed the witness, or returned a correct proof. Broader stale-worker detection and result reconciliation remain tracked by [#8855](https://github.com/anthony-chaudhary/fak/issues/8855).

## Ranked weakest-link deployment map

These are the current highest-value places to spend a bounded Astra read-only audit. Ranking reflects potential correctness impact and fit to formal reasoning, not measured Astra superiority. Each issue assigns implementation and test execution to a separate worker after the formal obligation is settled.

1. [#12109 — lease fence holder identity](https://github.com/anthony-chaudhary/fak/issues/12109): prove the positive-generation holder invariant and isolate the empty-identity counterexample. This is the closest boundary to stale-writer exclusion.
2. [#12112 — non-finite model-routing telemetry](https://github.com/anthony-chaudhary/fak/issues/12112): derive the minimal validation frontier preventing NaN or infinity from becoming feasible, improved, or unserializable routing evidence.
3. [#12110 — goal reopen state machine](https://github.com/anthony-chaudhary/fak/issues/12110): derive the exhaustive terminal-to-active transition table and the mutation-free refusal cases.
4. [#12111 — goal binding equivalence](https://github.com/anthony-chaudhary/fak/issues/12111): prove one canonical equivalence relation across bind, resolve, and unbind.

Do not open a duplicate for the `PublishFenced` check-then-publish takeover race. [#11835](https://github.com/anthony-chaudhary/fak/issues/11835) already owns authoritative acquisition and fencing at the actual write boundary, including late stale-holder refusal. #12109 is narrower: it covers missing holder identity at equal positive generation.

## Benefit and harm standard

| Axis | Operating standard |
|---|---|
| Indication | A closed formal kind, fixed assumptions, exact proposition, bounded surfaces, specified output, and deterministic witness. |
| Comparator/manual pinning | Compare with the existing manually pinned model on the same packet and acceptance witness; manual pinning remains the control path. |
| Expected benefit | Hypothesis: clearer proof obligations and fewer missed state, identity, or numerical counterexamples. No measured benefit is claimed by this page. |
| Harms/cost | Higher model cost and latency, duplicated reasoning, false confidence in polished proofs, and opportunity cost if used on mechanical work. |
| Uncertainty | Repository-specific quality and cost deltas are not yet established by a matched accepted-outcome study. |
| Contraindications | Exploration, coding, broad review, documentation, issue triage, test execution, effectful access, vague propositions, or unbounded repository scope. |
| Safeguards/dose | One focused read-only Astra audit at `xhigh`, one to three surfaces, compact required output, deterministic witness, then separate implementation and verification. |
| Control | `--profile off`, environment defaults, task pins, and CLI worker pins control routing in the documented precedence; receipts must preserve the winning source. |
| Surveillance | Inspect eligibility/reasons, selected model/effort, access mode, assignment digest, witness result, token/cost receipt, and follow-on defect rate. Re-rank only from comparable evidence. |

The deterministic package witness for the route contract is:

```powershell
go test ./internal/orchestration/...
```

Runtime behavior and tests are the final authority when this page and the executable disagree.
