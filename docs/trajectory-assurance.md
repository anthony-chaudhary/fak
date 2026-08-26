---
title: "Trajectory assurance receipt"
description: "Trajectory assurance emits a deterministic, JSON-ready health receipt. Version 1 is shadow-only: every receipt has \"shadow\":true,"
---
# Trajectory assurance receipt

Trajectory assurance emits a deterministic, JSON-ready health receipt. Version 1 is shadow-only: every receipt has `"shadow":true`, and the package exposes no stop, kill, mutation, or action callback. Consumers may display or store the receipt, but must not treat it as control authority.

The receipt contains five ordered layers: `deterministic_floor`, `objective_progress`, `efficiency_with_quality`, `delegation_integrity`, and `semantic_review`. Every layer reports `state`, `source`, `provenance`, `authority`, `freshness`, and `reason`. The four states are `PASS`, `WARN`, `FAIL`, and `UNKNOWN`. Missing required evidence is `UNKNOWN`, never an inferred pass.

Aggregation uses a non-averaging partial order. Any `FAIL` produces overall `FAIL`; otherwise any `UNKNOWN` produces `UNKNOWN`; otherwise any `WARN` produces `WARN`; only all-pass evidence produces `PASS`. Scores are never averaged. In particular, semantic judgment cannot override a deterministic failure. Such disagreement is retained in `conflicts` rather than resolved in favor of semantics.

Efficiency is quality-constrained, not a raw usage score. It requires an observed outcome, observed constraint satisfaction, and complete accounting for both the parent and all children. The receipt emits parent, child, and total units. Missing any required part makes this layer `UNKNOWN`; failed outcome or constraints makes it `FAIL`.

`missing_evidence` names unknown layers in lexical order. `conflicts` has deterministic ordering. Every receipt contains exactly one recommendation string, phrased as advice and explicitly preserving shadow behavior.

The schema intentionally has no raw prompt, raw tool payload, tool input, transcript body, or arbitrary metadata fields. The decoder rejects unknown fields, preventing accidental serialization of sensitive payloads. Inputs should contain stable identifiers and typed summaries only.

The package API is `trajectoryassurance.Assess` followed by `trajectoryassurance.Marshal`. The registered `fak trajectory assurance` command accepts one strict JSON input and writes one compact JSON receipt. It can also strictly adapt the public `fak.ultracode_status.v1` schema with `--ultracode-status FILE`; direct stdin behavior remains unchanged.

The v1 ultracode producer currently emits `unverified` with `not_observed` effect-readback, witness, and reconciliation fields, and it can emit explicit invalid/budget/activation failures. The adapter therefore reads identity and failure evidence but deliberately returns `UNKNOWN` rather than accepting speculative positive aliases. Issue #8834 tracks the missing authoritative positive terminal contract and the completed-worker activation contradiction.

## Command

```bash
fak trajectory assurance < typed-evidence.json
```

The command rejects unknown JSON fields, including raw prompt or tool-payload fields, and writes exactly one versioned JSON receipt. The top-level receipt carries the objective ID, trajectory ID, and observation window. Every layer carries a typed reason token in addition to bounded explanatory text. The command remains shadow-only and cannot stop, kill, or mutate a session.