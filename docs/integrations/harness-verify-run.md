---
title: "Verify what a harness actually ran"
description: "A verified lock proves what the harness promised before launch. fak harness verify-run compares that promise with a runtime observation,"
---
# Verify what a harness actually ran

A verified lock proves what the harness promised before launch. `fak harness verify-run` compares that promise with a runtime observation, so “trust but verify” continues after execution.

## Command

```bash
fak harness verify-run \
  --lock product.lock.json \
  --observation run-observation.json
```

## The runtime observation schema

The runtime observation uses schema `fak-harness-runtime-observation/1`:

```json
{
  "schema": "fak-harness-runtime-observation/1",
  "lock_id": "sha256:...",
  "run_id": "run-42",
  "capabilities": [
    {
      "capability": "instruction:response-style",
      "source": "operator-override",
      "value": "detailed"
    },
    {
      "capability": "tool:search_kb",
      "source": "repo:defaults"
    }
  ],
  "events": [
    {
      "kind": "route",
      "capability": "route:model",
      "source": "gateway",
      "outcome": "selected fast"
    },
    {
      "kind": "policy",
      "capability": "tool:search_kb",
      "source": "company:security",
      "outcome": "allowed"
    }
  ]
}
```

## Classification and exit behavior

The observation must name the exact verified lock ID. Every effective capability is classified as:

- `matched`: runtime value and provenance match the lock;
- `changed`: the capability exists, but source, value, reference, boundary, grants, or denials differ;
- `added`: runtime reported a capability absent from the lock; or
- `omitted`: the lock promised a capability that runtime did not report.

Route, approval, and policy events remain visible below the capability comparison with their runtime source and outcome. A deviation exits `3`, matching `harness preview`'s decision-required convention. Use `--json` for an admission gate or dashboard.

## Where the evidence comes from

This command consumes evidence; it does not invent runtime telemetry. Harness adapters should emit the observation from their existing trace or receipt seam. That keeps provider-specific event collection outside the lock comparator while preserving one portable verification contract.
