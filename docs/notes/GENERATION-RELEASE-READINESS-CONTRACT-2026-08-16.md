---
title: "Generation-aware release staleness and readiness contract"
description: "Additive report contract that separates current-release debt from growth in later generation horizons."
---

# Generation-aware release staleness and readiness contract

This note closes #1649. It defines the agent-runnable readout to add to both
`fak release-staleness --json` and `fak release status --json` so an operator can
tell whether apparent lag is current-release debt or portfolio growth. The
existing release verdict and thresholds remain authoritative; this is an
additive attribution seam, not a new release gate.

## Report contract

Each report adds a `generation` object:

```json
{
  "generation": {
    "schema": "fak-release-generation/1",
    "distribution": {
      "gen/now": 12,
      "gen/next": 7,
      "gen/second-next": 3,
      "gen/future": 5,
      "unclassified": 1
    },
    "current_release": {
      "relevant": 12,
      "not_current": 15,
      "unclassified": 1,
      "basis": "gen/now"
    },
    "source": "open_issues"
  }
}
```

The producer counts each open issue once from its generation label. Exactly one
of `gen/now`, `gen/next`, `gen/second-next`, or `gen/future` contributes to
`distribution`; no generation label contributes to `unclassified`. Multiple
generation labels are invalid input and must be reported as unclassified rather
than guessed. `current_release.relevant` equals `distribution["gen/now"]`;
`not_current` is the sum of the other three horizons. Text output renders one
compact line:

```text
generation: current release 12 · later horizons 15 · unclassified 1 (now 12, next 7, second-next 3, future 5)
```

A release operator reads `current_release.relevant` alongside the existing
staleness verdict. A large `gen/now` count is candidate current-release debt; a
large later-horizon count is portfolio growth and does not make the current
release stale. Counts describe horizon, not value or completion, so the report
must not subtract them from commit lag or change the release decision.

## Independence rules

Generation remains orthogonal to three other controls:

- **Priority:** generation says *when the product horizon expects the work*;
  priority still ranks value and urgency within or across horizons.
  `gen/future` is never treated as lower priority by default.
- **Shared trunk:** all generations continue to use `main`, lane leases, and
  path-scoped commits. The report must not infer or create a branch per horizon.
- **Runtime feature gates:** generation labels do not enable code. Existing
  runtime gates continue to decide exposure; a `gen/next` feature remains gated
  until its own safety and dogfood evidence supports changing that gate.

## Continuation and evidence

The smallest implementation slice is one shared fold that accepts issue labels
and returns `fak-release-generation/1`, then embeds that object in both release
JSON reports and renders the exact text line above. Its contract test should
cover one issue in each horizon, one unlabeled issue, and one multiply labeled
issue; it should also prove that changing the generation mix does not change the
existing release-staleness verdict. A captured invocation of both commands is
the operator witness.

**Promotion evidence toward `gen/now`:** both commands expose the same tested
fold, a dogfood run reads live issue labels without changing the old verdict,
and an operator uses the split during a release review.

**Demotion or retirement evidence:** demote the surface to `gen/next`, or remove
the additive object in a schema-versioned change, if live label coverage is too
low to support release decisions, conflicting labels are common, or operators
do not use the split in release review. Priority labels and runtime gates remain
unchanged when doing so.

**Invalidating assumption:** this contract assumes `gen/now` is the sole horizon
relevant to the current release and that open GitHub issue labels are the
portfolio source of truth. If releases intentionally include another horizon,
or dispatch moves to a different authoritative inventory, `basis` and `source`
must become configurable before the counts can guide readiness.
