---
title: "Micro-context S8a adaptive filter selector"
description: "Fixture-backed selector that routes 1,000 issue records across exact exclusion, semantic filtering, group widening, and escalation."
status: observed
last_reviewed: 2026-08-09
---

# S8a: adaptive micro-window filter selection

## Verdict

**Observed in the fixture:** micro-windows can serve as a bounded control plane as well as a
data plane. A versioned selector receives one issue record and may emit only one allowlisted
stage: `exclude`, `run(auth-relevance)`, `widen(issue-neighborhood)`, or `escalate`.
The controller—not selector prose—executes that typed declaration.

Captured artifact:
[`s8a-local-filter-selector-1000-pass-2026-08-09.json`](../../experiments/microcontext/s8a-local-filter-selector-1000-pass-2026-08-09.json).

This is a fixture-backed experiment. Its cost units model relative work (`selector=1`, semantic
filter `=10`); they are not dollars, tokens, or live-model latency.

## Reproduce

```bash
go run ./cmd/microcontextdemo \
  -filter-selector-selfcheck -workers 16 \
  -filter-selector-output /tmp/filter-selector.json

go run ./cmd/microcontextdemo -verify-filter-selector /tmp/filter-selector.json
```

## Witness

- 100 easy documentation records use exact field-level exclusion and remain model-free.
- 900 residual records receive selector micro-windows through 16 bounded workers.
- Routing confusion matrix: 972 correct exclusions, 26 correct semantic-filter routes, one
  correct group-widen route, one correct escalation, zero wrong decisions.
- The relation-dependent record widens to an issue neighborhood instead of guessing from its
  isolated record; the fixture error escalates instead of becoming a negative.
- Adaptive modeled work is 1,160 units versus 9,000 for deterministic-prefilter-plus-semantic
  residual processing and 10,000 for always-run, at the same zero-false-negative fixture target.
- An unchanged replay makes zero selector calls and reuses all 900 selector decisions.
- An undeclared stage such as an arbitrary shell request is denied by the catalog contract.

## Interpretation

This proves the missing composition from the general operator note: a micro-window can decide
**which filter should run**, whether context must widen, or whether work should escalate. It does
not acquire tool authority and it does not hide uncertainty. Real endpoint economics and
quality remain for the tuned-baseline program (#6033); read-only tool stages remain #6031.
