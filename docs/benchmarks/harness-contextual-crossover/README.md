# Contextual harness crossover study

**Verdict:** the declared nine-task scenario is a negative result for the
contextual harness. The tuned Copilot-native alternative totals **833.5 s** of
modeled operator-equivalent work versus **1,161 s** for fak. Under the same
metric, fak crosses below that tuned baseline only after **29.04 domain
switches**. This is a model, not a productivity claim.

## Question and alternatives

The study asks whether automatic contextual selection reduces total operator
work across repeated coding → legal → integrated transitions after charging for
setup, maintenance, switching, explanation, context, and wrong-layer recovery.
It compares fak with tuned native capabilities documented by each host, not
factory defaults:

- **Codex:** user/project configuration and named profiles, plus global and
  repository `AGENTS.md` instruction hierarchy.
- **Claude Code:** managed, user, project, and local settings, plus managed,
  project, user, and local `CLAUDE.md` memory hierarchy.
- **GitHub Copilot:** repository-wide and path-specific custom instructions.
- **fak:** scoped contextual selection, verified effective-product locks, and
  risk-only preview.

The exact official URLs and retrieval date (`2026-08-15`) are embedded in
[`study.json`](study.json). Redirecting product documentation is cited through
its stable public URL rather than copied into this repository.

## Fixed metric

For alternative `a`, total operator-equivalent seconds are:

```text
setup + maintenance
+ switch_actions × 15 s
+ wrong_layer_incidents × 300 s
+ explanation_seconds
+ context_tokens × 0.002 s
```

The scenario contains nine tasks and eight domain switches. Every value in this
first study is labeled **modeled**. There are no witnessed timing samples or
observed wrong-layer incidents, so the result must not be promoted to an
empirical gain/loss claim. Wrong-layer incidents are zero for every alternative
rather than guessed in fak's favor.

The fixed setup/maintenance/explanation disadvantage for fak versus the best
native baseline is 452 s. Its declared per-switch advantage is 15.5625 s, which
produces the 29.04-switch crossover. If setup or maintenance rises, crossover
moves later; if native switching is automated, it may disappear. If measured
wrong-layer incidents differ, the predeclared 300 s recovery weight applies to
all alternatives unchanged.

## Reproduce

```sh
go run ./cmd/fak harness study crossover \
  --input docs/benchmarks/harness-contextual-crossover/study.json
```

The checked-in [`witness.json`](witness.json) is the exact output. The pure runner
uses `fak.harness-crossover-study/v1alpha1`, rejects missing coding/legal/
integrated coverage, requires documentation for every alternative, and accepts
only `witnessed`, `observed`, or `modeled` provenance. It sorts by the same total
whether fak wins or loses; the negative-result test pins that behavior.

## What this does and does not show

This clears the methodological spine: real tuned alternatives, all requested
cost categories, explicit provenance, a reproducible negative result, and a
crossover condition. It does **not** establish that the modeled values describe
real operators. The next empirical rung should replace individual values with
captured setup/switch/explanation measurements while retaining this schema and
metric; changing the metric after seeing results would invalidate comparison.

