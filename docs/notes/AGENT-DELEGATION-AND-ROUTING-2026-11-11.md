# Agent delegation and routing: model selection policy (2026-11-11)

Canonical leaf for model-scoped routing detail referenced from `AGENTS.md`
(router-shape gate, Ref #1798): scoped model-routing rules live here, not in
the router itself.

## Astra worker reservation

Reserve `gpt-6-astra` subagents for highly pre-specified formal-mathematical
leaves only. A qualifying packet states:

- fixed definitions and assumptions,
- the exact proposition or derivation,
- the required output form,
- a deterministic witness.

Send exploration, implementation, testing, review, and general reasoning to
another suitable worker model. The reservation keeps highest-capacity model
spend on leaves where its precision pays for itself; everything else routes
to the default worker tier.
