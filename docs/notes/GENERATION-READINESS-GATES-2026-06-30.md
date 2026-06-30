# Generation Readiness Gates

This note closes #1644. It defines the readiness checks a `gen/next` feature must
pass before it can be treated as `gen/now`. Code existing on disk is not enough.

Generation stays orthogonal to priority, shared trunk, and runtime feature gates:
priority still ranks urgency, shared trunk still requires path-scoped commits on
`main`, and runtime gates still decide exposure. These checks only decide whether
the generation evidence is strong enough to move the work closer to `now`.

## Promotion Gate

A `gen/next` item is ready to promote toward `gen/now` only when all six gates are
green or explicitly waived in the issue with evidence:

| Gate | Green Condition | Refusal / Hold |
|---|---|---|
| Integration surface | The artifact has a real caller or operator entrypoint: CLI verb, report field, handoff schema, dispatch path, index query, dashboard, or documented workflow. | Hold if the code exists only as an uncalled helper, prototype, or private note. |
| Runtime exposure | The artifact is either safe by default or isolated behind an explicit feature/runtime gate with a named owner. | Hold if `gen/next` is being used as a substitute for a default-off runtime gate. |
| Witness | A focused test, captured command, before/after readout, or committed planning artifact proves the generation-specific claim. | Hold if the only evidence is a self-report, stale screenshot, or broad green suite unrelated to the claim. |
| Docs / self-index | A future agent can find the surface through `fak index`, `INDEX.md`, `llms.txt`, a linked note, or a local command help string. | Hold if using it requires rereading the parent epic or guessing the intended horizon. |
| Operator visibility | The operator can see status, debt, readiness, or next action in the relevant report, issue comment, milestone view, or ledger. | Hold if the work changes selection/exposure but leaves no readout for humans or agents. |
| Rollback / retirement | The issue names how to disable, demote, park, or retire the feature if its assumption fails. | Hold if promotion is one-way or rollback depends on branch/worktree separation. |

## Required Closeout Fields

When closing generation-readiness work, include:

- integration surface;
- runtime gate or "safe by default" statement;
- witness command or artifact;
- doc/self-index discovery path;
- operator visibility path;
- rollback, demotion, or retirement path;
- invalidating assumption.

## Demotion Path

If any gate regresses after promotion:

1. Record the failed gate and evidence in the issue or successor issue.
2. Move the item away from `gen/now` only if the evidence changes the horizon.
3. Prefer demotion, parking, or retirement over leaving a stale `gen/now` label.
4. Keep priority labels unchanged unless urgency also changed.

## Invalidating Assumption

This gate assumes generation readiness is reviewed at issue close, release
readiness, or operator milestone review time. If no workflow reads these gates,
the checklist becomes decorative; the next implementation step should wire the
six booleans into a command or report before using the gate to block promotion.

