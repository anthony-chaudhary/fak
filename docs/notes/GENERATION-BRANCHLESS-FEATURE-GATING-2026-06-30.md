# Generation Branchless Feature Gating

This note closes #1646. It specifies how generation metadata maps to runtime
feature flags, experimental commands, docs-only work, and no-op scaffolds
without creating a branch per horizon.

Generation remains orthogonal to priority, shared trunk, and runtime feature
gates:

- Priority ranks urgency or value inside the eligible work set.
- Shared trunk means every stream lands on `main`, by explicit path, with the
  same witness and commit guards.
- Runtime feature gates decide exposure. A generation label never makes code
  reachable and never substitutes for a default-off flag.

## Branchless Rule

Generation labels are planning metadata. Runtime code must not parse `gen/now`,
`gen/next`, `gen/second-next`, or `gen/future` to decide whether a feature is
enabled.

The branchless rule is:

1. Land the smallest artifact on `main`.
2. Keep exposure controlled by the artifact's own mechanism: flag, command
   mode, documentation status, or inert scaffold.
3. Record the generation label in the issue, milestone, commit sidecar, report,
   or note.
4. Promote, demote, park, or retire the work only after concrete evidence
   changes the horizon.

## Exposure Matrix

| Surface | Allowed generation use | Required exposure gate | Promotion evidence | Demotion or retirement evidence |
|---|---|---|---|---|
| Runtime feature flag | Generation explains why the flag exists and which horizon owns the assumption. It is not read by the hot path. | An explicit flag, env var, config field, policy field, or command option with a documented default. `gen/next` and later default to off or safe-no-op unless the feature is read-only and safe by construction. | Focused test or captured command shows gate-off is inert, gate-on reaches only the intended path, and the feature has a caller or dogfood workflow. | Gate-on regresses, the owner/witness goes stale, the flag remains default-off with no caller after review, or the feature cannot be exposed safely. |
| Experimental command | Generation identifies the command as near-term, second-next, or future work for dispatch and release notes. | A visible command mode such as `--experimental`, `--dry-run`, `--execute`, read-only default behavior, or an operator-only subcommand. The generation label is not the authorization check. | Help text, JSON output, and a focused command witness show the experimental path is discoverable, bounded, and does not change defaults accidentally. | The command leaks into default workflows, lacks an owner, repeats failed witnesses, or is superseded by a stable command. |
| Docs-only work | Generation states the horizon and assumption so agents can carry the idea without treating it as shipped runtime behavior. | No runtime gate. The gate is document status: committed note, index entry, issue comment, or saved view that says what decision the doc can influence. | A later issue, dispatch rule, release-readiness row, or operator decision cites the doc and turns it into runnable work or an explicit parked option. | The assumption fails, no decision consumes the note by its review date, or a stronger shipped artifact supersedes it. |
| No-op scaffold | Generation labels the scaffold as option value only. A scaffold alone is not promoted. | Compile-time registration, inert default, no live caller, no changed default policy, and a test proving the off/no-op path. | A live caller, runtime gate, or operator workflow lands with a focused witness; the scaffold moves from inert option to bounded experiment. | The scaffold stays uncalled, obscures real status, creates maintenance load, or violates layer/import rules. |

## Stream Defaults

`gen/now` work should normally be exposed through the product's current default
path once its normal witness is green. If it still needs a default-off runtime
gate, the issue should explain why it is `now` rather than `next`.

`gen/next` work may land behind an explicit gate or as read-only operator
surface. Its closeout must name the gate, the owner, and the witness that could
move it toward `gen/now`.

`gen/second-next` work should usually land as a doc, compatibility policy,
simulation, or no-op scaffold. A runtime flag is acceptable only when gate-off
is proven inert and gate-on is not part of normal operator flow.

`gen/future` work should usually land as research, narrative, or option mapping.
Runtime code in this stream needs a stronger explanation: what cheap artifact is
being preserved now, why it cannot stay as a doc, and how it will be retired if
the premise fails.

## Closeout Fields

A branchless generation issue that touches exposure should close with:

- generation stream;
- runtime gate name, command mode, doc artifact, or no-op scaffold;
- default behavior;
- gate-off or no-op witness when code is present;
- promotion evidence that would move the item closer to `now`;
- demotion, parking, or retirement evidence;
- invalidating assumption.

## Invalidating Assumption

This contract assumes runtime gates remain grep-able and documented near the
feature they expose. If gates are scattered across env vars, command flags, and
policy fields with no machine-readable registry, generation labels may become
the only visible control plane by accident. At that point, add a native gate
inventory or report before using generation metadata to make exposure decisions.

