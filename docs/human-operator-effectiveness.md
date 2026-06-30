# Human Operator Effectiveness

Human operator effectiveness is an ongoing program, not a feature with a
completion bar. The goal is to keep fak understandable and steerable as the
agent fleet, CLI surface, refusal vocabulary, and background loops grow.

The program's job is to make a human better used:

- Use humans for judgment: policy choices, priority calls, missing witnesses,
  trust-boundary decisions, and review of repeated regressions.
- Let agents do routeable work: issues with a lane, a next action, and a witness.
- Keep learning bounded: one current focus, one practice step, and one explicit
  thing to skip.
- Keep evidence coherent: cached panes are useful only when their source stamps
  are visible.
- Keep check-ins cheap: an operator who read the last brief should start with
  what changed, resolved, or persisted.
- Keep the surface light: a growing system should not require a growing amount
  of operator ceremony.

## Frontiers

The program is measured by frontiers and trends, never by "100% complete":

| Frontier | Surface | Meaning |
|---|---|---|
| Human attention | `fak operator brief` | Separates `human`, `agent`, `watch`, and `background` work so a person sees what actually needs them. |
| Learning pace | `learning_agenda` in the operator brief | Names the one thing to understand now, the practice step, the skip rule, and the drill-down sections. |
| Change compression | `since_previous` in the operator brief | Compares the current brief with a previous brief, showing new, resolved, and persistent attention-bearing items before detail. |
| Source coherence | `coherence` in the operator brief | Says whether cadence/program/milestone/heaviness reports describe one snapshot or a mixed cached state. |
| Surface pressure | `fak operator heaviness` | Reads the command surface, guard flags, refusal vocabulary, doc-map discoverability, and appeal channel. |
| Strategic tracking | `fak program report` | Tracks `human-operator-effectiveness` beside kernel and cache programs, using operator-heaviness lightness as the first deterministic metric. |

## Operating Rule

Do not make the operator read every transcript. A useful control plane should
compress the fleet into:

1. The state of the system.
2. The choice a human can actually make.
3. The challenge worth understanding.
4. The work agents should keep doing.
5. What changed since the previous review.
6. The learning focus for this review.

If a surface cannot answer those five, it is background telemetry, not an
operator page.

## Current Witnesses

- `fak operator brief --collect --json` folds the source reports into the
  operator envelope; `--previous last-brief.json` adds the `since_previous`
  change delta.
- `fak operator heaviness --json` emits `heaviness_pressure` and
  `heaviness_debt`.
- `fak program report --json` includes the `human-operator-effectiveness`
  program signal.

The current metric is intentionally modest: `human_metric = max(0, 100 -
heaviness_pressure)`, forced to `0` when hard `heaviness_debt` exists. It is a
first proxy for how light the system feels to drive. Better future signals can
add operator response time, repeated watch items, unresolved human choices, and
brief read-time, but they must stay witnessed rather than self-reported.
