# Generation-Aware Dispatch Selection

This note is the selection contract for #1641. It tells dispatch workers how to
use generation labels without turning them into priority labels, branches, or
runtime exposure flags.

## Inputs

The dispatch picker may read these facts from the routed issue payload:

- lane pressure: lane `step_budget`, issue count, live leases, cooldowns, and
  path scope;
- issue urgency: existing `priority/P0`, `priority/P1`, and `priority/P2`
  weights;
- generation: exactly one of `gen/now`, `gen/next`, `gen/second-next`,
  `gen/future`, or `unclassified`.

Generation is an eligibility and portfolio-balance signal. It is not a priority
override. A `priority/P0` issue still outranks a default-priority issue inside
the same eligible horizon, but a high-priority future issue is not silently
pulled into immediate ship lanes unless the operator asks for that horizon.

## Default Window

The default dispatch window is:

1. `gen/now` issues are eligible for normal dispatch.
2. `gen/next` issues are eligible when they are scoped, witnessed, and
   dogfoodable, because next-gen foundation work should not starve behind
   current-lane pressure forever.
3. `gen/second-next` and `gen/future` issues are held by default unless the
   operator explicitly selects that horizon, selects an issue directly, or
   routes a research/planning worker. Holding them is not a value judgment; it
   prevents long-horizon research from consuming immediate ship slots by
   accident.
4. `unclassified` issues are held for classification repair unless they carry a
   direct target issue override.

## Ordering Rule

For an automatic wave with no explicit generation filter:

1. Drop live, cooling, blocked, unclassified, `gen/second-next`, and
   `gen/future` candidates from the launchable set.
2. Rank lanes by existing lane pressure: higher `step_budget`, then higher
   count, then lane name. This preserves the current dispatch pressure model.
3. Inside each eligible lane, rank issues by existing priority weight, then by
   the lane's oldest/newest tiebreak setting.
4. Apply the gen/next starvation guard:
   - If at least one scoped `gen/next` issue exists in a launchable lane and no
     gen/next issue has been selected in the current planning window, reserve
     at most one slot for the highest-ranked gen/next candidate after the first
     gen/now slot.
   - A single-worker tick may choose either the highest pressure gen/now
     candidate or the reserved gen/next candidate only when the gen/next lane's
     pressure is within one step of the winning gen/now lane. This keeps next
     work moving without letting it erase current pressure.
   - Multi-worker waves should report the reservation decision in the price
     payload so an operator can see whether next-gen work is being starved or
     intentionally deferred.

This keeps generation orthogonal: priority answers "how urgent is this issue?";
generation answers "which product horizon may consume this dispatch slot?"

## Explicit Horizon

An explicit horizon filter changes eligibility, not priority:

- `--generation now` admits only `gen/now`.
- `--generation next` admits only `gen/next`.
- `--generation second-next` admits only `gen/second-next`.
- `--generation future` admits only `gen/future`.
- `--generation all` admits every classified generation but still holds
  unclassified issues for classification repair.

Direct `--target-issue` selection remains the narrowest operator intent. The
worker still must obey shared trunk, path-scoped commits, feature gates, and the
issue witness.

## Implementation Shape

When this policy moves from note to native code, the router/wave payload should
surface enough data for an operator to audit the decision:

- per-issue `generation`;
- per-lane `generation_counts`;
- price/run rows with `generation`, `generation_rank_reason`, and
  `generation_held_reason`;
- a wave-level summary such as `generation_window`, `next_reserved`, and
  `held_generation_count`.

Focused tests should cover:

- future and second-next candidates are held by default;
- explicit horizon filters admit their requested horizon;
- priority still wins inside an eligible horizon;
- lane step budget still wins before generation balancing;
- the gen/next starvation guard can reserve one slot without displacing the
  first gen/now launch.

## Evidence Rules

Promotion evidence: a gen/next dispatch rule can move toward `gen/now` only
after a dry-run or live wave shows the selected next-gen worker had a scoped
issue, a clear witness, no branch split, and no default exposure leak.

Demotion or retirement evidence: a gen/next candidate should be held, split, or
demoted if its issue has no witness, no path scope, stale assumptions, or repeat
worker failures that do not produce a committed artifact.

Invalidating assumptions:

- The rule assumes dispatch workers consume the native routed issue payload. A
  legacy path that bypasses `fak dispatch route` needs the same generation
  fields or it will not observe this policy.
- The rule assumes generation labels are correct. If issue labels drift from the
  milestone or issue body, classify first and dispatch second.
- The rule assumes `step_budget` is a usable pressure proxy. If step budgets are
  noisy, the pressure model must be repaired before generation balancing is
  trusted.

