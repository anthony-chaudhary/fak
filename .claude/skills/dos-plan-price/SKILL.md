---
name: dos-plan-price
description: "Price a proposed multi-agent fan-out before launching workers. Use when a packet, goal fleet, or hand-written plan would run several agents over declared file trees and you need DOS to catch collisions before any worker starts."
---

# dos-plan-price - price fan-out before launch

Use this before launching more than one worker from a `dos-next-up` packet, a
goal fleet, or an operator-supplied partition. `dos arbitrate` is still the floor
at worker acquire time; this skill is the earlier pricing pass that avoids
spending launches on a partition whose trees already collide.

## Load-Bearing Rule

Do not launch a colliding fan-out. Price the proposed worker trees first, then
either launch the safe set or narrow the partition and re-price.

## Step 0 - Discover Workspace Geometry

```bash
dos doctor --workspace . --json
```

Read `lanes.trees` and resolve lane names to their configured trees. Do not
hardcode lane trees. A worker whose tree is unknown must stay unknown in the
partition; unknown scope is a collision risk, not permission to launch.

## Step 1 - Build The Proposed Partition

Create one entry per worker:

```json
{"name": "worker-name", "tree": ["repo-relative/glob/**"]}
```

Sources can be a `dos-next-up` dispatch list, a `dos-goal-fleet` wave, or an
operator's explicit worker list. If a worker has no named tree, record an empty
tree and treat the result as unsafe until the scope is made explicit.

## Step 2 - Price The Partition

There is no first-party `dos price-plan` CLI verb yet. Use the shipped interim
example if available, and log the gap instead of pretending the verb exists:

```bash
python examples/plan_price/plan_price.py --json
```

For a custom partition, import the example API from `examples/plan_price` if it
exists in the current workspace or in the installed DOS kernel checkout:

```python
from plan_price import Agent, price_plan

price = price_plan([Agent(name, tree) for name, tree in proposed])
```

Log this every run while the first-party verb is absent:

```text
log: predictive fan-out pricing is example-backed pending a first-party dos price-plan verb; dos arbitrate remains the acquire-time floor.
```

## Step 3 - Act On The Price

- `collisions == 0`: launch the full fan-out, then have every worker call
  `dos arbitrate` at acquire time.
- `collisions > 0`: do not launch the colliding plan. Launch only `safe_now`, or
  narrow the colliding trees and re-run the price.
- `unknown tree`: stop and name the worker's scope before launch, unless the
  operator explicitly chooses a serial run.

Return the action in a machine-readable final line:

```text
Action: LAUNCH_ALL | LAUNCH_SAFE_SET | REPARTITION_AND_REPRICE | SERIALIZE_UNKNOWN_SCOPE
```

## Step 4 - Keep The Reactive Floor

Pricing is a forward estimate over declared scope. It does not hold a lease and
does not replace `dos arbitrate`:

```bash
dos arbitrate --workspace . --lane <lane> --kind cluster
```

Honor the acquire/refuse verdict. Do not use pricing as a reason to force an
arbiter refusal.

## Anti-Patterns

- Launching a fan-out after the price reports a collision.
- Treating the price as a lease.
- Skipping `dos arbitrate` because the price was clean.
- Resolving unknown scope to `**/*` silently and launching anyway.
- Hardcoding lane names or trees instead of reading `dos doctor --json`.
