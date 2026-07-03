---
title: "Work-class axis: infra vs dev vs front-door/mainline"
description: "The selectable axis that separates fleet infrastructure work from product dev leaves and the public release/front-door path — derived from the file-tree lane an issue routes to, backfilled to class:* GitHub labels, and surfaced as three issue-views."
---

# Work-class axis — infra · dev · front-door/mainline

The backlog is tracked along several axes — priority (P0/P1/P2), area
(agentic-serving, compute, gpu, …), tracks A–G, generations, milestones. None of
them answers *what KIND of work is this?* — is it fleet plumbing, a product leaf,
or something on the public release path? Without that axis the default dispatch
surface mixes CI/dispatch/observability machinery, product/kernel leaves, and
release-path work into one undifferentiated stream, and you cannot ask "hide the
plumbing, show me product leaves" or "what open work touches the front door?".

The **work-class axis** adds exactly that. It is orthogonal to area and priority:
area says *which subsystem*, class says *which kind of work*.

## The three classes

| Class | `class:*` label | What it is |
|---|---|---|
| **front-door / mainline** | `class:frontdoor` | The public release path — install / README / getting-started front door, release promotion + version-everything, the branch-regime cutover. User-visible; review-heavier. The *fenced* bucket. |
| **infra** | `class:infra` | Fleet machinery — CI/CD, dispatch/supervisor loops, observability/metrics, slack cadence, build, testing infra, host maintenance. |
| **dev** | `class:dev` | Product/kernel leaves — the residual default (`engine`, `model`, `gateway`, `compute`, `recall`, …). The clean day-to-day dispatch surface. |

The vocabulary matches the branch-regime ADR ([`branch-regime.md`](branch-regime.md),
#1694), which already defines **front-door / development / release** roles at the
*branch* level. This axis carries the same distinction down to the *issue* backlog
and the lane taxonomy.

## How the class is derived (not hand-labeled)

The class falls out of the **file-tree lane an issue already routes to** — it is not
picked by hand per issue. `tools/issue_lane_router.py` routes every open issue to a
`dos.toml` lane (by scope, label, or a path-grep confidence ladder), and
`derive_class(issue, lane)` maps that lane to a class:

1. **Front-door override wins first** (regardless of lane), whenever a cross-cutting
   release-path signal fires — a front-door label (`version-everything`, …), a
   release-path scope (`release`, `install`, `readme`, `promote`, `version`,
   `branch-regime`, …), or a public front-door surface named in the title/body
   (`README.md`, `INSTALL.md`, `install.sh`, `docs/branch-regime*`, a release
   workflow). Widest-signal on purpose: a false-positive *into* the fenced bucket is
   safer than a release-path issue leaking into the default dev stream.
2. Else the lane's **seed class** decides (`LANE_CLASS` in the router): the hard-classed
   infra lanes (`ci`, `metrics`, `slackwire`, `slackoutbox`, `dispatchauto`,
   `loopdrive`, `rsiloop`, `tracesink`, `operatorbrief`) and front-door lanes
   (`appversion`, `shipgate`, `release`).
3. Else, for a **mixed lane** (`tools`, `docs`, `cmd`) a fleet-plumbing cue
   (dispatch / observability / CI / slack / build) promotes it to `infra`.
4. Else it falls to the **`dev` residual**.

Precedence: **frontdoor > infra > dev**. Same issue, same class — the derivation is
pure and deterministic.

## Using it

**See the split** (read-only, no writes):

```bash
python tools/issue_lane_router.py            # prints a "by class:" rollup + per-lane class mix
python tools/issue_lane_router.py --json     # class field on every routed issue + a `classes` section
```

**Dispatch by class** — three issue-views ([`.github/issue-views.json`](../.github/issue-views.json)):

- `dev-leaves` — product/kernel leaves only; infra + front-door hidden. The clean
  day-to-day surface.
- `infra` — fleet plumbing only.
- `front-door` — the fenced release path only.

```bash
fak index work dev-leaves            # the view's ready-to-run gh query
fak dispatch tick --view dev-leaves  # scope a dispatch tick to product leaves
```

**Backfill the labels** onto existing issues (the one write path, operator-gated):

```bash
python tools/issue_lane_router.py --apply-labels          # DRY-RUN: prints the label diff, writes nothing
python tools/issue_lane_router.py --apply-labels --apply-labels-write   # actually create labels + edit issues
```

New issues get their class at triage time — `tools/issue_triage.py` carries a
`needs-class` gap and the `class:*` axis alongside `area`.
