---
title: "Bottleneck map loop"
description: "Standing process for deciding what limits the fak fleet and open GitHub work before dispatch."
---

# Bottleneck map loop

Use this loop to answer one operating question before dispatch or recovery:
what is limiting us right now?

The loop has two inputs. The issue side is read-only on stdout:

```bash
python tools/issue_triage.py --json
```

The fleet side has one sharp edge: `tools/fleet_bottleneck.py report|json`
refreshes `tools/_registry/BOTTLENECKS.txt`, `bottlenecks.json`, and
`fleet_bottleneck.prom`. That is fine when the `tools` lane or control-pane loop
owns the refresh. If `tools` is held by another worker, collect the same evidence
without writing artifacts:

PowerShell form:

```powershell
@'
import importlib.util, json
from pathlib import Path
p = Path("tools/fleet_bottleneck.py")
spec = importlib.util.spec_from_file_location("fleet_bottleneck", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
snap = m.collect(audit=False)
ranked = m.rank_bottlenecks(snap)
r = snap.get("registry") or {}
print(json.dumps({
    "generated_utc": snap.get("generated_utc"),
    "health": ranked.get("health"),
    "headline": ranked.get("headline"),
    "counts": r.get("counts"),
    "top_bottlenecks": ranked.get("bottlenecks", [])[:8],
}, indent=2, default=str))
'@ | python -
```

Do not mutate GitHub from this loop. Label, close, assign, and comment actions
belong to `/issue-triage` after operator approval.

## Cadence

Run the loop:

- at the start of every dispatch window;
- after a fleet stall, rate-limit event, or auth failure;
- after a large issue-filing burst;
- before declaring the operating process healthy.

## Decision Order

Classify before choosing an action:

| Field | Values | Meaning |
|---|---|---|
| Evidence scope | `public`, `operator-private` | Public evidence can be tracked or filed publicly: GitHub issue numbers/titles/labels, public docs, commit refs. Operator-private evidence includes account names/dirs, auth state, session IDs, transcript details, reset times, private host paths, and token/cost details; public notes should aggregate or redact it. |
| Horizon | `transient`, `semi-durable`, `structural` | Rate limits, auth failures, temporary API errors, and at-cap pressure are transient unless they repeat. Watchdog absence, auto-resume backlog, stale telemetry, and surfacing backlog are semi-durable. Missing labels, orphan P0/P1, broken routing, and missing ownership are structural. |
| Weight | `dispatch-gating`, `process-debt`, `strategic` | Transient CRITICAL/HIGH rows gate dispatch now but should be rechecked before becoming strategy. Semi-durable rows become process debt if they persist. Structural rows carry strategic weight because time will not clear them. |

Keep two decisions separate:

- **Can dispatch safely continue now?** Transient account/auth/rate-limit pressure
  can answer no.
- **What durable loop should improve next?** Structural ownership/taxonomy debt can
  remain the top durable problem even while dispatch is temporarily gated.

Then use the first true condition:

| Evidence | Bottleneck | Next process |
|---|---|---|
| Fleet health is CRITICAL or HIGH from transient account pressure | Capacity gate | Pause/cap dispatch; recheck after reset/relogin before treating it as strategic debt. |
| Fleet health is CRITICAL or HIGH from recovery/telemetry rows | Semi-durable recovery debt | Fix watchdog, auto-resume, and surfacing backlog before broad dispatch. |
| Issue actions include mechanical stale/dormant work | Backlog gardening | Ask the operator to approve the `/issue-triage` action manifest. |
| Orphan P0/P1 count is nonzero | Ownership | Claim or explicitly defer top P0/P1 rows before launching more workers. |
| `needs_priority` or `needs_area` dominates | Taxonomy debt | Run a focused `/issue-triage` pass for priority or area labels. |
| Fleet is below HIGH and top issues are owned | Execution | Dispatch one issue per worker with issue-bound commits and witness gates. |
| The same blocker repeats across passes | Process debt | File or update one small process issue naming the missing guard. |

Do not recommend "more workers" while the fleet is CRITICAL/HIGH from account,
auth, watchdog, auto-resume, or surfacing failures. More workers are only useful
after the constrained operating loop is healthy enough to absorb them. Also do
not let a transitory rate-limit reset hide durable backlog debt: record both the
current gate and the structural problem that will remain when the reset clears.

## Evidence Note

When the operator asks for a map, write a tracked dated note:

```text
docs/notes/BOTTLENECK-MAP-YYYY-MM-DD.md
```

Record:

- exact commands and UTC timestamps;
- the top system bottlenecks and fixes from `fleet_bottleneck.py`;
- issue counts, action-manifest shape, and top P0/P1 rows from `issue_triage.py`;
- a scope/horizon/weight row that says which evidence is public vs
  operator-private and which bottlenecks are transient vs structural;
- the next durable process to run;
- concrete next-pass done conditions.

## Standing Hook

The permanent control-pane hook belongs in `tools/control_pane.loops.json` and
must wait for the `tools` lane to be safe:

```bash
python tools/fleet_control_pane.py loop-add bottleneck-map --scope repo --status-cmd "<read-only status command>" --apply
```

The hook should report ACTION when fleet health is CRITICAL/HIGH, orphan P0/P1
is nonzero, or label-debt counts regress above the last baseline. It should
report PASS only when counts are flat or improved and fleet health is below
HIGH.
