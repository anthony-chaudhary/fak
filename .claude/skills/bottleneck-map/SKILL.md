---
name: bottleneck-map
description: Map the current "what is limiting us right now?" state across the running agent fleet and the live GitHub issue backlog. Runs fleet bottleneck detection plus issue triage, records the dominant system bottlenecks, open-work bottlenecks, and the durable process loop to run next. Use when the operator asks for bottlenecks, open-work constraints, fleet health, issue/backlog limits, or a durable ongoing process for keeping both visible.
allowed-tools: Read, Bash, Write, Grep, Glob
metadata:
  opencode: claude-only   # local note writes + no-GitHub-mutation discipline are load-bearing and not portable per-skill
---

# /bottleneck-map - system + backlog limiter map

> Wraps `tools/fleet_bottleneck.py` and `tools/issue_triage.py`. The point is
> one operator answer: what is the actual limiter right now, what issue work is
> blocked behind it, and which durable loop should run next?

This skill is a read-mostly coordination pass. It may write local evidence
artifacts and a dated note, but it must never mutate GitHub state. Labeling,
closing, assigning, or commenting on issues stays in `/issue-triage` and still
requires operator approval.

## Project Contract

Read `.claude/project.yaml` from the repo root. Required keys:

- `python` - interpreter path, default `python`.
- `helpers.fleet_bottleneck` - fleet/system bottleneck helper.
- `helpers.issue_triage` - open-issue triage helper.
- `audits_dir` - ignored operator snapshots, default `docs/_audits/`.

If either helper is absent, print one line naming the missing key and stop. Do
not reconstruct these scans by hand from ad hoc `gh issue list` pages.

## Step 1 - Refresh System Bottlenecks

If the `tools` lane is safe to mutate, run the cheap registry-only CLI pass:

```bash
<p> <h.fleet_bottleneck> report --no-audit
<p> <h.fleet_bottleneck> json --no-audit --out tools/_registry/bottleneck-map-latest.json
```

The CLI refreshes `tools/_registry/BOTTLENECKS.txt`, `bottlenecks.json`, and
`fleet_bottleneck.prom` as a side effect. If `tools` is held by another worker,
do not run the CLI. Use a read-only import of `collect()` + `rank_bottlenecks()`
instead and write only the human note under `docs/notes/`:

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

Use `--audit` only when the question is specifically about token/cache waste or
top spenders; it reads transcripts and is slower:

```bash
<p> <h.fleet_bottleneck> report
```

Extract only the load-bearing fields:

- `health` and registry age.
- headline bottleneck.
- top 3-6 ranked bottlenecks with score/severity/symptom/fix.
- active slot counts: `active`, `workers_on_throttled`, `rate_limited`,
  `auth_blocked`, `auto_resume`, `hanging`.
- classification: evidence scope, horizon, and weight.

Interpretation rule: if any system bottleneck is CRITICAL, do not recommend
launching more workers as the first move. Name the recovery/account fix first,
but do not confuse a transient gate with the most important durable problem.

## Step 2 - Refresh Issue/Open-Work Bottlenecks

Run the same read-only issue pass `/issue-triage` uses:

```bash
<p> <h.issue_triage> --markdown --out <audits_dir>/issue-triage-<YYYY-MM-DD>.md
<p> <h.issue_triage> --actions  --out <audits_dir>/issue-actions-<YYYY-MM-DD>.json
```

Read the markdown report and actions manifest. Extract:

- open issue count and missing-label counts.
- orphan P0/P1 count.
- mechanical action count vs review-only count.
- top P0 rows and the top P1 rows if they explain the bottleneck.

Interpretation rule: if mechanical actions are zero and review-only actions are
large, the backlog bottleneck is judgment/ownership, not stale cleanup. Do not
invent automatic closes.

## Step 3 - Decide The Next Loop

First classify each bottleneck:

| Field | Values | Rule |
|---|---|---|
| Evidence scope | `public`, `operator-private` | Public evidence is GitHub issue numbers/titles/labels, public docs, and commit refs. Operator-private evidence is account names/dirs, auth state, session IDs, transcript details, reset times, private host paths, and token/cost details. Tracked public notes should aggregate or redact private evidence. |
| Horizon | `transient`, `semi-durable`, `structural` | Rate limits, auth failures, temporary API errors, and at-cap pressure are transient unless repeated across passes. Watchdog absence, auto-resume backlog, stale telemetry, and surfacing backlog are semi-durable operating-loop problems. Missing labels, orphan P0/P1, broken routing, and missing ownership are structural open-work problems. |
| Weight | `dispatch-gating`, `process-debt`, `strategic` | Transient CRITICAL/HIGH rows gate dispatch now but should be rechecked before becoming strategy. Semi-durable rows become process debt if still present next pass. Structural rows carry strategic weight because they do not clear with time. |

Keep two outputs: `stop/continue dispatch now` and `what durable loop should
improve next`. They are allowed to differ.

Use this table. The first true row wins.

| Evidence | Bottleneck | Next durable loop |
|---|---|---|
| fleet health CRITICAL/HIGH from transient account pressure | system capacity gate | pause/cap dispatch; recheck after reset/relogin before treating it as strategic debt |
| fleet health CRITICAL/HIGH from recovery/telemetry rows | semi-durable recovery debt | fix watchdog/auto-resume/surfacing; keep broad dispatch capped |
| issue mechanical actions > 0 | stale/dormant gardening | ask operator to approve `/issue-triage` mechanical batch |
| orphan P0/P1 > 0 | ownership/claiming | claim/defer the top rows; then dispatch one issue per worker |
| needs-priority/area dominates | taxonomy debt | run focused `/issue-triage --scope priority` or `--scope area` |
| system green and top issues routed | execution | run issue-dispatch / DOS dispatch with witness gates |
| repeated same blocker over passes | process debt | file or update a small process issue naming the missing guard |

Never say "work on everything." Pick one limiter and one next loop.

## Step 4 - Record A Dated Note When Asked

When the operator asks for a map, writes should be a tracked dated note under:

```text
docs/notes/BOTTLENECK-MAP-YYYY-MM-DD.md
```

Use this shape:

1. Evidence gathered: exact commands, UTC timestamp, output files.
2. System bottleneck map: ranked table from `fleet_bottleneck`.
3. GitHub/open-work map: triage counts, top P0/P1 rows, action-manifest shape.
4. Scope/horizon/weight map: public vs operator-private, transient vs structural.
5. Durable loops: which skill/process runs daily, weekly, and before dispatch.
6. Next-pass done condition: concrete numbers that should fall.

Keep the note short enough to read in one screen. Link the ignored audit files
instead of pasting large JSON.

## Step 5 - Hand Off To Standing Loops

The permanent process home is the fleet control pane:

```bash
python tools/fleet_control_pane.py loop-add bottleneck-map --scope repo --status-cmd "<status command>" --apply
```

Only make that `tools/control_pane.loops.json` edit when the `tools` lane is safe
to edit. Until then, write the pending hook into the dated note and keep this
skill as the manual front door.

The standing loop should be read-only and should report ACTION, not mutate:

- ACTION if fleet health is CRITICAL/HIGH.
- ACTION if orphan P0/P1 is nonzero.
- ACTION if needs-priority or needs-area regresses above the last baseline.
- PASS if the counts are flat/improved and system health is below HIGH.

## Hard Rules

- Do not run `gh issue edit`, `gh issue close`, `gh issue comment`, or `gh issue
  assign` from this skill.
- Do not run the `fleet_bottleneck.py` CLI while another worker holds the
  `tools` lane; import `collect()`/`rank_bottlenecks()` read-only instead.
- Do not treat a stale/missing fleet registry as healthy. Say "flying blind" and
  fix telemetry first.
- Do not treat `orphan` as "unimportant." P0/P1 orphan work is the top backlog
  coordination cost.
- Do not recommend more workers while account throttling, auth, or recovery
  bottlenecks score CRITICAL.
- Do not publish account directory names, session IDs, auth details, reset
  timestamps, private host paths, or token/cost details in a tracked public note;
  aggregate them unless the operator explicitly asks for private output.
- Do not paste full issue reports or bottleneck JSON into chat; summarize the
  top evidence and point at the generated files.

## Cadence

Run at the start of a dispatch window, after a large issue-filing burst, after
any fleet stall, and before marking the operating process healthy. On a loop
cadence, this skill is the front door; it routes into `/issue-triage`,
issue-dispatch, or recovery/account maintenance.
