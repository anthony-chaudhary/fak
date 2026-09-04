---
title: "fleetctl: the public box-fleet control surface"
description: "fleetctl is fak's public, transport-agnostic Go core for operating a fleet of GPU and worker boxes: typed roster, deterministic fold, readiness score."
---

# fleetctl — the public box-fleet control surface

`fleetctl` (`cmd/fleetctl/`) is the **public, transport-agnostic** Go core for operating
a fleet of boxes — GPU servers, worker nodes — that the operator drives over the private
private control bridge. It is the single Go home the scattered `tools/fleet_*.py` helpers
port into: a typed roster, a deterministic fold, a 0–100 readiness score, and a view that
stays readable as the fleet grows toward (and past) 100 boxes.

It is Go-only and depends on nothing outside the standard library. Build it like any other
binary in this module:

```bash
go build -o fleetctl ./cmd/fleetctl
```

## The public / private boundary is a data contract, not a code import

The **live control plane** — the private control bridge that actually reaches the lab boxes —
is private. It speaks a lab protocol and carries lab identifiers (host, channel, token), so
it lives in `fak-private`, never here. See [`gpu-server-private-boundary.md`](gpu-server-private-boundary.md)
for what is public vs private and which gates enforce it, and
[`private-comms-channel.md`](private-comms-channel.md) for how to reach the channel.

The seam between that private bridge and this public tool is a **per-box report JSON**
(`fak.fleet.report/v1`). The private bridge writes one report file per box from live state;
`fleetctl` reads, folds, renders, and scores them. Neither side imports the other, and
nothing in this tree names a host, a channel, or a token — only a generic box id, a class,
a state word, a version, and an age.

```
  private (fak-private)                 public (this repo)
  ┌───────────────────┐   report JSON   ┌────────────────────────────┐
  │  Slack bridge      │ ──────────────▶ │  fleetctl                  │
  │  (reaches the box) │   one file/box  │  roster + fold + score     │
  └───────────────────┘                 │  + render                  │
                                         └────────────────────────────┘
```

## The roster

A roster is a JSON file listing the boxes you control. Every field but `id` is optional;
`endpoint` is an **opaque** reference the transport resolves (the public file transport
treats it as the report-file stem; the private bridge resolves it to a channel/session).

```json
{
  "schema": "fak.fleet.roster/v1",
  "boxes": [
    {"id": "box-001", "class": "a100x8", "group": "lab-1", "labels": {"region": "us-west"}},
    {"id": "box-002", "class": "h100x8", "group": "lab-2"}
  ]
}
```

## Adding up to 100 boxes — one command

`template` scaffolds a roster of N boxes (ids are zero-padded so they sort in order). This
is the "how do I stand up 100 boxes?" answer — scaffold, then edit:

```bash
fleetctl template --count 100 --class a100x8 --group lab-1 > roster.json
fleetctl validate --roster roster.json     # fail-loud on a duplicate/empty/bad id
```

## The report seam

Each box's current state is one JSON file the private bridge writes into a reports
directory, named `<endpoint-or-id>.json`:

```json
{
  "schema": "fak.fleet.report/v1",
  "state": "live",          // live | idle | draining | down | unknown
  "version": "0.31.0",
  "age_sec": 12.5,
  "note": "throttled until 14:05",
  "inference": {
    "status": "ready",      // ready | degraded | warming | blocked | unknown
    "engine": "fak",
    "model": "qwen",
    "output_tps": 1.75,
    "reason": "scrubbed-reason"
  }
}
```

A box with no report file is shown as **unreachable** — the view never crashes on one
silent box, and an empty reports directory honestly scores the fleet at 0. The reader
floors each box's age at its report file's own mtime, so a **frozen** file from a bridge
that stopped updating ages out and trips the stale warn instead of reading green forever
— the producer must therefore re-stamp `age_sec` on every write.

Two seam fields are operator-facing and **must stay generic** in anything committed
publicly: `note` is rendered verbatim (keep it pre-scrubbed — never a lab hostname,
channel, or operator path), and a roster's `endpoint`/`labels` must never carry a real
channel/session/token (the private bridge owns the id→channel map on its side).

The optional `inference` block answers the question liveness cannot: is this box useful
for model inference right now? `ready` and `degraded` count as useful; `warming`,
`blocked`, and `unknown` do not. The labels are public-safe serving facts only. Never put
a URL, host, channel id, token, private model path, or raw bridge transcript in the block.
If no producer can prove the serving state, omit `inference` or set `status` to
`unknown`; do not report false idle/ready.

## Lab-machine dev readiness

Dispatch planning needs a smaller yes/no surface than the full operator report: can this
class of lab machine safely take dev work right now? The public answer is a scrubbed
readiness record. It is derived from private readbacks, but it carries only generic class
and status words:

```json
{
  "schema": "fak.lab_readiness/v1",
  "machine_class": "gpu-server",
  "checked_at": "2026-07-04T14:00:00Z",
  "status": "WAIT_PRIVATE_RECOVERY",
  "next_action": "confirm-private-control-session",
  "evidence": "scrubbed-private-readback"
}
```

`status` is a closed vocabulary:

| Status | Dispatch meaning |
|---|---|
| `READY_FOR_DEV_WORK` | The machine class may be offered to lab-backed dev workers. |
| `WAIT_PRIVATE_RECOVERY` | Do not dispatch work there; an operator must recover the private control path first. |
| `GATEWAY_UNREACHABLE` | Do not dispatch model/gateway work there; keep any existing watcher alive and recover the gateway privately. |
| `AUTH_OR_CHANNEL_BLOCKED` | Do not retry public dispatch; the private auth/channel state needs operator action. |
| `INDETERMINATE` | Fail closed. Use local workers or another ready class until a stronger readback exists. |

The record may name a generic class (`gpu-server`, `mac-worker`, `linux-worker`) and a
generic next-action class. It must not name a host, endpoint, channel, token, account id,
raw transcript, or private filesystem path. Super-loop and issue-dispatch planning should
treat anything other than `READY_FOR_DEV_WORK` as no lab-machine capacity.

The readiness record does not have to be hand-authored. Once the private bridge or a
self-reporting box has written scrubbed `fak.fleet.report/v1` files with inference status,
derive and publish the gate from those reports:

```bash
fak lab readiness --from-reports --write-default --json
```

`--from-reports` (formerly `--from-status`, still accepted as a deprecated alias) admits
lab-backed dispatch only when at least one healthy box reports
fresh `inference.ready` or `inference.degraded`; `warming`, `blocked`, stale reports,
missing reports, and unknown inference all fail closed with a generic next action. With a
private roster, pass that roster locally (`--roster <private-roster>`); never commit the
roster path or any endpoint it resolves.

## Lab inference targets

Readiness answers whether lab-backed work may be admitted. A guarded turn still needs a
local endpoint target. Keep that second seam local too: the private bridge/tunnel producer
writes `$FAK_LAB_TARGETS`, or the default fak config file `fleet/lab-targets.json`, with
private coordinates in an untracked local file:

```json
{
  "schema": "fak.lab_targets/v1",
  "targets": [
    {
      "alias": "@lab/glm-5.2",
      "base_url": "http://localhost:PORT",
      "model": "glm-5.2",
      "roster": "C:/local/private/roster.json"
    }
  ]
}
```

`box_id` is optional. Use it only for a display-safe generic box ID; when the entry
points at a private roster, omit `box_id` so `fak lab target --json` can validate the
scrubbed report without printing the private report key.

Validate the alias without printing the resolved coordinates:

```bash
fak lab target @lab/glm-5.2 --json
```

The resolver requires all three witnesses before it admits: `READY_FOR_DEV_WORK`
readiness, a local target config entry, and a fresh healthy scrubbed report whose
`inference` block is `ready` or `degraded` for the requested model. Then the user-visible
guard path is:

```bash
fak manage --remote-serve @lab/glm-5.2 --probe -- codex
```

### First green run (before the private bridge is wired)

To see a populated frame without the bridge, drop a sample report into a directory and
point `status` at it:

```bash
fleetctl template --count 3 > roster.json
mkdir reports
echo '{"schema":"fak.fleet.report/v1","state":"live","version":"0.31.0"}' > reports/box-001.json
echo '{"schema":"fak.fleet.report/v1","state":"idle","version":"0.31.0"}' > reports/box-002.json
fleetctl status --roster roster.json --reports reports     # box-003 shows unreachable
```

## Commands

```bash
fleetctl ls     --roster roster.json [--group G] [--class C] [--json]
fleetctl status --roster roster.json --reports DIR [--group G] [--class C] [--json] [--all]
fleetctl score  --roster roster.json --reports DIR [--min N] [--group G] [--class C]
```

`status` summarizes by default (counts by state and class, the version picture, a capped
attention list) so a 5-box fleet and a 500-box fleet print a frame of the same bounded
height; `--all` appends the per-box table. `--group`/`--class` scope any of the three to
a subset — the first thing you reach for at 100 boxes. `--stale-min N` tunes how many
minutes of silence flags a box (default 15). `score --min N` exits non-zero when readiness
is below `N`, so it drops straight into a watchdog or a `/loop`.

Exit codes are scriptable: **0** ok · **1** the `score --min` gate fired · **2** a
usage / roster / `--reports` error. A missing or mistyped `--reports` directory fails loud
with exit 2 rather than silently scoring 0, so a watchdog never mistakes a config typo for
a fleet-wide outage.

Example summary:

```
== fleet - 100 box(es) - readiness 96/100 =============================

REACHABLE  98/100
STATE      live=86 idle=10 down=2 unknown=2
CLASS      a100x8=64 h100x8=36
VERSION    0.47.1  (6 reachable box(es) on other/none)
INFERENCE useful=72/100 reported=80 ready=68 degraded=4 warming=3 blocked=2 unknown=3

ATTENTION
  [CRIT] 4 box(es) down or unreachable
        box-031, box-047, box-068, box-091
  [CRIT] 2 box(es) blocked for inference
        box-014(blocked/needs-operator), box-062(blocked/no-model)
  [WARN] 4 box(es) off the fleet version 0.47.1
        box-005@0.47.0, box-018@0.47.0, box-052@0.47.0, box-077@0.47.0
```

## The readiness score

`score` is a deliberately simple, predictable 0–100 blend an operator can reason about:

```
score = 100 * ( 0.6*usable_frac + 0.2*reach_frac + 0.2*version_coverage_frac )

  usable_frac           = healthy boxes (live|idle|draining) / total
  reach_frac            = boxes that returned a trustworthy report (incl. down) / total
  version_coverage_frac = boxes on the single most common version / total
```

Usability dominates (an unreachable or down box is the real problem); reach gives credit
for observability (knowing a box is down beats not knowing); version coverage rewards a
single consistent fleet version. An all-healthy single-version fleet scores 100, an
all-down-but-visible fleet scores 20, an all-silent fleet scores 0. The score is a fence,
not a benchmark — the per-state counts and the attention list carry the detail.

## Commit-throughput health

For an active coding fleet, process activity is not enough: healthy means real work is
landing. `fak fleet health` counts file-changing commits on `HEAD`'s first-parent history
in adjacent ten-minute windows. Working-tree edits, side-branch commits, empty commits,
and agent self-reports do not increase the count.

```bash
fak fleet health
# commits/10m=3 previous=1 state=healthy latest_age=2m0s
# active_workers=3 healthy=true reason=3 landed commit(s) in the last 10 minutes

fak fleet health --json
fak fleet metrics | grep fak_fleet_commit
```

The top-level state contract is:

| Active workers | Current 10m | Previous 10m | State | Meaning |
|---:|---:|---:|---|---|
| 0 | any | any | `idle` | Neutral: no fleet is expected to ship. |
| >0 | >0 | any | `healthy` | Positive landed throughput. |
| >0 | 0 | >0 | `stalled` | The current window stopped producing; inspect admission, leases, tests, and pushes. |
| >0 | 0 | 0 | `blocked` | Two consecutive zero windows; stop adding fan-out and repair the shared blocker. |
| >0 | unknown | any | `unknown` | Git history or the durable worker registry is unreadable; fail closed. |

`fak fleet metrics` exports `fak_fleet_commits_per_10m`,
`fak_fleet_commits_previous_10m`, `fak_fleet_commit_throughput_healthy`,
`fak_fleet_commit_throughput_measured`, `fak_fleet_latest_commit_unixtime`, and the
one-hot `fak_fleet_commit_throughput_state{state=...}` family. Alert on
`fak_fleet_commit_throughput_healthy == 0` only while the fleet has active workers;
an idle fleet is deliberately healthy-neutral.

The reusable loop gate is `superloop.GateCommitThroughput`: it converts an otherwise
clean active loop with zero throughput into a recovery member, without displacing
concrete work already selected by the loop.

## Honesty

`fleetctl` is the public **core**: roster + fold + render + score + the file transport that
reads reports off disk. It does **not** reach a live box — producing the reports is the
private bridge's job. Pointed at a reports directory the bridge wrote (or a fixture) it is
fully exercised; pointed at no reports it honestly shows every box as unreachable. Witness:
`go test ./cmd/fleetctl`.
