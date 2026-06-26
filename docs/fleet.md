# fleetctl — the public box-fleet control surface

`fleetctl` (`cmd/fleetctl/`) is the **public, transport-agnostic** Go core for operating
a fleet of boxes — GPU servers, worker nodes — that the operator drives over the private
Slack control-bridge. It is the single Go home the scattered `tools/fleet_*.py` helpers
port into: a typed roster, a deterministic fold, a 0–100 readiness score, and a view that
stays readable as the fleet grows toward (and past) 100 boxes.

It is Go-only and depends on nothing outside the standard library. Build it like any other
binary in this module:

```bash
go build -o fleetctl ./cmd/fleetctl
```

## The public / private boundary is a data contract, not a code import

The **live control plane** — the Slack control-bridge that actually reaches the lab boxes —
is private. It speaks a lab protocol and carries lab identifiers (host, channel, token), so
it lives in `fak-private`, never here. See [`dgx-slack-boundary.md`](dgx-slack-boundary.md)
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
  "note": "throttled until 14:05"
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
VERSION    0.31.0  (6 reachable box(es) on other/none)

ATTENTION
  [CRIT] 4 box(es) down or unreachable
        box-031, box-047, box-068, box-091
  [WARN] 4 box(es) off the fleet version 0.31.0
        box-005@0.30.0, box-018@0.30.0, box-052@0.30.0, box-077@0.30.0
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

## Honesty

`fleetctl` is the public **core**: roster + fold + render + score + the file transport that
reads reports off disk. It does **not** reach a live box — producing the reports is the
private bridge's job. Pointed at a reports directory the bridge wrote (or a fixture) it is
fully exercised; pointed at no reports it honestly shows every box as unreachable. Witness:
`go test ./cmd/fleetctl`.
