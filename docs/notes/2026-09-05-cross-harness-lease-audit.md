# Cross-harness lease audit

Audit date: September 5, 2026 Pacific (September 6 UTC). Trackers:
[FAK MCP admission #11834](https://github.com/anthony-chaudhary/fak/issues/11834),
[authoritative acquisition #11835](https://github.com/anthony-chaudhary/fak/issues/11835),
and [DOS strict snapshots #269](https://github.com/anthony-chaudhary/dos-kernel/issues/269).

## Executed observations

- The installed FAK CLI returned an empty array from `fak leaseref live` while
  DOS admission saw live `gateway`, `cmd`, and `issue-11548` leases. These views
  read different stores; an empty reference view did not prove the workspace free.
- The live DOS MCP doctor answered. A DOS MCP arbitration request timed out at
  five seconds; the direct DOS CLI returned a collision refusal for the held
  gateway lane. This distinguishes tool availability from each operation's health.
- A proposed write sent to the live `fak_adjudicate` MCP endpoint against a held
  gateway tree returned `ALLOW`, attributed to `monitor`. Its receipt explicitly
  said `execution=not_executed`; no file was written. A capability decision alone
  did not establish lease admission.
- In a disposable git repository, two `fak leaseref acquire` commands used
  different IDs and holders with the same `src/file.txt` tree. Both returned
  `ok=true`, generation 1. Both exact fixture leases were subsequently released.
  This witnessed per-ID fencing, not exclusive acquisition of overlapping trees.

## Source-confirmed boundaries

FAK reference leases live under `refs/fak/locks/*` in the git common directory.
`internal/leaseref/fence.go` compares and swaps one reference ID. The package
explicitly describes git synchronization as cross-machine visibility, with a
remaining simultaneous-acquisition race. `fak loop region --no-queue` checks the
reference view without taking a tree lease.

DOS leases live in the configured lane journal, normally
`<workspace>/.dos/lane-journal.jsonl`. The journal-path environment override takes
precedence over the configured path. DOS serializes its acquire/arbitrate/append
operation with its lane mutex. FAK dispatch documentation separately asks workers
to acquire DOS leases; publishing a FAK reference does not publish a DOS grant.

DOS's default `lease-lane live` view is a structural journal fold. It can include
unreleased old holders. Admission uses the canonical liveness-filtered view.
The strict-snapshot change under DOS #269 adds an opt-in
`lane_lease.live_leases(config, strict=True, expire_dead=True)` API. It serializes
the read with appenders and propagates I/O, UTF-8, JSON and non-object-record
failures. It reuses existing replay and liveness semantics. It is not a complete
schema or cryptographic integrity check, a tree acquisition, or a reservation
covering a later write. The default compatibility reader remains unchanged.

## Consequences for integration

Every edit adapter needs a common authority, a canonical workspace root and a
host-bound owner identity. Merely installing the same MCP names in two harnesses
does not establish those properties. An owner supplied in model-controlled tool
arguments cannot safely exempt a competing lease. Windows, WSL, symlinks and
managed worktrees must resolve to the intended common workspace while preserving
the edit's actual target.

A before-tool collision check can protect the covered tool path, but it leaves a
gap between checking and writing. Arbitrary shell writes, unwrapped processes,
expired holders resuming, and partitions require separate enforcement and real
write-boundary fencing. A coherent snapshot and per-ID compare-and-swap cannot
be combined into a cross-host atomicity claim. #11835 tracks that stronger
contract; #11834 tracks the native/MCP admission boundary.

This note records the baseline and the strict-read prerequisite. It does not
certify every installed harness, a restarted live MCP server, or universal
cross-host exclusion. Each integrated adapter needs its own executed witness:
held-tree refusal, disjoint/read-only success, exact owner behavior, release,
and unavailable-authority refusal.

## Same-session repair witnesses

The OpenCode native edit adapter's real fixture witness passed nine checks,
including conflicting reference/DOS leases, disjoint paths, reads, release and
corrupt DOS journal rejection. The coordinator independently executed that witness.
After the native binding repair, live DOS MCP arbitration refused the held gateway
lane. This establishes that operation's health without upgrading the original FAK
capability verdict into a lease-enforcement claim.

The installed native hook treated the exact `collaborationlist_agents` tool as
an unknown file footprint and warned on every observation. A narrow native alias
change recognizes list/wait, DOS doctor, and FAK read/tool-search calls. The
regression failed before the change and passed afterward; the hook package and
vet passed. The installed backend then changed that observation from advisory to
passthrough, while an identical proposed write into a held gateway tree retained
its collision warning. A subsequent live coordinator list-agents call emitted no
advisory. Only this user's active plugin binary was replaced, with a retained
backup; Python fallback alias coverage remains tracked in DOS #270. Spawn and
unknown tool names remain conservative.

This is also evidence against treating a held lease as sufficient protection:
while the OpenCode worker held its declared path lease, another worker swept the
plugin and test into a broader commit (`54af19455af8`). A lease needs enforcement
at every relevant write and landing surface. The incident reinforces #11835's
scope rather than demonstrating that universal ownership is solved.
