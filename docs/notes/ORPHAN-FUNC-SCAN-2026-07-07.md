---
title: "Orphan-func scan: catching 'built but never wired up' (2026-07-07)"
description: "A pure syntactic scan (internal/orphanscan) flags unexported functions referenced nowhere; a dogfood run found 12, including inert cap-crash recovery."
---

# Orphan-func scan: catching "built but never wired up" (2026-07-07)

## The failure mode

Go does not complain about an unexported top-level function that nothing calls. Unused
imports and unused locals are compile errors; an unused **function** is silent. So the
shape a dropped piece of work takes — a handler written but never added to the dispatch
switch, a recovery path authored beside its twin but never given its two call sites, a
provider variant that supersedes another but is never installed — compiles green and
ships inert. A green build is a self-report; an orphaned function is one place a green
build hides work that isn't connected to anything.

## The structure: `internal/orphanscan`

A small, pure, tier-1 syntactic pass (stdlib `go/parser` only, no types, no build, no
network) that flags **unexported, receiver-less, top-level funcs whose name is
referenced nowhere in their own package** — across every `.go` file including tests, so
a test-only helper is never flagged. It parses each file independently, so a package
that does not fully compile (normal in a shared tree with in-flight siblings) is still
scanned.

**Precision over recall, by design.** It would rather miss a real orphan than cry wolf.
Only unexported funcs (an exported symbol may be API; a method may satisfy an interface
and dispatch dynamically). References are counted as real AST identifiers, so a comment
or string that merely *names* the func does not count as a use — the exact place a
naive `grep` over-counts. Self-recursion counts as a use (a conservative false
negative). Excluded up front: `main`, `init`, blank `_`, generated files, anything with
`//go:linkname`/`//export`, and anything a maintainer marks with the visible escape
hatch `//orphanscan:keep <reason>`.

Run the dogfood scan over the live tree:

    go test ./internal/orphanscan/ -run TestDogfoodRealRepo -v

It **logs** candidates without failing — turning a finding into a red build is a gate
decision that lives elsewhere; the scan never blocks the tree on the messy real world.

## Validation: 12/12 true positives, zero false positives

The first dogfood run flagged 12 funcs in `cmd/fak` (0 in `internal/gateway`). Every one
was cross-checked against `grep`: the six that `grep` counted twice each had their
second hit in the func's **doc comment**, not a call site — the AST scan correctly
ignored those, where `grep` would have false-cleared them. All files parsed (no
parse-skips hiding call sites). So all 12 are genuinely unreferenced.

## Findings, triaged

### A. Confirmed dropped-wiring regressions — restore the call sites

- **`guardMaybeRecoverCapCrash`** (`cmd/fak/guard_cap_park.go:264`). Its own doc says it
  is "wired at the SAME two recovery sites (`runGuardChildAndReport` /
  `runGuardChildSupervisedAndReport`) as its counterpart `guardMaybeRecoverAuthCrash`."
  The counterpart **is** wired — called at `cmd/fak/guard_child.go:1020` and `:1082`.
  The cap-crash function is called **nowhere**. Cap-crash auto-recovery is inert: the
  binary cannot auto-recover from a cap crash even though the code to do so exists and
  is (presumably) unit-tested in isolation. Fix is the two-line insertion beside its
  auth twin at those two sites.

- **`newGuardFleetProviderMaybeSpine`** (`cmd/fak/guard_spine.go:33`). Doc: "the fleet
  provider `fak manage` installs." No caller. So `fak manage` installs the plain
  disk-only provider and the live multicast self-discovery **spine wrapper is
  unreachable** from the entrypoint. Caveat: `guard_spine.go` is part of an active
  in-flight `gateway.SessionFleet` sibling refactor (with `guard_fleet.go` /
  `info_fleet.go`), so this orphan may be transient churn that resolves when that
  refactor lands its new install site — re-run the scan after it settles rather than
  acting now. (Contrast §A's cap-crash finding, whose file is *not* in that break.)

### B. Intentionally unwired — leave as-is (or add the `//orphanscan:keep` note)

- **`cmdBenchIngest`** (`cmd/fak/benchingest.go:32`). Its comment gives the exact recipe
  to wire it (`case "bench-ingest": cmdBenchIngest(os.Args[2:])`). Deliberate
  scaffolding, not a regression.

### C. Unreachable verb / helper candidates — triage (wire or delete)

Handlers with no dispatch case, so the verb is currently unreachable:
`cmdBalance` (`balance.go:37`), `cmdTasks` (`tasks.go:16`),
`cmdCIPreflight` (`ci_preflight.go:40`), `cmdFocusScore` (`focusscore.go:13`),
`cmdGuardSessions` (`guard_sessions.go:27`), `indexCtxPlans` (`ctxplans.go:19`, an
`fak index` sub-handler with no case in the index sub-dispatch). Plus two helpers with
no call site: `guardKV` (`guard_format_layout.go:106`),
`witnessArchiveDefault` (`worktree.go:208`). Each is either a verb to finish wiring or
dead code to delete — a human/owner call per item.

## The ablation thread: no regression there

The originating worry touched an info panel and ablations. Confirmed clean:
`cmd/fak/info_panels.go` and `guard_info*.go` contain **no** `ablat` reference — the info
panels never carried ablations. The ablation surface is the `fak ablate` verb family
(`cmd/fak/ablate.go`), which is fully wired (`cmdAblate` is **not** in the orphan list).
So nothing regressed on the ablation front; the real dropped wiring is cap-crash
recovery (§A).

## Why the `cmd/fak` fixes were not applied here

`cmd/**` is held **exclusive** by a fleet loop, and `guard.go` / `guard_child.go` are
actively modified in the working tree — exactly where the cap-crash call sites live.
Restoring that wiring means editing the file the loop is writing. This note surfaces the
finding with exact locations so the owning loop (or a human) can restore it without a
collision. The reusable structure (`internal/orphanscan`) is what ships from this
session; it will catch the next dropped wiring automatically.

---

## Follow-up (2026-08-06): the scan became a gate (#3167)

The note above ships a scanner whose only caller was a `t.Logf`-only test — it found
orphans and then ignored them, which is a report, not a gate. [#3167] closes that: the
dogfood test now **asserts**, so a newly orphaned func reds the tree.

### Ratchet, not cliff

A scan over a real repo reports standing debt as well as fresh mistakes, so asserting
on a raw finding count either reds the trunk on day one or gets tuned down to a number
nobody can defend. The gate instead diffs against a checked-in baseline. Two
suppression tiers, in the order you should reach for them:

1. **`//orphanscan:keep <reason>`** on the func's doc comment — local, greppable,
   travels with the code, visible to whoever reads the definition. Honored inside
   `ScanDir`, so a kept func never even reaches the gate. Prefer this.
2. **`internal/orphanscan/keep_allowlist.txt`** — for what tier 1 cannot express: a
   func whose package is held by another loop's lane lease (so the source cannot be
   edited at all right now), plus the standing baseline that makes the ratchet possible.
   Format is `<package-dir> <func-name> # <reason>`; the **reason is required** and a
   missing one is a parse error, because an unexplained suppression is how a ratchet
   quietly becomes a rubber stamp.

Exemptions key on **package + func name, never file:line** — a func that moves between
files in its package, or just drifts down its file as neighbours grow, is the same
exemption and must not silently lapse.

### Running it

The gate is the named test, the same opt-in shape as `internal/windowgate` — a command
a human or a CI job invokes deliberately, not a new default-on `make ci` rung:

```
go test ./internal/orphanscan -run TestDogfoodRealRepo -v
```

It judges the packages in `dogfoodPkgs` (`cmd/fak`, `internal/gateway`) — deliberately a
short named list, not the whole tree, since widening to packages nobody has baselined
would red the trunk with pre-existing debt. On a finding it prints the `ORPHAN_FUNC`
line **and** the exact remedy, including the allowlist line to paste.

### Two degradations that keep it from crying wolf

Both matter because this is a shared tree with in-flight siblings:

- **A package with an unparseable file is not judged.** References are counted per file,
  so a file that will not parse contributes none — and if it held a func's only call
  site, that func looks orphaned when it is not. `ScanDirReport` reports `Unparsed` so
  the gate can see the input was incomplete and decline rather than fail on a mid-save
  neighbour. (`ScanDir` keeps its original signature for report-only callers such as
  `internal/antipattern`.)
- **A stale allowlist entry logs, never fails.** That case fires when someone *fixed* an
  orphan; pruning the line is the fix landing, and a peer should not red the tree for
  improving it. Staleness is judged only for packages the run actually scanned, so a
  partial scan never claims the rest of the file is dead.

### Baseline at landing: 4

`cmdGitSpawnBench` and `dispatchPreflightChurn` (redundant wrappers whose live callers
use the `run*` / `...State` form), `isMidflightVerb` (classifier never consulted by the
route it was written for), and `cmdRunaway` — the interesting one: `cmd/fak/main.go` has
no `case "runaway":`, yet `cmd/fak/runaway.go` re-execs `<self> runaway --json`, so that
self-call cannot resolve. Each is a live finding for its owning lane, recorded with that
reasoning in the allowlist.

Every §A/§B/§C finding from the original run above is **gone** from the current scan —
wired or deleted in the intervening month, including `cmdBenchIngest`, which #3167 named
as the exemplar allowlist case. It is deliberately *not* listed: `main.go` now dispatches
`case "bench-ingest"`, so allowlisting it would create a stale entry on day one.

### Not landed here

The `fak orphanscan` CLI verb #3167 also asks for. It needs `cmd/fak/orphanscan.go` plus
a `main.go` dispatch case, and `cmd/**` was held exclusive by another loop's lease
(`claude-5031`) — the same collision the section above hit a month earlier, so treat it
as the standing condition for this lane rather than bad luck. The `Gate` /
`SuppressionHint` / `ScanDirReport` seams are shaped for it: the verb is a thin
`os.Exit(runOrphanscan(...))` shell in the `cmd/fak/stallscan.go` mould over exactly
those three calls, with no further work in this package.

[#3167]: https://github.com/anthony-chaudhary/fak/issues/3167
