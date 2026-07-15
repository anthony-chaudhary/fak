# cmd lane split — the real collision bottleneck (#4320)

**Issue:** #4320 `fix(dispatch): split the cmd lane lease — the real collision bottleneck (~76% rate on low traffic)` · labels `dispatch`, `class:infra` · **OPEN**

**Status of this doc:** design + decomposition. The *literal* task ("split `cmd = ["cmd/**"]` into finer sub-lanes in `dos.toml`") is **blocked by three machinery constraints** — a naive `[lanes.trees]` edit reddens `dos lint` in `make ci`. But investigation (claims below adversarially re-verified against the code + `.fak/loops.jsonl`) shows the **measured** bottleneck has a *different, entirely in-repo* root cause and a **sharp fak-side fix that needs no `dos.toml` change and no `dos` kernel change**. This doc records the verified mechanism, the blockers (with `file:line`), the fix options, and the first checkable step, so the epic lands correctly instead of chasing the wrong layer.

> **Correction (verified):** an earlier draft attributed the phantom `internal/<scope>/**` lease tree to the external `dos` kernel deriving a cluster tree from the `fix(<scope>):` prefix. That is **wrong**. The phantom tree is fabricated **in-repo** by fak's own `tools/issue_resolve_dispatch.py` `lane_tree()` fallback (line 3247), and `lane_kind:"cluster"` is a fak-side literal (line 4132). `dos` only returns `dos.toml`'s declared trees via `dos doctor --json`; it never sets the held lease's tree.

---

## TL;DR / recommendation

The `cmd` lane is one exclusive lease over **1251 `cmd/fak/*.go` files** spanning dozens of unrelated subsystems (`guard` 166, `dispatch` 115, `resume` 33, `info` 32, `issue` 27, `session` 25, …). Per the `dos.toml [lanes]` invariant, every leaf keeps its tree at `internal/<leaf>/**` and its `cmd/fak/<leaf>.go` shim **in the `cmd` lane** — so cmd/fak is a flat, shared, co-tenant directory under one lease.

But the **measured** dominant refusal is **not** the `cmd` lane lease. Over `.fak/loops.jsonl`, **9 of 9** refused events naming a `cmd/fak/*` path are **`DIRTY_PATH_COLLISION`**; zero are lane-lease reasons on cmd/fak paths. Two converging **in-repo** defects cause it:

1. **The router can't see cmd/fak work.** `issue_lane_router._PATH_RE` extracts `cmd/...` paths only in the `fak/cmd/...` doc-link form; a **bare `cmd/fak/guard_stops.go` is matched only by `_BINDING_PATH_RE`**, which `named_repo_paths` does not use. So a `fix(dispatch):` issue naming `cmd/fak/guard_stops.go` gets **no path-confirmation to `cmd`**, and routing falls through to the scope token → lane `dispatch`.
2. **The lease region is then phantom.** `lane_tree("dispatch")` finds no `dispatch` entry in `dos.toml` and returns the fallback **`internal/dispatch/**`** (`issue_resolve_dispatch.py:3247`) — a directory that does not exist and does **not cover** the real `cmd/fak/dispatch_*.go` work site. The arbiter (`laneadmit`/`dispatchorder.TreesOverlap`) therefore sees no overlap between two cmd/fak workers on disjoint phantom trees and admits both; the cruder working-tree **dirty-path guard is the only backstop that fires.**

**Recommended fix (Option C, verified in-repo, no `dos` change):** make `lane_tree()`'s fallback emit the real **`cmd/fak/<scope>_*.go`** glob for cmd-scoped scope-lanes instead of `internal/<scope>/**`. Because the arbiter treats two `cmd/fak/<a>_*.go` / `cmd/fak/<b>_*.go` globs as **disjoint** (verified below), this gives per-subsystem leases that *serialize same-subsystem work* while *not colliding across subsystems* — achieving the de-serialization #4320 wants **and** restoring region fidelity, entirely fak-side. Land it behind measurement (it changes lease geometry for every undeclared scope-lane dispatch fleet-wide).

---

## Measured mechanism (ledger evidence)

From `.fak/loops.jsonl` (`fak.loop-event.v1`):

```
#4347  lane=dispatch    tree=internal/dispatch/**     paths=cmd/fak/guard_stops.go
       reason=DIRTY_PATH_COLLISION
#4345  lane=guard-stops tree=internal/guard-stops/**  paths=cmd/fak/guard_stops.go,docs/nightrun/guard-stops.jsonl
       reason=DIRTY_PATH_COLLISION
```

Two different issues, routed to two different scope-lanes with two disjoint **phantom** trees, both name `cmd/fak/guard_stops.go`. They collide on the working-tree dirty path, not on lease geometry. Verified: `internal/dispatch`, `internal/guard-stops`, `internal/info`, `internal/issue` do **not** exist; the ledger tally is 9/9 `DIRTY_PATH_COLLISION` on cmd/fak paths.

**Where the phantom tree is set (all in-repo):** `issue_resolve_dispatch.py:5643` `acquire_lane_lease(root, chosen_lane, tree=lane_tree(root, chosen_lane), …)` → `lane_tree()` (`:3227-3247`) returns `dos.toml [lanes.trees][lane]` if declared, else `return [f"internal/{lane}/**"]` (`:3247`) → forwarded to the in-binary lease via `fak leaseref acquire --tree` (`:3106-3108`; `cmd/fak/leaseref.go:513-533` writes `Record.TreeGlobs`) → echoed to the ledger (`:4135-4142`), with `lane_kind:"cluster"` a literal at `:4132`. The Go native path is symmetric: `cmd/fak/dispatch_tick.go:1161` `regionadmit.ResolveTree(req, tax)`.

---

## Why the naive "split `cmd/**`" in `dos.toml` is blocked

### Blocker 1 — `dos lint` (in `make ci`) rejects the sub-lane as a shadowed region
`make ci` runs `dos lint` (`Makefile:26`, target `:304-309`), and `make ci` deliberately uses **default** `dos lint` (gates on **warn** OR error; `Makefile:299-301`) — a warn exits 1 and reddens the build. Adding `cmddispatch = ["cmd/fak/dispatch_*.go"]` while keeping `cmd = ["cmd/**"]` makes `cmd/fak/dispatch_*.go` a **strict subset** of `cmd/**`, so the lint's shadow branch fires first and emits **`LANE_REGION_SHADOWED`** (`config_lint.py:355-357`, `_shadow_finding` `:374-386`) — *not* `CONCURRENT_LANES_OVERLAP` (that's reserved for incidental intersection where neither region swallows the other, `:358-359`). Both are `Severity.WARN`, so either way the gate reddens.

### Blocker 2 — the arbiter's overlap is directory-prefix-only; a residual `cmd` can't be expressed
`internal/dispatchorder/dispatchorder.go` `normalizeTree` (`:1115-1122`) strips trailing `/`, `/**`, `/*`; `treeOverlap` (`:1104-1113`) returns `a == b || strings.HasPrefix(a, b+"/") || strings.HasPrefix(b, a+"/")` — literal directory-prefix containment, no glob expansion. Traced:
- `cmd/fak/dispatch_*.go` vs `cmd/fak/guard_*.go` → **disjoint** (neither is a prefix of the other). *This is why Option C's per-subsystem leases don't collide across subsystems.*
- `cmd/fak/dispatch_*.go` vs `cmd/**`→`cmd` **or** `cmd/fak/**`→`cmd/fak` → **overlap** (`HasPrefix("cmd/fak/dispatch_*.go", "cmd/")`). So a residual `cmd` lane disjoint from carved sub-lanes **cannot be expressed** without enumerating every filename prefix.

### Blocker 3 — the path→lane resolvers don't honor filename-prefix globs, and multi-match routes alphabetically
- `internal/devindex/devindex.go` `parseLanes` (`:283-290`) registers a prefix→lane entry **only** for `**`-suffixed globs and an exact entry **only** for wildcard-free globs; `cmd/fak/dispatch_*.go` yields **neither**, so `LaneForPath` (`:521-547`) still resolves dispatch files to `cmd`. `internal/hooks/commitstamp.go` `laneForPath` (`:360-367`, `:396-425`) mirrors this byte-for-byte — so a `(fak cmddispatch)` stamp would not diff-witness.
- `tools/issue_lane_router.py` `path_matches_lane` (`:486`) returns **all** matching lanes; on >1 match the router picks `sorted(routable_path_lanes)[0]` (`:787-792`). `cmd` sorts before `cmddispatch` → routes to the broad lane anyway.

**Net:** the taxonomy machinery (arbiter + `dos lint` + devindex + hooks + router) is built for **disjoint `dir/**` or exact globs** with no most-specific-wins tiebreak. Filename-prefix sub-lanes carved from `cmd/**` are honored by none of it consistently — which is why the fix belongs at the **lease-tree fallback**, not in `dos.toml`.

---

## Options

### Option C — fix the in-repo `lane_tree()` fallback (recommended, no `dos` change)
Change `issue_resolve_dispatch.py:3247` (and the symmetric Go seam `regionadmit.ResolveTree`) so that for a cmd-scoped scope-lane with no `dos.toml` tree, the fallback returns **`cmd/fak/<scope-prefix>_*.go`** instead of `internal/<scope>/**`. Then:
- `fix(dispatch):` → lease region `cmd/fak/dispatch_*.go`; `fix(guard-stops):` → `cmd/fak/guard_*.go` — **disjoint** per Blocker 2, so cross-subsystem workers stop colliding, and same-subsystem workers serialize on the shared region.
- It's a **lease-tree fallback**, not a declared `[lanes.trees]` lane, so it introduces **no `dos lint` shadow/overlap finding** (Blocker 1 does not apply) and needs no `dos.toml` edit.
- **Companion:** teach `named_repo_paths`/`_PATH_RE` to also extract bare `cmd/fak/...` paths, so the router can path-confirm the real work site. *Caution:* path-confirmation alone would route every cmd/fak issue to `cmd` (rank-5 `path-confirmed` > rank-4 `exact-scope`) and **re-serialize** the bottleneck — so the extraction fix must land *with* the sub-scope lease regions, not before them.
- **Open design nut:** the scope-token→filename-prefix mapping (`guard-stops` → `guard_*.go`? `guard_stops*.go`?). Pick the glob granularity deliberately; broader (`guard_*.go`) serializes more but is safe, narrower maximizes concurrency.

### Option A — move cmd/fak subsystem files into real subpackages
`cmd/fak/dispatch_*.go` → `cmd/fak/dispatch/`, etc., then each sub-lane is a clean `cmd/fak/dispatch/**`. **Cost:** the files are `package main`; a subdir is a distinct package → large refactor (exported seams, build wiring) across ~1251 files. Highest effort; only worth it if a *declared* lane split is required.

### Option B — teach the machinery filename-prefix sub-lanes + most-specific-wins
Add longest-glob precedence to `dispatchorder.treeOverlap`, `devindex.parseLanes`/`LaneForPath`, `hooks.laneForPath`, the router's multi-match tie-break, and the `dos lint` shadow rule; then declare `cmd/fak/<prefix>_*.go` sub-lanes in `dos.toml`. **Cost:** coordinated change across 4 in-repo surfaces + the `dos` lint. Medium effort; only needed if the split must be a first-class **declared** lane (e.g. for `(fak <sub>)` ship-stamps to diff-witness). Option C does not require it.

---

## First checkable step

1. **Red witness (in this repo, hermetic):** a test asserting the region-fidelity invariant #4320 must establish — *an issue whose only concrete work site is `cmd/fak/<scope>_*.go` must resolve to a lease region that covers that file.* It fails today (scope-lane region `internal/<scope>/**` misses the cmd/fak path) and passes when Option C lands. Shipped alongside this doc as `tools/issue_lane_router_test.py::PhantomTreeRegionFidelityTest` (marked `@unittest.expectedFailure` — a committed, non-breaking marker; remove the marker when Option C lands).
2. **Decide the glob granularity** for the `lane_tree()` fallback (per-file-prefix vs per-subsystem), then implement Option C behind a before/after measurement over `.fak/loops.jsonl`.

## Acceptance (from #4320)
- cmd-family per-lane collision rate (`DIRTY_PATH_COLLISION` + `LANE_LEASE_HELD` on cmd/fak paths) drops materially, measured before vs after over `.fak/loops.jsonl`.
- No increase in cross-lane collisions (the change follows real file-coherence boundaries, not prose scopes).

## Do NOT
- Do **not** add `cmd/fak/<prefix>_*.go` sub-lanes to `dos.toml` while keeping `cmd = ["cmd/**"]` — it reddens `dos lint` via `LANE_REGION_SHADOWED` (Blocker 1) and still routes to `cmd` (Blocker 3).
- Do **not** land the `_PATH_RE`/`named_repo_paths` cmd/fak extraction fix *alone* — without sub-scope lease regions it re-serializes every cmd/fak issue onto the `cmd` lease.
- Do **not** flip the `lane_tree()` fallback blind — it changes lease geometry for every undeclared scope-lane dispatch fleet-wide; land it behind the before/after measurement above.
