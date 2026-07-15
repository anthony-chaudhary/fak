# cmd lane split — the real collision bottleneck (#4320)

**Issue:** #4320 `fix(dispatch): split the cmd lane lease — the real collision bottleneck (~76% rate on low traffic)` · labels `dispatch`, `class:infra` · **OPEN**

**Status of this doc:** design + decomposition. Every load-bearing claim below was **adversarially re-verified** by an independent pass against the code + `.fak/loops.jsonl`; where verification refuted an earlier draft, the correction is kept inline so the epic lands on the *measured* mechanism, not a plausible-sounding one.

> **Two corrections the verification forced (read before acting):**
> 1. The phantom `internal/<scope>/**` lease tree is fabricated **in-repo** by fak's own `lane_tree()` fallback (`tools/issue_resolve_dispatch.py:3247`), **not** by the external `dos` kernel. So the lease-geometry fix is entirely fak-side — no `dos` change.
> 2. The dominant *measured* refusal on cmd/fak paths is **`DIRTY_PATH_COLLISION`, which is a pre-lease, text-only guard** (`dirty_path_collision(text, dirty_paths)`, `:3385`; runs at `:5559`, **before** the lease acquire at `:5643`; takes no lane/tree/lease). **A lease-geometry change therefore does NOT directly stop those refusals**, and a large fraction of them are *correct*. This doc's fix (Option C) targets lease-region fidelity / `LANE_LEASE_HELD` serialization — the thing #4320's title actually names — not the DIRTY_PATH guard.

---

## TL;DR / recommendation

`cmd = ["cmd/**"]` is one exclusive lease over **1251 `cmd/fak/*.go` files** across dozens of unrelated subsystems (`guard` 166, `dispatch` 115, `resume` 33, `info` 32, `issue` 27, `session` 25, …). Per the `dos.toml [lanes]` invariant every leaf's `cmd/fak/<leaf>.go` shim lives in the `cmd` lane, so cmd/fak is a flat co-tenant directory under one lease — two workers on unrelated cmd/fak files serialize on that single lease (`LANE_LEASE_HELD`). Splitting that lease is the real goal.

The literal task ("carve `cmd/**` into finer `[lanes.trees]` entries") is **blocked by three machinery constraints** (Blockers 1–3) — a naive edit reddens `dos lint` in `make ci`. Investigation then found a distinct **in-repo defect**: work routed to an undeclared scope-lane (`dispatch`, `guard-stops`, …) acquires a **phantom `internal/<scope>/**` lease region** that does not exist and does not cover the real `cmd/fak/<scope>_*.go` work site — so same-subsystem workers do not even serialize on a faithful region.

**But be precise about what a fix buys.** The `.fak/loops.jsonl` refusals break into three *independent* guards, and only some are a bug:
- **`LANE_LEASE_HELD`** (67 events, all on the phantom `internal/<scope>/**` cluster trees) — the serialization channel a lease split actually governs.
- **`DIRTY_PATH_COLLISION`** (22 events / 11 issues) — a **pre-lease, text-only** guard that fires when an issue body *mentions* a path that is dirty in `git status`. Independent of lease geometry. Its largest single cluster is **#4776 gateway → `internal/gateway/responses.go` ×6** — orphan WIP on a hot file in a lane with a *correct* tree, nothing to do with cmd. The cmd-family DIRTY_PATH events split into **own-scope TRUE collisions** (#4345 `guard_stops.go`, #4346, #4479 `dispatch_tick_preflight.go` — the worker really edits that dirty file; *correctly* refused) and **cross-lane citations** (#4347 dispatch→`guard_stops.go`, #3022 ci→`guard.go`, … — already independently caught by the `MULTI_LANE_SCOPE` guard, `:3269`).

**Recommendation:**
1. **Measure the decomposition first** (LANE_LEASE_HELD vs DIRTY_PATH vs MULTI_LANE_SCOPE; own-file-true vs cross-lane) over `.fak/loops.jsonl` before any code change. A chunk of these refusals are *correct* and must not be "fixed away."
2. **Fix the lease-region-fidelity defect (Option C, fak-side, no `dos` change):** make `lane_tree()`'s fallback (`:3247`) emit the real `cmd/fak/<scope>_*.go` glob for cmd-scoped scope-lanes. The arbiter treats two `cmd/fak/<a>_*.go` / `cmd/fak/<b>_*.go` globs as **disjoint** (verified, Blocker 2), so this de-serializes cross-subsystem work while faithfully serializing same-subsystem work — improving `LANE_LEASE_HELD` behavior. It does **not** target the DIRTY_PATH guard; expect only an indirect reduction there (cleaner serialization → shorter dirty windows).
3. Land behind the before/after measurement — it changes lease geometry for every undeclared scope-lane dispatch fleet-wide.

---

## Measured mechanism (ledger evidence)

`DIRTY_PATH_COLLISION` is the guard most visible on cmd/fak paths, e.g.:

```
#4347  lane=dispatch    tree=internal/dispatch/**     paths=cmd/fak/guard_stops.go   reason=DIRTY_PATH_COLLISION
#4345  lane=guard-stops tree=internal/guard-stops/**  paths=cmd/fak/guard_stops.go   reason=DIRTY_PATH_COLLISION
```

**What it is (verified):** `dirty_path_collision(text, dirty_paths)` (`:3385`) + `text_mentions_repo_path(text, path)` (`:3377`) take **only** the issue title+body text and the `git status` dirty set — no lane, tree, or lease — and the guard runs at `:5559`, **before** `acquire_lane_lease` (`:5643`). It fires when the body mentions any dirty path (the word-boundary regex matches `file.go:120` citations). So it protects *orphan WIP* and is structurally **independent of lease geometry** — no `lane_tree`/lease change can directly suppress it, and #4345/#4479-style events where the worker really edits the dirty file are *correct* refusals.

**Where the phantom lease tree comes from (all in-repo):** `:5643` `acquire_lane_lease(root, chosen_lane, tree=lane_tree(root, chosen_lane), …)` → `lane_tree()` (`:3227-3247`) returns `dos.toml [lanes.trees][lane]` if declared, else `return [f"internal/{lane}/**"]` (`:3247`). `dos.toml` declares no `dispatch`/`guard-stops`/`info`/`issue` lane (verified: those `internal/<scope>` dirs don't exist), so the fallback fires. The tree flows to the in-binary lease via `fak leaseref acquire --tree` (`:3106`; `cmd/fak/leaseref.go:513-533` writes `Record.TreeGlobs`) and the ledger (`:4135-4142`); `lane_kind:"cluster"` is a fak-side literal (`:4132`). The Go native path is symmetric (`cmd/fak/dispatch_tick.go:1161` `regionadmit.ResolveTree`). `dos` only returns declared trees via `dos doctor --json`; it never sets the held lease's tree.

---

## Why the naive "split `cmd/**`" in `dos.toml` is blocked

### Blocker 1 — `dos lint` (in `make ci`) rejects the sub-lane as a shadowed region
`make ci` runs `dos lint` (`Makefile:26`, target `:304-309`) with **default** severity gating (warn OR error; `make ci` deliberately omits `--strict`, `Makefile:299-301`), so a warn exits 1 and reddens the build. Adding `cmddispatch = ["cmd/fak/dispatch_*.go"]` while keeping `cmd = ["cmd/**"]` makes the former a **strict subset** (prefix `cmd/fak/dispatch_` ⊂ `cmd/`), so the lint's shadow branch fires first and emits **`LANE_REGION_SHADOWED`** (`config_lint.py:355-357`, `_shadow_finding` `:374-386`) — *not* `CONCURRENT_LANES_OVERLAP` (reserved for incidental intersection where neither region swallows the other, `:358-359`). Both are `Severity.WARN`, so either way the gate reddens.

### Blocker 2 — the arbiter's overlap is directory-prefix-only
`internal/dispatchorder/dispatchorder.go` `normalizeTree` (`:1115-1122`) strips trailing `/`, `/**`, `/*`; `treeOverlap` (`:1104-1113`) returns `a==b || HasPrefix(a,b+"/") || HasPrefix(b,a+"/")` — literal directory-prefix containment, no glob expansion. Verified traces:
- `cmd/fak/dispatch_*.go` vs `cmd/fak/guard_*.go` → **disjoint** (this is what makes Option C's per-subsystem leases not collide across subsystems).
- `cmd/fak/dispatch_*.go` vs `cmd/**`→`cmd` → **overlap**. So a residual `cmd` lane disjoint from carved sub-lanes cannot be expressed without enumerating filename prefixes.

### Blocker 3 — path→lane resolvers ignore filename-prefix globs; multi-match routes alphabetically
- `internal/devindex/devindex.go` `parseLanes` (`:283-290`) registers a prefix entry only for `**`-suffixed globs and an exact entry only for wildcard-free globs; `cmd/fak/dispatch_*.go` yields **neither**, so `LaneForPath` (`:521-547`) still resolves dispatch files to `cmd`. `internal/hooks/commitstamp.go` `laneForPath` (`:360-367`, `:396-425`) mirrors this byte-for-byte — a `(fak cmddispatch)` stamp would not diff-witness.
- `tools/issue_lane_router.py` `path_matches_lane` (`:486`) returns **all** matching lanes; on >1 match the router picks `sorted(routable_path_lanes)[0]` (`:787-792`) → `cmd` sorts before `cmddispatch`.
- Companion gap: `named_repo_paths`/`_PATH_RE` (`:469`) extract `cmd/...` only in the `fak/cmd/...` doc-link form; a **bare `cmd/fak/guard_stops.go` is matched only by `_BINDING_PATH_RE`**, which `named_repo_paths` does not use — so the router never path-confirms bare cmd/fak work to `cmd` and falls through to the scope token.

**Net:** the taxonomy machinery (arbiter + `dos lint` + devindex + hooks + router) assumes **disjoint `dir/**` or exact globs** with no most-specific-wins tiebreak. Filename-prefix sub-lanes carved from `cmd/**` are honored by none of it consistently — which is why the fix belongs at the **lease-tree fallback**, not in `dos.toml`.

---

## Options

### Option C — fix the in-repo `lane_tree()` fallback (recommended for the *lease-serialization* goal; no `dos` change)
Change `issue_resolve_dispatch.py:3247` (and the symmetric Go seam `regionadmit.ResolveTree`) so a cmd-scoped scope-lane with no declared tree falls back to **`cmd/fak/<scope-prefix>_*.go`** instead of `internal/<scope>/**`. Then `fix(dispatch):` leases `cmd/fak/dispatch_*.go`, `fix(guard-stops):` leases `cmd/fak/guard_*.go` — **disjoint** per Blocker 2, so cross-subsystem workers stop colliding on `LANE_LEASE_HELD` and same-subsystem workers serialize on a faithful region. It's a lease-tree fallback (not a declared `[lanes.trees]` lane), so **no `dos lint` finding** (Blocker 1 N/A) and no `dos.toml` edit. **Scope note:** this improves `LANE_LEASE_HELD` fidelity; it does **not** address `DIRTY_PATH_COLLISION` (pre-lease/text-only) — do not sell it as fixing the DIRTY_PATH events. **Open nut:** the scope-token→filename-prefix mapping (`guard-stops` → `guard_*.go`? `guard_stops*.go`?); pick granularity deliberately (broader = safer, narrower = more concurrency).

### Option A — move cmd/fak subsystem files into real subpackages
`cmd/fak/dispatch_*.go` → `cmd/fak/dispatch/`, then each sub-lane is a clean `cmd/fak/dispatch/**`. **Cost:** files are `package main`; a subdir is a distinct package → large refactor across ~1251 files. Only worth it if a *declared* lane split is required.

### Option B — teach the machinery filename-prefix sub-lanes + most-specific-wins
Add longest-glob precedence to `dispatchorder.treeOverlap`, `devindex.parseLanes`/`LaneForPath`, `hooks.laneForPath`, the router tie-break, and the `dos lint` shadow rule; then declare `cmd/fak/<prefix>_*.go` sub-lanes. **Cost:** coordinated change across 4 in-repo surfaces + `dos` lint. Only needed if the split must be a first-class **declared** lane (e.g. for `(fak <sub>)` ship-stamps). Option C does not require it.

---

## First checkable step

1. **Decompose the collision rate** over `.fak/loops.jsonl`: per-guard counts (`LANE_LEASE_HELD` / `DIRTY_PATH_COLLISION` / `MULTI_LANE_SCOPE`), and for DIRTY_PATH the own-file-true vs cross-lane split. This tells you how much of #4320's "~76%" a lease split can even move (only the `LANE_LEASE_HELD` share) vs what is a correct refusal or a different guard.
2. **Red witness (shipped with this doc):** `tools/issue_lane_router_test.py::PhantomTreeRegionFidelityTest` asserts the lease-region-fidelity invariant — an issue whose only work site is `cmd/fak/<scope>_*.go` must acquire a lease *region* that covers that file. It fails today (region is the phantom `internal/<scope>/**`) and passes when Option C lands (`@unittest.expectedFailure`; remove the marker then). It witnesses the *lease-geometry* defect only — not the DIRTY_PATH guard.
3. Decide the fallback glob granularity, then implement Option C behind the before/after measurement from step 1.

## Acceptance (from #4320)
- The `LANE_LEASE_HELD` share of cmd-family collisions drops materially, measured before vs after over `.fak/loops.jsonl`.
- No increase in cross-lane collisions, and no suppression of *correct* own-file DIRTY_PATH refusals.

## Do NOT
- Do **not** add `cmd/fak/<prefix>_*.go` sub-lanes to `dos.toml` while keeping `cmd = ["cmd/**"]` — it reddens `dos lint` via `LANE_REGION_SHADOWED` (Blocker 1) and still routes to `cmd` (Blocker 3).
- Do **not** claim Option C fixes the `DIRTY_PATH_COLLISION` events — that guard is pre-lease and text-only (`:3385`, runs `:5559` before the lease at `:5643`); a lease-geometry change cannot directly suppress it, and several of those events are correct refusals of real orphan-WIP collisions.
- Do **not** relax the `DIRTY_PATH`/`MULTI_LANE_SCOPE` guards to make refusals disappear — verified independently that this only reclassifies (cross-lane citations are already caught by `MULTI_LANE_SCOPE`) or unsafely admits workers stacking onto orphan WIP.
- Do **not** flip the `lane_tree()` fallback blind — it changes lease geometry for every undeclared scope-lane dispatch fleet-wide; land it behind the before/after measurement.
