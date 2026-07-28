---
title: "Splitting the cmd lane lease (#4320)"
description: "Why one exclusive lease over 1251 cmd/fak files serializes unrelated workers, what the measured refusal really is, and the lease split that addresses it."
---

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

## Step 1 result — measured decomposition (2026-07-26)

Step 1 above is now **done**, and the answer changes the priority order. Method: one
single pass over `.fak/loops.jsonl` (5859 records; reason-bearing window
**2026-07-16 12:24 → 2026-07-26 05:36**), counting the `reason` field and, for
`DIRTY_PATH_COLLISION`, parsing the dirty path out of `summary` and the lane/tree/contract
paths out of `evidence_refs`.

| reason | events | share |
|---|---|---|
| `DIRTY_PATH_COLLISION` | 146 | **84.9%** |
| `MULTI_LANE_SCOPE` | 13 | 7.6% |
| `NO_LANE` | 5 | 2.9% |
| `ISSUE_CONTRACT_HOLD` | 4 | 2.3% |
| **`LANE_LEASE_HELD`** | **3** | **1.7%** |
| `SAME_ISSUE_WIP` | 1 | 0.6% |

The last-24h slice holds the same shape (37 / 5 / 1 / 1, zero `LANE_LEASE_HELD`).

**These counts do not match the 67 / 22 cited in the TL;DR**, and not by growth — the
lease figure moved *down* while the dirty figure moved *up*, so the two passes are reading
different corpora (a rotated ledger, or a cmd-family-only filter) rather than the same one
at different times. Whoever implements Option C should re-derive its own before-number from
the ledger it will be measured against; do not inherit 67.

**What this means for Option C:** nothing about its correctness — the lease-region-fidelity
defect is real and independently witnessed by `PhantomTreeRegionFidelityTest`. But its
addressable share of *current* dispatch blocking is **3 events**, so it cannot be justified
as a throughput fix on this ledger. It is a correctness fix; rank it as one.

### Where the blocking actually is: stale orphan WIP, and the guard is right

Ranking the 146 dirty refusals by the dirty path named:

```
87 blocks  cmd/fak/version_modules.go      (7.3 days stale)
11 blocks  internal/gateway/http.go        (3.6 days stale)
10 blocks  cmd/fak/guard.go                (fresh — peer mid-edit)
 6 blocks  internal/gateway/native_serve.go (6.6 days)
 5 blocks  cmd/fak/serve.go                (6.9 days)
 …26 distinct paths total
```

Splitting those by mtime: **125 of 146 (86%) name a path that has been dirty ≥2 days**
(abandoned orphan WIP, landable), and only **19** name a genuinely fresh peer edit
(a correct, transient refusal). 14 stale files carry the whole 125.

The single largest cluster is decisive about the mechanism. All **87**
`cmd/fak/version_modules.go` refusals were lane `docs`, lease tree
`docs/**,README.md,INDEX.md,llms.txt,…`, refused because the issue contract's `paths`
list named one cmd/fak file — and 85 of the 87 named *only* that file. The blocked issues
include **#2477 itself**, whose implementation *is* that dirty file: the work was refused
admission because its own half-finished diff was sitting in the tree.

This **reinforces the third Do-NOT below rather than weakening it.** The guard was not
misfiring; it was correctly protecting 8 days of abandoned work from being overwritten. The
fix was not to intersect the dirty set against the lane tree (that would admit a docs-lane
worker whose tree forbids the cmd/fak file it needs to edit — admitting a worker that
cannot work), and not to relax the guard. The fix was to **land the WIP**, which cleared
87 refusals' worth of blocking in one commit.

**Consequence for #4320's framing:** on this ledger, dispatch concurrency is gated by
orphan-WIP hygiene, not lease geometry. Lease geometry is the next constraint after that,
not the current one.

The ranking above was mechanical, so it is now a verb rather than a one-off:

```
fak wip blocked [--landable | --residue] [--stale-days N] [--ledger <path>] [--json]
```

It re-derives exactly this table from the ledger the guard already writes — parsing the
dirty paths out of the `DIRTY_PATH_COLLISION` / `SAME_ISSUE_WIP` summaries
(`wipattr.ParseBlockedPaths`), joining them against `git status`, and folding the result
through `wipattr.Rank`. Rows come back `LAND` (blocking, and the whole change set is
idle — the lever), `WAIT` (blocking, but the set is live: the refusal is *correct*),
`IDLE`, or `ACTIVE`; `--landable` prints just the queue and exits 3 when it is non-empty,
so a sweep or hook can branch on "there is a lever here" without parsing output.

Two properties are load-bearing, both learned the hard way while landing the WIP above:

- **Staleness is a property of the change SET, not the file.** `internal/adjudicator/
  reversibility.go` read 5.1 days idle and its package tested green, but the `Policy`
  field its untracked test referenced lived in a peer's `decide.go`, edited 30 minutes
  earlier. Landing the "stale" half alone would have put a test referencing a
  non-existent symbol on the trunk. `Rank` therefore classifies on the set's freshest
  member and names the sibling responsible, so a per-file mtime can never recommend
  committing half a change set. (`TestRankChangeSetPinsStaleFileLive` pins this.)
- **The cost figure is a lower bound.** The guard lists at most 8 dirty paths per refusal
  and elides the rest as `(+N more)`, so `blocks` under-counts wide refusals. That bias is
  the safe direction — it can only under-rank a lever, never invent one.

`--landable` is deliberately conservative in the same direction: a path whose mtime cannot
be read (a staged deletion, a vanished file) is treated as maximally FRESH, so work whose
staleness cannot be established is never offered for landing.

### Correction (2026-07-26): "land the WIP" is the right lever and the wrong instruction

The sentence above — *the fix was to land the WIP* — is what this doc told the next
implementer to do, and following it literally on the tree three days later would have
destroyed work. A third property turned out to be load-bearing, and it is not visible from
the path and mtime this ranking was originally built on:

- **Dirty is not the same as carrying work.** Path plus mtime prove a path is dirty. They
  say nothing about whether its bytes differ from what is already committed, and **three
  shapes are dirty while holding no new work at all — each destructive to land**: a stale
  INDEX entry (worktree already equals HEAD, staged at an older base a peer has since
  landed over — committing REVERTS the peer), a phantom DELETE (staged deleted while a
  byte-identical file sits on disk — committing deletes live code; git reports this as
  *two* porcelain lines, `D  path` and `?? path`, so a deduplicated path set loses the
  signal), and content already LANDED UPSTREAM.

Both of the first two were live on this tree when it was measured. `internal/agent/loop.go`
ranked with **8 blocked admissions** while its index held the pre-#5235 blob, so landing it
would have reverted `66e132fbf`; `internal/gateway/role_alternation.go` and its test were
staged deleted with byte-identical 14580/10650-byte files on disk, so landing them would
have removed **632 lines the trunk still has**. Under the original path-and-mtime ranking
all three read as ordinary landable WIP.

`Rank` now takes a `Content` dimension the caller probes and returns a **`RESIDUE`** state
that pre-empts every other verdict — age can neither promote nor excuse a stale index
entry, since a fresh one is exactly as destructive to commit as an abandoned one. RESIDUE
ranks directly below LAND because it is the *cheapest* lever on the board: those admissions
come back by clearing the entry, with nothing committed and nothing reviewed.
`ResidueBlocks` is reported separately from `BlocksRecovered` precisely so the two cannot be
summed into a number that advertises a revert as throughput.

The probe costs two whole-tree git reads regardless of dirty-set size
(`git diff --name-only HEAD` and `... @{upstream}`) and **fails toward the OLD ranking**: a
failed HEAD read leaves every path `ContentUnprobed`, reproducing the pre-`Content` verdict
exactly. An emptied queue would *hide* real work, which is the worse failure — the same
degrade-to-the-truth-you-have rule the ledger read already uses. Known gap, deliberate: an
untracked file whose bytes already exist upstream still reads as work, because neither diff
read covers untracked paths.

Measured on this tree after the change: **164 dirty, 34 RESIDUE** (3 stale-index,
2 phantom-delete, 29 landed-upstream) carrying **19 admissions recoverable without any
commit**, and **0 rated landable**. Note what that last number means for the orphan-WIP
lever this doc identified: it is not inexhaustible. Once the genuinely stale, genuinely
divergent WIP has been landed, what remains looks identical to it and is not.

## Acceptance (from #4320)
- The `LANE_LEASE_HELD` share of cmd-family collisions drops materially, measured before vs after over `.fak/loops.jsonl`. **Re-derive the before-number** (see Step 1 result — the 67 in the TL;DR is not reproducible on the current ledger).
- No increase in cross-lane collisions, and no suppression of *correct* own-file DIRTY_PATH refusals.

## Do NOT
- Do **not** add `cmd/fak/<prefix>_*.go` sub-lanes to `dos.toml` while keeping `cmd = ["cmd/**"]` — it reddens `dos lint` via `LANE_REGION_SHADOWED` (Blocker 1) and still routes to `cmd` (Blocker 3).
- Do **not** claim Option C fixes the `DIRTY_PATH_COLLISION` events — that guard is pre-lease and text-only (`:3385`, runs `:5559` before the lease at `:5643`); a lease-geometry change cannot directly suppress it, and several of those events are correct refusals of real orphan-WIP collisions.
- Do **not** relax the `DIRTY_PATH`/`MULTI_LANE_SCOPE` guards to make refusals disappear — verified independently that this only reclassifies (cross-lane citations are already caught by `MULTI_LANE_SCOPE`) or unsafely admits workers stacking onto orphan WIP.
- Do **not** land a dirty path because it ranks stale and blocks admissions — check its
  `content` first (`fak wip blocked --json`, or just read the `RESIDUE` rows). A stale index
  entry, a phantom delete, and a landed-upstream copy all look exactly like abandoned WIP
  under path-and-mtime, and committing any of them reverts a peer or deletes live code. See
  the 2026-07-26 correction above.
- Do **not** flip the `lane_tree()` fallback blind — it changes lease geometry for every undeclared scope-lane dispatch fleet-wide; land it behind the before/after measurement.
