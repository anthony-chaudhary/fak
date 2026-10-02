<!-- fak-dualsync-key: workspace-scratch-sprawl-2026-09-29-02 -->
# fix(workerworktree): the `.fak-cand-validate-*` sweep only runs on a land and only in one parent, so killed lands leak candidates into the workspace root forever

```routing
lane: workerworktree
paths: ["internal/workerworktree/candidate.go", "internal/workerworktree/land.go", "cmd/fak/worktree_worker_reap.go"]
expected_steps: 5
priority: P1
```

Process cause: lifecycle · **Lane:** workerworktree (public,
native-runtime/gate-automatic) · **Work unit:** collector placement +
eligibility, not the producer.

## Context

`verifyTopologyCandidate` (`internal/workerworktree/land.go:1246-1268`)
materializes a verify-only detached worktree beside the selected root and removes
it in a `defer`. A kill (`taskkill /T /F`, agent tool timeout) skips the defer
(see `candidate.go:15-38`), so every killed land leaves a full checkout plus its
git worktree registration. A downstream consumer repository whose committed
`go.work` references a sibling module via `../` sets
`workspaceNeedsSiblingTopology` (`land.go:1286-1300`) true, so the candidate
parent is `filepath.Dir(<consumer repo>)` — the workspace root, shared by every
repo and agent session.

## Problem frame

The only *automatic* collector is `sweepTopologyCandidatesBeforeCreate`
(`candidate.go:391-409`). Four properties make it structurally unable to drain
the leak:

1. **It runs only on a land.** It is called from exactly one site,
   `verifyTopologyCandidate` (`land.go:1257`). If no land is running, nothing
   collects — the host candidates are exactly the residue of ended lands.
2. **It scans one parent.** `SweepTopologyCandidates(root, parent, ...)`
   (`candidate.go:152-217`) reads exactly the `parent` it is handed. A candidate
   can live in `root`, in `filepath.Dir(root)` when sibling-topology, or in
   `os.TempDir()` (`TopologyCandidateParent`, `candidate.go:136-145`); no single
   call covers every parent.
3. **Its budget cannot drain a backlog.** Rate-limited to
   `candidateSweepInterval = 10m` (`candidate.go:49`) with
   `candidateSweepLandLimit = 2` (`candidate.go:53`): at most two reaps per
   parent per ten minutes — a 15-item backlog needs ~8 lands. Legacy names wait
   `LegacyCandidateMaxAge = 2h` (`candidate.go:45`), so a legacy candidate
   created after a session's last land is only eligible the next working day.
4. **A stale registration can outlive its removed directory.** `pruneDirs`
   (`candidate.go:183`) is keyed on `candidateCommonGitDir` (`:342-361`), which
   reads the candidate's own `.git`. A de-registered or emptied candidate dir
   has no `.git`, so `candidateCommonGitDir` returns `""`, root's common dir is
   never pruned, and the `git worktree list` entry survives the directory.

The manual, unbounded verb already exists — `fak worktree worker gc --candidates
--apply` (`cmd/fak/worktree_worker_reap.go:85-138`) — but it is operator-driven
and inherits the single-parent placement (`:122-124`).

## Exact file:line seams

- `internal/workerworktree/land.go:1257` — sole caller of
  `sweepTopologyCandidatesBeforeCreate`; the "only on a land" gate.
- `internal/workerworktree/candidate.go:391-409` — the bounded, rate-limited
  sweep: `candidateSweepInterval` (`:49`), `candidateSweepLandLimit = 2` (`:53`),
  one `parent`.
- `internal/workerworktree/candidate.go:136-145` — `TopologyCandidateParent`:
  one parent, `os.TempDir()` unless sibling-topology.
- `internal/workerworktree/candidate.go:152-217` — `SweepTopologyCandidates`:
  single `parent`; `pruneDirs` (`:183`) fed only by `candidateCommonGitDir`;
  `sweepLockedCandidateRegistrations` (`:274`) covers only `root`'s common dir.
- `internal/workerworktree/candidate.go:198-204` — owner-named candidates reap
  only when the owner pid is gone (`candidateOwnerState`, `:223-233`); an
  unparseable name waits `LegacyCandidateMaxAge` (`:45`).
- `internal/workerworktree/candidate.go:342-361` — `candidateCommonGitDir`
  returns `""` for an empty/de-registered dir, dropping the prune.
- `cmd/fak/worktree_worker_reap.go:122-124` — manual collector, one parent.

## Reproduction witness

Host, 2026-09-28: **15** `.fak-cand-validate-*` sat in the workspace root, some
legacy-named, some empty. At authoring, 4 remain (rest reclaimed by hand). A
legacy sample is a bare numeric suffix:

```
$ ls -d <workspace-root>/.fak-cand-validate-* | wc -l      # 4
drwxr-xr-x ... 1894488361   (163 entries, legacy name — waits 2h)
drwxr-xr-x ... 1907036884   (1 entry, empty/de-registered)
drwxr-xr-x ... 2872256751   (117 entries, legacy name)
drwxr-xr-x ... 3130403981   (163 entries, legacy name)
```

`parseTopologyCandidateOwner(".fak-cand-validate-1894488361")` returns `false`
(two `-` fields, not three — `candidate.go:77-95`), so these are legacy and only
a 2-hour `LegacyCandidateMaxAge` sweep can take them. The lands that produced
them have long exited, so `sweepTopologyCandidatesBeforeCreate` runs again only
on the next land — and then only twice.

## Definition of done

1. The collector is **independent of a landed create**: sweeping every candidate
   parent happens on a normal land and on the operator verb, not only inside
   `verifyTopologyCandidate`.
2. It is **aware of every parent that can hold a candidate**: `root`,
   `filepath.Dir(root)` when `workspaceNeedsSiblingTopology(root)`, and
   `os.TempDir()` — deduplicated by absolute path, each swept.
3. The land-time budget drains a backlog (raise `candidateSweepLandLimit` and/or
   drop `candidateSweepInterval` for the multi-parent pass); the explicit
   `gc --candidates` verb stays unbounded.
4. Removing a candidate dir **always prunes root's common git dir**, including
   when the dir had no `.git` (empty/de-registered case).
5. No daemon is added; the existing land and `gc` paths are extended.

## Blast radius

`internal/workerworktree` is public-native runtime, gate-automatic. Touching
`candidate.go` changes the selection set (more parents, larger limit) every land
exercises; `land.go` only moves the call site; `worktree_worker_reap.go` is the
operator surface. Risk: a wider sweep removing a live owner's candidate. The
fail-toward-keeping rules (`owner_process_live`, unreadable start = live,
`legacy_too_young`) must be preserved verbatim — this changes placement and
budget, not liveness semantics.

**Quarantined fallback:** if a multi-parent pass cannot be exercised safely
during a test, add the second parent behind `candidateSweepParents(root)` and
keep `gc --candidates` single-parent; the prune-on-remove fix lands
independently.

## Witness

```
go test ./internal/workerworktree/... -count=1
```

with a new case: a **legacy** candidate (bare suffix, e.g.
`.fak-cand-validate-1894488361`) placed in the workspace parent
(`filepath.Dir(root)` for a sibling-topology repo) owned by no live process, an
age past `LegacyCandidateMaxAge`, is reported eligible and removed by the
multi-parent sweep, and its root-common-dir registration is pruned even after
the candidate's `.git` file is absent.
`TestSweepTopologyCandidatesKeepsLiveAndYoungLegacy`
(`internal/workerworktree/candidate_test.go:110`) must keep passing unchanged.

## Lane

`workerworktree`, public only — no private import.

## Done condition

`go test ./internal/workerworktree/... -count=1` green including the new
legacy-in-workspace-parent case; `gofmt`/`go vet` clean; committed on the public
trunk. A killed land must no longer be able to leave a `.fak-cand-validate-*` in
the workspace root past the next land's sweep.
