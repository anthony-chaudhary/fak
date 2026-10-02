<!-- fak-dualsync-key: workspace-scratch-sprawl-2026-09-29-03 -->
# perf(witness): witness and candidate checkouts materialize the full working tree; reuse the block-clone / --no-checkout / pool machinery

```routing
lane: witness
paths: ["internal/witness/execution.go", "internal/workerworktree/land.go", "internal/workerworktree/blockclone.go", "internal/workerworktree/workerworktree.go", "internal/workerworktree/pool.go"]
expected_steps: 5
priority: P1
```

The conventional subject is the upstream commit subject and carries context
`(fak internal/workerworktree)`.

## Context

Two producers create a *fresh full checkout per invocation* beside the workspace
parent: the execution-witness scratch worktree
(`internal/witness/execution.go:336-362`, prefix `fak-exec-witness-*`) and the
land candidate (`internal/workerworktree/land.go:1246-1268`, prefix
`.fak-cand-validate-*`). Both shell out to a raw
`git worktree add --detach <dir> <ref>`, which writes every tracked byte into
the new tree.

Measured one surviving witness copy, `fak-exec-witness-<id>`: 250 MB / 15,249
files, objects shared via its gitdir (so ~95% of the bytes are checked-out
*tracked* content, not git data). `docs/` 129 MB (incl.
`docs/research/inventory/*.jsonl` 24 MB, `docs/benchmarks/receipts` 34 MB,
`docs/media` 19 MB, `docs/tensor-build-extraction` 16 MB), `artifacts/` 15 MB,
and a root media file ~5.7 MB. The same tree is paid again on every witness run
and every land.

## Problem frame

The bytes are not un-reap-able junk; they are a **materialization strategy**.
Per copy, the producer asks git for a complete working tree even though the
consumers (a red-then-green rebuild/test, and a verify of the same candidate)
need the tree to be *correct*, not *eagerly written*. The repository already
owns three cheaper strategies, but neither producer calls any of them:

- **ReFS/APFS/Linux block-clone backend** —
  `internal/workerworktree/blockclone.go` (whole-tree native clone at `:36`
  `CloneTree`, backed by `materialize` at `:194`; `:209` already does
  `git worktree add --detach --no-checkout` then clones the tree; `:332`
  `materializeTreeClone` is the fast-path body). Selected by
  `workerworktree.go:141` `defaultIsolationBackend = nativeIsolationBackend()`,
  the block-clone backend by default on Windows (`blockclone_windows.go:22`).
- **Phased no-checkout provisioning** — `git worktree add --no-checkout` then
  `reset --hard` (`workerworktree.go:839-842`): bounds the registration write
  separately from the populate and survives a slow checkout.
- **Warm pool** — `internal/workerworktree/pool.go`
  (`FLEET_WORKER_WORKTREE_POOL`; `pool.go:350` `reserveAndResetLocked`), which
  amortizes a materialized tree across dispatches.

Sparse checkout is absent from the repo entirely (no `core.sparseCheckout`
anywhere under `internal/`); a possible additional lever, not the first one.

## Exact file:line seams

- `internal/witness/execution.go:349` —
  `os.MkdirTemp(parent, "fak-exec-witness-*")`; `:357`
  `git worktree add --detach --quiet dir ref` — the raw full checkout and the
  sole materialization path for the execution witness.
- `internal/workerworktree/land.go:1258` —
  `os.MkdirTemp(parent, topologyCandidatePattern())`; `:1266`
  `git worktree add --detach candDir ref` — the raw full candidate checkout
  (plus the `git apply` of `prospectiveDiff` at `:1275`).
- `internal/workerworktree/blockclone.go:36` (`CloneTree`), `:194`
  (`materialize`), `:209` (`--no-checkout` inside the backend), `:332`
  (`materializeTreeClone`): the existing fast path both producers should use.
- `internal/workerworktree/workerworktree.go:141` (`defaultIsolationBackend`),
  `:839-842` (no-checkout + checkout/reset fallback): backend selection and the
  phased shape to reuse rather than re-derive.
- `internal/workerworktree/pool.go:350` (`reserveAndResetLocked`):
  the amortization seam if a persistent witness/candidate root is acceptable
  (not the default; see Quarantined fallback).

## Reproduction witness

```
# pick one surviving witness copy and attribute its bytes
du -sh    <workspace-root>/fak-exec-witness-<id>                 # 250M
find      <workspace-root>/fak-exec-witness-<id> -type f | wc -l # 15249
du -sh    <workspace-root>/fak-exec-witness-<id>/docs            # 129M
git -C <consumer-repo> worktree list | grep <id>                 # objects shared via gitdir:
```

Because the git objects are shared, essentially every measured byte is
re-written tracked content. A producer that materializes via
block-clone/`--no-checkout`+pool should drop the per-copy figure by an order of
magnitude on a ReFS host (Windows default); report the before/after bytes for
one fixture, not just assert it.

## Intended fix direction (do not over-prescribe)

Route the witness scratch and the land candidate through the cheap machinery
rather than a raw `git worktree add`:

1. Prefer the block-clone backend (default on Windows) via the same call shape
   `blockclone.go` uses (`add --detach --no-checkout` + clone), or
2. fall back to the phased `--no-checkout` + `reset --hard` shape from
   `workerworktree.go:839-842`, and/or
3. bound a sparse-checkout set excluding the pure-content trees
   (`docs/research/inventory`, `docs/benchmarks/receipts`,
   `docs/tensor-build-extraction`, `docs/media`, `artifacts`, root media files).

Keep the sibling parent (`filepath.Dir(repoRoot)`) — required because a
downstream consumer repository's committed `go.work` references a sibling module
via `../` and the module graph breaks otherwise. Preserve correctness: the
verified ref's tree must still be complete for whatever builds; the exclusion
set is content-only and must be proven harmless by the witness run.

Sub-finding to **note, not fix here**: the multi-MB media that dominates the
copy is tracked with no `filter=lfs` attribute (`.gitattributes` has no `lfs`
line), so every checkout writes full bytes. A later ticket (`.gitattributes` +
LFS migration) is the root fix; this ticket avoids re-materializing it.

## Definition of done

- [ ] `scratchWorktree` and `verifyTopologyCandidate` no longer perform an
      unconditional full `git worktree add --detach` when a cheap path is
      available on the host/filesystem.
- [ ] The block-clone backend (or phased no-checkout + reset, or a bounded
      sparse set) is actually invoked — witnessed via the existing
      `TreeCloneFastPathCount()` / a focused test, not by inspection.
- [ ] Measured per-copy bytes for one fixture reported before/after in the
      commit message (target: well under the ~250 MB baseline on a ReFS host).
- [ ] Sibling parent preserved; the materialized tree hash equals the ref's
      tree for the included paths.
- [ ] `go test ./internal/workerworktree/... ./internal/witness/... -count=1`
      green.
- [ ] No new daemon, no reaper/TTL change; no private path imported.
- [ ] Committed in the public repo (`(fak internal/workerworktree)`); receipt.

## Blast radius

Public `internal/witness` and `internal/workerworktree` only (public-native
runtime). Touches the shared materialization seam that TICKET-01 also routes
through; land them in the same lane or isolate with `fak-flow start`. No private
import, no cross-repo commit.

## Quarantined fallback

If block-clone routing changes `git apply` or ref-verification semantics, keep
the raw checkout for the candidate and ship the witness half plus the bounded
sparse-checkout set. If sparse checkout perturbs a witness build, drop it and
keep only the clone/no-checkout routing. Do not adopt the warm pool for the
witness scratch unless the tree can be proven reset clean between refs.

## Witness

```
go test ./internal/workerworktree/... ./internal/witness/... -count=1
```

plus the measured `du -sh`/file-count before/after for one fixture, cited in the
commit message.

## Lane

`witness`. Isolate with `fak-flow start` if TICKET-01 lands concurrently against
the same materialization seam.

## Done condition

Both producers materialize via a cheaper existing path with a green
`./internal/workerworktree/... ./internal/witness/...` run, the per-copy byte
reduction is measured and cited, and the commit (carrying
`(fak internal/workerworktree)` and provenance trailers if code and tests land
together) is pushed.
