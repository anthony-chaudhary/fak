<!-- fak-dualsync-key: workspace-scratch-sprawl-2026-09-29-01 -->
# fix(witness): make the sibling-topology exec-witness scratch dir crash-safe, owner-named, and swept before creation

```routing
lane: witness
paths: ["internal/witness/execution.go", "internal/witness/*_test.go"]
expected_steps: 4
priority: P1
```

The conventional subject is the upstream commit subject and carries context
`(fak internal/witness)`.

## Context

`ExecutionVerifier.scratchWorktree` materializes a detached worktree so a
red-then-green execution witness can run against a committed ref. When the
committed `go.work` escapes the repo root (a downstream consumer repository
whose committed `go.work` references a sibling module via `../`), the checkout
must sit beside its sibling or the module graph breaks, so the parent is
`filepath.Dir(repoRootAbs)` — the workspace root shared by every repository and
agent session.

On a Windows fleet host, 100+ `fak-exec-witness-*` directories have accumulated
beside the repos, one measured at 250 MB / 15,248 files. Two independent defects
produce the leak, and the creation seam currently lacks every safety property
the sibling `internal/workerworktree` candidate seam already has.

## Problem frame

Three properties are missing at the creation seam:

1. **Crash-unsafe.** Cleanup is a `defer`-returned closure. A supervised land is
   ordinarily bounded and killed with the process tree (`taskkill /T /F` on
   Windows) — a defer does not run on a kill, so every timeout strand leaves a
   full checkout plus its git worktree registration.
2. **Owner-anonymous.** `os.MkdirTemp(parent, "fak-exec-witness-*")` encodes no
   creator identity, so a later sweep cannot distinguish a live owner's scratch
   from a dead one's and must fail toward keeping forever.
3. **No sweep-before-create.** Nothing collects a predecessor's orphan before
   making another, so the count only grows.

## Exact file:line seams

- `internal/witness/execution.go:336-362` — `scratchWorktree`:
  - `:344` gates on `v.workspaceNeedsSiblingTopology(ctx, ref)` (`:322-334`,
    reads the committed `go.work` at `ref`).
  - `:343-348` `parent = filepath.Dir(rootAbs)`.
  - `:349` `os.MkdirTemp(parent, "fak-exec-witness-*")` — owner-anonymous.
  - `:357` `git worktree add --detach --quiet dir ref` — full checkout.
  - `:366-369` `cleanup` closure doing `worktree remove --force` + `RemoveAll`,
    returned as the `defer` — skipped under a kill.

Contrast — the sibling seam already solved this in
`internal/workerworktree/candidate.go`:
  - `:38` `topologyCandidatePrefix = ".fak-cand-validate-"`.
  - `:65-72` `topologyCandidatePattern()` = prefix + `<pid>-<start36>-*`, where
    start is the kernel process-start time (`internal/processstart`).
  - `:77-95` `parseTopologyCandidateOwner` reads that identity; a legacy name
    without one is judged by age.
  - `:152-217` `SweepTopologyCandidates(root, parent, git, opts)`.
  - `:391-409` `sweepTopologyCandidatesBeforeCreate` — the bounded,
    rate-limited sweep run before a candidate is made.

## Reproduction witness

```
# beside the temp repo root the verifier chooses:
mkdir <parent>/fak-exec-witness-<deadpid>-<deadstart36>-abc
mkdir <parent>/fak-exec-witness-<livepid>-<live  start36>-def
go test ./internal/witness/... -count=1
```

Pre-fix the dead-owner dir survives and the live-owner dir is indistinguishable
by name. A kill of a running `ExecuteWitness` (rather than a clean return)
leaves its scratch dir behind permanently. Census (substitute the workspace
root for `<parent>`):

```
ls -d <parent>/fak-exec-witness-* | wc -l
```

## Intended fix direction (do not over-prescribe)

- Sweep this prefix before create in the **same parent**, mirroring
  `candidate.go` — name-encoded owner identity, collect only when the owner is
  provably gone (unreadable start → treat live; legacy bare-suffix name → age
  bound).
- Name the dir with owner identity `<pid>-<start36>-` so a later sweep can prove
  the owner is gone.
- Route the checkout through the existing block-clone / `--no-checkout` +
  `reset --hard` machinery (`internal/workerworktree/blockclone.go`; see
  TICKET-03) to cut the ~200MB copy.
- Keep the sibling parent — it is required for the `../` module resolution.
- No new daemon; reuse the existing sweep shape.

## Definition of done

- [ ] `scratchWorktree` creates an owner-named scratch dir under
      `fak-exec-witness-<pid>-<start36>-*` (mirroring `candidate.go:65-72`).
- [ ] A bounded sweep of the watch prefix runs in the same parent before create.
- [ ] A killed-owner `fak-exec-witness-*` beside a temp repo root is collected;
      a live-pid one is kept (new test, `internal/witness/*_test.go`).
- [ ] The checkout uses the block-clone / no-checkout path where available.
- [ ] `go test ./internal/witness/... -count=1` green.
- [ ] No new daemon; sibling parent preserved.
- [ ] Committed in the public repo (`(fak internal/witness)`); same-session
      receipt.

## Blast radius

Public `internal/witness` only (public-native runtime). No private path
imported. The sibling `internal/workerworktree` seam is read for grammar reuse,
not modified here (TICKET-03 owns the shared checkout path).

## Quarantined fallback

If the block-clone routing is unsafe for a ref, keep the full checkout path but
still land owner-naming + sweep-before-create; the crash-safety and
count-bounding wins hold independently of the size win.

## Witness

```
go test ./internal/witness/... -count=1
```

plus the new owner-liveness test named in the DoD.

## Lane

`witness`. Isolate only if the shared checkout path is touched concurrently with
TICKET-03.

## Done condition

The three defects are closed with a green `./internal/witness/...` run and the
new killed-owner/live-owner test committed upstream.
