---
title: "Generated-output defaults for fak workspaces"
description: "Where fak commands should place diagnostics, captures, fixtures, and other generated output so source trees stay clean and shared work remains safe."
---

# Generated-output defaults

Generated diagnostics and command captures belong under the ignored repository scratch
namespace, not beside source files. Ask `fak` for the path instead of inventing a root
filename:

```powershell
$run = fak tree-doctor --scratch-dir fleet-loop
fak loop --json > (Join-Path $run 'dosloop.out')
fak dispatch tick --json > (Join-Path $run 'tick.json')

$cover = fak tree-doctor --scratch-path coverage/unit.cover
fak go test -coverprofile $cover ./internal/...
```

```bash
run="$(fak tree-doctor --scratch-dir fleet-loop)"
fak loop --json >"$run/dosloop.out"
fak dispatch tick --json >"$run/tick.json"

cover="$(fak tree-doctor --scratch-path coverage/unit.cover)"
fak go test -coverprofile "$cover" ./internal/...
```

Both forms create missing directories, print an absolute path, and refuse absolute paths,
`..` traversal, or a duplicated `_scratch/` prefix. `--scratch-path` additionally requires
a producer directory (for example `coverage/unit.cover`, not a flat `cover.out`) so unrelated
runs do not recreate a junk drawer inside `_scratch`.

Use the OS temporary directory instead when an artifact has no value after the command exits.
When one producer finishes, remove only its declared top-level namespace and retain the receipt:

```powershell
fak tree-doctor --reap-scratch fleet-loop --json
```

`--reap-scratch` accepts one literal producer name, resolves the absolute
`_scratch/<producer>` target, refuses roots, paths, traversal, globs, and symlink/junction
trees, then removes enumerated exact entries bottom-up. Human and JSON receipts name the
resolved target, verdict, and removed-entry count; an already-absent producer is an idempotent
zero-removal result.

Do not substitute `git clean -Xdf -- _scratch/<producer>`: because `_scratch/` is the ignored
ancestor, Git can traverse unrelated ignored siblings despite the descendant pathspec. The
explicit whole-namespace maintenance operation remains
`fak tree-doctor --sweep-scratch --dry-run` followed by `fak tree-doctor --sweep-scratch`;
never use it for one producer. The `.gitignore` rules remain a compatibility backstop for older
commands and hand-written redirects; they are not the preferred output path.

## Repository Go compiler scratch

An inherited `GOTMPDIR` under `_scratch/go-tmp` is maintained through its own bounded path:

```powershell
fak tree-doctor --go-tmp --json
fak tree-doctor --go-tmp --apply --json
```

Preview is the default. The doctor inventories immediate children once, caps each recursive
walk, and takes one process snapshot for the pass. On Windows it matches each canonical
candidate against both `Win32_Process.CommandLine` and `ExecutablePath`; supported Unix hosts
use their executable and command-line process references. Fresh, referenced, reparse-point,
nested-repository, outside-root, unreadable, and process-indeterminate children are kept with a
typed JSON reason.

Apply moves only stale unreferenced `go-build*` directories into a unique OS-temp quarantine,
rechecks source and quarantine references, then removes enumerated exact files and directories
bottom-up. It uses no recursive wildcard and terminates no process. `--go-tmp-root` can name the
configured root, but the command rejects any root outside the repository `_scratch` subtree.

Do not use `--sweep-scratch` for this job. That generic ignored-tree operation sees the whole
scratch namespace, including live `.dos`, dispatch, and producer state; it has no Go process
liveness witness. The Go-temp mode is explicit maintenance (and the daily maintenance fold),
not a hook installed in every compiler child.

## Closing a run without leaving scratch

Keep source changes, generated deliverables, manifests, logs, and receipts separate from
reproducible snapshots and compiler caches. A terminal process receipt (`reaped: true`) proves
process settlement, not filesystem cleanup. A run closes with a recorded disposition for its
scratch: cleaned, transferred to a named owner, or retained for a specific next use and review
time. Apply this on success, failure, and cancellation after consumers have settled.

- **Clean released caches through the standard tool.** Once cleanup is authorized, the owner
  and known borrowers have released the exact cache, current-consumer checks pass, and new
  consumers are held off through settlement, run `GOCACHE="<released-cache>" go clean -cache`
  through the normal guarded execution route. Hold the existing shared build/maintenance lock
  where applicable. Preserve source, receipts, required binaries, module caches, and active
  caches. An explicit path alone does not prove ownership; a process-namespace scan alone does
  not establish host-wide absence of consumers. Check the command's exit and remaining cache
  contents, then record reclaimed space on its filesystem.
- **Use current evidence and respect actual refusals.** A predicted refusal or rejection of a
  different action is not itself a denial of the current command; applicable restrictions still
  hold. An actual denial of the same action remains binding until materially new, relevant
  evidence or authorization resolves it. Retry only once per such change when the action is
  known not to have executed; check an uncertain outcome first. Preserve the exact receipt and
  follow documented recovery. Do not disable guards, rename a target, or substitute another
  deletion implementation to evade a denial. Reuse still-valid ownership, input, and verification
  evidence; refresh only what changed. Deleting a released reproducible cache does not itself
  require rebuilding or requalifying a preserved binary.
- **Close snapshot storage separately.** After retaining the source revision, exact overlays,
  outputs, and receipts needed for reproduction, reap only the declared disposable snapshot
  through its supported cleanup path. Keep unique source and active inputs. Do not retain a
  full source archive for every diagnostic retry when a coordinated stable input root suffices.
  Bound any intentional retention; do not assume an OS-temp directory will clean itself.
- **Budget accumulated storage.** Before another run, count retained prior snapshots, caches,
  and concurrent reservations, not only the new run's growth limit. Repository scratch quotas
  cover `_scratch`, not arbitrary external run directories. Track external allocations under
  the same owner and cleanup plan rather than letting them escape accounting.
- **Check the filesystem that needs space.** Inspect the target's filesystem and free capacity.
  `/tmp` may be tmpfs while the workspace uses an overlay filesystem; reclaiming one does not
  free the other. Report the measured before/after capacity for each affected filesystem.

For retained private caches managed by the Go-cache lifecycle, allocate a `go-build` basename;
the lifecycle intentionally refuses arbitrary cache-root names. Go's automatic trimming does
not clean abandoned per-run cache directories. Do not rename a live cache to satisfy a tool.
These practices preserve the ambient-cache race guard and all normal ownership, permission,
and consumer checks; they do not authorize clearing shared or in-use caches.

## Control prompts and test fixtures

Treat `.claude/` as project infrastructure, not an automatic home for every Claude run.
Reusable skills, hooks, and generic goal-prompt templates belong there and should be committed.
Issue-numbered launch fuel, recovery prompts, transcripts, and per-run state do not: allocate an
ignored `_scratch/<producer>/` path (or private storage) before launch, then run
`fak tree-doctor --reap-scratch <producer>` when the run closes. `fak tree-doctor` includes any
untracked `.claude/` artifact in its durable-WIP
inventory and types stale entries `park-or-delete` so a completed run cannot leave silent residue.

`testdata/` is committed test input, not an output directory. A fixture belongs there only when a
test reads it and the fixture lands in the same coherent change. Generated candidates, reports,
and local corpora go to `_scratch/<producer>/` until deliberately promoted. `fak tree-doctor`
types untracked files under any `testdata/` directory `land-or-delete`; this prevents local-only
fixtures from masking a clean-clone failure while preserving active peer edits.
