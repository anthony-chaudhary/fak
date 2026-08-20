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
fak tick --json > (Join-Path $run 'tick.json')

$cover = fak tree-doctor --scratch-path coverage/unit.cover
fak go test -coverprofile $cover ./internal/...
```

```bash
run="$(fak tree-doctor --scratch-dir fleet-loop)"
fak loop --json >"$run/dosloop.out"
fak tick --json >"$run/tick.json"

cover="$(fak tree-doctor --scratch-path coverage/unit.cover)"
fak go test -coverprofile "$cover" ./internal/...
```

Both forms create missing directories, print an absolute path, and refuse absolute paths,
`..` traversal, or a duplicated `_scratch/` prefix. `--scratch-path` additionally requires
a producer directory (for example `coverage/unit.cover`, not a flat `cover.out`) so unrelated
runs do not recreate a junk drawer inside `_scratch`.

Use the OS temporary directory instead when an artifact has no value after the command exits.
Use `fak tree-doctor --sweep-scratch --dry-run` to preview ignored scratch reclamation and
`fak tree-doctor --sweep-scratch` to reap it. The `.gitignore` rules remain a compatibility
backstop for older commands and hand-written redirects; they are not the preferred output path.

## Control prompts and test fixtures

Treat `.claude/` as project infrastructure, not an automatic home for every Claude run.
Reusable skills, hooks, and generic goal-prompt templates belong there and should be committed.
Issue-numbered launch fuel, recovery prompts, transcripts, and per-run state do not: allocate an
ignored `_scratch/<producer>/` path (or private storage) before launch, then delete it when the
run closes. `fak tree-doctor` includes any untracked `.claude/` artifact in its durable-WIP
inventory and types stale entries `park-or-delete` so a completed run cannot leave silent residue.

`testdata/` is committed test input, not an output directory. A fixture belongs there only when a
test reads it and the fixture lands in the same coherent change. Generated candidates, reports,
and local corpora go to `_scratch/<producer>/` until deliberately promoted. `fak tree-doctor`
types untracked files under any `testdata/` directory `land-or-delete`; this prevents local-only
fixtures from masking a clean-clone failure while preserving active peer edits.
