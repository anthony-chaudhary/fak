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
