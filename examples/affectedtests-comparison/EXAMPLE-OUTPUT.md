# Captured affected-set output

Command, run from the repository root on 2026-08-20:

```powershell
./examples/affectedtests-comparison/run.ps1
```

Output:

```text
PASS affected set for internal/a/a.go
selected: 4
  example.com/diamond/cmd/app
  example.com/diamond/internal/a
  example.com/diamond/internal/b
  example.com/diamond/internal/c
excluded: example.com/diamond/internal/isolated
```

Exit code `0` means the selected set matched the fixture's exact four-package
oracle. Any selector failure, missing importer, extra package, or accidental
selection of `internal/isolated` makes the runner exit nonzero.
