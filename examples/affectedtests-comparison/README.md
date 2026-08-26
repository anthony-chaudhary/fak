# Same-diamond affected-test comparison fixture

All arms encode five nodes: changed leaf `internal/a`; importers `internal/b` and
`internal/c`; top importer `cmd/app`; and `internal/isolated`. The exact affected
set is `{internal/a, internal/b, internal/c, cmd/app}`. Tool-specific commands and
pinned versions are recorded in the issue benchmark report.

## Run the fak arm

Prerequisite: an installed `fak` executable on `PATH`. The check uses only the
committed local Go fixture; it needs no model, API key, network, or GPU. From the
repository root, run:

```powershell
./examples/affectedtests-comparison/run.ps1
```

The runner asks `fak affected` for the reverse dependency closure of
`internal/a/a.go` and exits nonzero unless the result is exactly the four-package
oracle, including exclusion of `internal/isolated`. The complete runner took
0.58 seconds in the observed warm Windows run on 2026-08-20; expect it to finish
in about 1 second with an installed `fak` binary.

The selected package list is deterministic for this pinned fixture and `run.ps1`
is safe to re-run; machine load may change elapsed time. See
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) for a captured successful run and the
[full alternatives benchmark](../../docs/notes/AFFECTED-TEST-SELECTION-ALTERNATIVES-2026-08-10.md)
for the tool versions and measurements.

## What this demonstrates

A successful run shows that fak follows both arms of the diamond to the top
importer without selecting the isolated package. This quick check does not claim
to re-run the Bazel, Nx, Gradle, or unavailable Pants arms, and it does not prove
a cross-system performance ranking; those claims require the separately pinned
benchmark protocol.

The Bazel arm also carries the platform-constraint routing witness from #6401.
From `bazel/`, query `//routing:route_effect` with each of
`//routing:probe_native_allowed`, `//routing:probe_windows_wsl`,
`//routing:probe_ci_only`, and `//routing:probe_unavailable` via `--platforms`.
The first three resolve to one `.route` artifact (`native`, `wsl`, or `ci`);
the unavailable platform intentionally resolves to zero artifacts.
