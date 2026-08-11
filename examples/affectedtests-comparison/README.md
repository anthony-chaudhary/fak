# Same-diamond affected-test comparison fixture

All arms encode five nodes: changed leaf `internal/a`; importers `internal/b` and
`internal/c`; top importer `cmd/app`; and `internal/isolated`. The exact affected
set is `{internal/a, internal/b, internal/c, cmd/app}`. Tool-specific commands and
pinned versions are recorded in the issue benchmark report.

The Bazel arm also carries the platform-constraint routing witness from #6401.
From `bazel/`, query `//routing:route_effect` with each of
`//routing:probe_native_allowed`, `//routing:probe_windows_wsl`,
`//routing:probe_ci_only`, and `//routing:probe_unavailable` via `--platforms`.
The first three resolve to one `.route` artifact (`native`, `wsl`, or `ci`);
the unavailable platform intentionally resolves to zero artifacts.
