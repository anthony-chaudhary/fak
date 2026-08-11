# Same-diamond affected-test comparison fixture

All arms encode five nodes: changed leaf `internal/a`; importers `internal/b` and
`internal/c`; top importer `cmd/app`; and `internal/isolated`. The exact affected
set is `{internal/a, internal/b, internal/c, cmd/app}`. Tool-specific commands and
pinned versions are recorded in the issue benchmark report.
