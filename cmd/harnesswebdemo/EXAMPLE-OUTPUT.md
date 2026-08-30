# Local harness web UI demo captured output

Command, run from the repository root:

```console
go run ./cmd/harnesswebdemo --selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=fa6f87a175d81b4e7aa98dfc2202e8d0077179ccf7dd9f9ababc79a7d2b2e9ba
```
<!-- END SELFCHECK OUTPUT -->
