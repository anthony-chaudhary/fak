# Local harness web UI demo captured output

Command, run from the repository root:

```console
go run ./cmd/harnesswebdemo --selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=d4832cb8e123b7e01c5f14e5425520849304e9a64238b3501c16571bc3cd01a9
```
<!-- END SELFCHECK OUTPUT -->
