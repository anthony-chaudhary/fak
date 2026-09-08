# Local harness web UI demo captured output

Command, run from the repository root:

```console
go run ./cmd/harnesswebdemo --selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=4955e5b3de7a2c3b1e97aa09e3fcf18e6c173551783dfcc5d9941dc007861ecb
```
<!-- END SELFCHECK OUTPUT -->
