# Local harness web UI demo captured output

Command, run from the repository root:

```console
go run ./cmd/harnesswebdemo --selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=95f60d757293b5b16d9d8e688c1a95f72af0eaba8039f5c865d1e1f1373563ef
```
<!-- END SELFCHECK OUTPUT -->
