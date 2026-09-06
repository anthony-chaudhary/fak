# Local harness web UI demo captured output

Command, run from the repository root:

```console
go run ./cmd/harnesswebdemo --selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=7a1adc81ceaa510fd5214889ea4a300d6bd980c021e8ab1a6aefc525edd1bd18
```
<!-- END SELFCHECK OUTPUT -->
