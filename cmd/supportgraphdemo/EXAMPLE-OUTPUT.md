# Support graph demo captured output

Command, run from the repository root:

```console
go run ./cmd/supportgraphdemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
L4 exact tuple: supported — highest-tier current evidence decides exact tuple evidence=[{"id":"l4-decode","state":"supported","tier":"witnessed","authority":"fak-lab","source":"device-decode:sha256:aaa","expires":"2026-09-15T00:00:00Z"}]
T4 exact tuple: unsupported — highest-tier current evidence decides exact tuple evidence=[{"id":"device-refusal","state":"unsupported","tier":"witnessed","authority":"fak-lab","source":"device-decode:sha256:bbb","expires":"0001-01-01T00:00:00Z","fallback":"awq-portable-cpu","penalty":"latency unmeasured"}] fallback=awq-portable-cpu penalty=latency unmeasured
old tuple: stale — all matching evidence expired evidence=[{"id":"old-run","state":"supported","tier":"observed","authority":"lab","source":"run:old","expires":"2026-08-01T00:00:00Z"}]
unknown layout: unknown — no exact support edge; not evaluated
SELFCHECK PASS: exact witnessed support, witnessed refusal, stale, and unknown remain distinct
```
<!-- END SELFCHECK OUTPUT -->
