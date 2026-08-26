# Harness stack preflight demo captured output

Command, run from the repository root:

```console
go run ./cmd/stackpreflightdemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
MINIMUM PATH: allow required=[cuda>=12 sm>=80]
  WARN recommended, not required: memory>=24GiB
  WARN recommended, not required: sm>=89
T4 PATH: refuse blockers=[support: exact tuple is unsupported]
  ALTERNATIVE 1: select awq-portable-cpu; impact=latency unmeasured
SELFCHECK PASS: mandatory support blocks launch; recommendations remain warnings
```
<!-- END SELFCHECK OUTPUT -->
