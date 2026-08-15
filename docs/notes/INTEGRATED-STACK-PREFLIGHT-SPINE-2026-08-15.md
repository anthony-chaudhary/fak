# Integrated stack preflight spine

**Date:** 2026-08-15 · **Issue:** [#6897](https://github.com/anthony-chaudhary/fak/issues/6897)

`internal/stackpreflight` consumes, without copying their schemas:

- a `stackresolve.Receipt` for mandatory composition;
- a `workloadfit.Assessment` for purpose-specific hard controls;
- a `supportgraph` exact-tuple query for hardware/quant evidence.

The captured selfcheck uses the shipped fixture authorities and produces:

```text
MINIMUM PATH: allow required=[cuda>=12 sm>=80]
  WARN recommended, not required: memory>=24GiB
  WARN recommended, not required: sm>=89
T4 PATH: refuse blockers=[support: exact tuple is unsupported]
  ALTERNATIVE 1: select awq-portable-cpu; impact=latency unmeasured
SELFCHECK PASS: mandatory support blocks launch; recommendations remain warnings
```

The receipt carries the declared `24GiB model residency` capacity target. Assembly or workload-fitness refusal returns before support selection, so a mandatory blocker cannot be bypassed. Required baselines block when support is unknown, stale, unsupported, or conflicting; recommendations only emit warnings. Ranked alternatives preserve fallback impact rather than presenting fallback as equivalent support.

This is a real execution of the integrated preflight code over synthetic fixture facts. It is not a live-GPU witness and does not prove the fixture's L4/AWQ claim. Real support evidence ingestion is #6896; after such facts are ingested, this seam consumes them unchanged.

Reproduce:

```bash
go test -count=1 ./internal/stackpreflight ./cmd/stackpreflightdemo
go vet ./internal/stackpreflight ./cmd/stackpreflightdemo
go run ./cmd/stackpreflightdemo -selfcheck
go run ./cmd/stackpreflightdemo -selfcheck -json
```

Machine receipt: `internal/stackpreflight/selfcheck-witness.json`.
