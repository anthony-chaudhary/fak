# Captured selfcheck

Captured from the repository root on Windows with Go 1.26.7. The selfcheck runs
the real kernel, adjudicator, and vDSO against a deterministic in-memory engine;
it exits 0 only when every cache, policy, and tenancy assertion passes.

```console
$ go run ./cmd/microcachedemo -selfcheck
microcachedemo -selfcheck: PASS (32 agents - 252/256 calls served locally - 98.4% engine work avoided - policy and tenant isolation hold)
```

Observed warm-run wall time: 2.6 seconds. A cold Go toolchain download or first
compile is outside that timing.
