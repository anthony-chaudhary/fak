# Captured selfcheck

Captured from the repository root on Windows with Go 1.26.7. The command uses
only deterministic local fixtures and exits 0 after checking every context,
residency, scheduler, tool-floor, and egress invariant.

```console
$ go run ./cmd/microfleetdemo -selfcheck
microfleetdemo -selfcheck: PASS (24 agents - 92.2% resident-context reduction - 2 denied actions never dispatched - 20 parked)
```

Observed warm-run wall time: 0.85 seconds. A cold Go toolchain download or first
compile is outside that timing.
