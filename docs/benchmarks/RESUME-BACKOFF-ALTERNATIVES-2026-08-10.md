# Failure-signature resume-backoff alternatives — 2026-08-10

Issue: [#6129](https://github.com/anthony-chaudhary/fak/issues/6129)

## Contract

Every arm receives the same repeated-failure restart trace, delay and ceiling policy, process lifecycle, and concurrency. An independent oracle checks the restart schedule, storm prevention, and eventual recovery. Report recovery time, restart count, CPU, peak RSS, network bytes, and total cost.

Required arms:

1. fak native same-signature resume backoff;
2. immediate-resume tuned baseline;
3. Kubernetes CrashLoopBackOff;
4. systemd RestartSec;
5. AWS Step Functions retry.

No equivalent first-class fak integration is declared today. If one ships, add a separate `fak + integration` arm.

## Local witness

`internal/resumebackoff/compare.go` schedules a third resume for one session after two repeated same-signature failures. fak returns a two-minute delay; the immediate baseline is marked incorrect. Every external scheduler remains unavailable with zero measurements.

Ryzen 9 9950X, Windows/amd64, Go benchmark, five samples:

```text
BenchmarkDecideResumeBackoff-32  180.5..219.0 ns/op  328 B/op  6 allocs/op
median: 201.9 ns/decision
```

This is the pure in-process schedule fold. It is not a pod, service, state-machine, recovery, resource, or billing witness.

## Honest status

The contract and local fixture are present, but the comparison is incomplete. Issue #6129 remains open until all five same-trace arms have independent correctness, latency, resource, and total-cost witnesses.
