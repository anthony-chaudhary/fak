# Issue #8973 mixed Q4_K/Q8 Metal witness

This receipt captures the focused native Metal test for the mixed Q4_K/Q8
observation repair. The source under test is base
`f83e759240c7fe1cb25cbe51d05fb0b10fe21c3e` plus the `internal/metalgemm`
changes committed with this directory. The host was Darwin 26.6.2 on arm64
(Apple M3 Pro), using Go 1.26.6.

Command:

```text
go test -tags fakmetal ./internal/metalgemm -run '^TestMixedQ4KQ8Observation' -count=1 -v -timeout 4m
```

The command exited 0. [`metal-test.stdout`](metal-test.stdout) has SHA-256
`1538eeec983da6f6773fcf16f054605e9f954e9dd55b0b9fb3f64ae39ef6c65f` and
records:

- two control events and one mixed candidate event;
- one caller-owned command buffer, two native encoders, commit, completed
  wait, and host readback;
- Q4_K parity at cosine 1.000000 and maximum relative error 0.000000;
- exact Q8 parity across two groups; and
- the injected native post-submit failure surfaced as
  `*metalgemm.MixedQ4KQ8PostSubmitError` after commit and wait, before
  readback.

The separate reference service on port 8090 was stopped only by a bounded
exact-owner TERM watcher. It signaled one PID and refused zero candidates.
After the test, the canonical service command was resolved from its managed
definition, matched SHA-256
`a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d`,
and was restored without launchd mutation. One listener with that same digest
returned HTTP 200 from both `/health` and `/v1/models` on the immediate and
10-second durable checks.

The reference service did not execute the test operation. The candidate path
was fak-native Metal; there is no llama.cpp runtime fallback in this witness.
This receipt proves lifecycle attribution and parity, not end-to-end model
performance.
