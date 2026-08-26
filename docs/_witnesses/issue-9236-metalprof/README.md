# Issue #9236 partial hardware read-back — not a completion packet

This directory preserves the smallest honest, deterministic projection of the live sanctioned-M3-Pro evidence already recorded on #9236. It **does not satisfy the issue done condition and must not be used to close the issue**.

Three exact-P32 fak-native repetitions and three pinned llama.cpp repetitions were captured on the registered Apple M3 Pro using the exact Qwen3.8 artifact. The committed summary retains only scrubbed values and hashes; bridge coordinates and raw private-host paths remain private.

The median fak wall was 4.6143 s. Measured Q4_K GPU execution was 1.3690 s (29.7%); Q8 CPU plus Q6_K attributed wall was 2.2864 s (49.5%). This narrows the investigation but does not meet the requested attribution: `xctrace`/Instruments was absent on the registered host, so queue/completion timelines, transfers, CPU self-time, dispatch identities/counts, counters, immutable traces, randomized arm order, RSS/swap/thermal state, and Q6_K GPU execution remain unavailable. They are explicitly typed in `partial-summary.json`; none are estimated.

The sanctioned bridge was exercised again during this resolution attempt. Its replacement persistent control session reached the hub but command read-back repeatedly ended as `COMMAND_DISPATCHED_READBACK_LOST`; the bridge doctor also found no responsive live shell. This is hardware/control-plane evidence, not grounds for inventing data or closing the issue.

## Deterministic witness

```text
go test ./docs/_witnesses/issue-9236-metalprof -count=1
```

The witness validates the exact envelope, three runs per arm, zero fak fallback, hash shape, arithmetic reconciliation of every available fak run, explicit missing-counter labels, and the required `close_issue=false` verdict.
