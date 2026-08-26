# Native-performance regression gate

`fak native-performance --gate request.json` converts the hill climb into an
envelope-scoped regression gate. The request contains the last accepted real
receipt, a candidate receipt, and the policy for that exact model, artifact,
hardware, controls, engine, and forward path. A policy is never portable across
envelopes.

## Verdict

The command emits `fak-native-performance-gate-verdict/v1` and exits 0 for
`pass` or `investigate`, 3 for `regression`, and 1 for invalid/incomparable
evidence. It checks all of these together:

- repeated-run coefficient of variation against the envelope noise band;
- end-to-end decode throughput against both the accepted receipt and absolute floor;
- the quality metric captured by the same run;
- `fak-native` engine and the pinned native forward path;
- zero fallbacks and the envelope peak-memory ceiling;
- derived `module@rev` movement between accepted and candidate receipts.

A noisy run or a loss inside the investigate band asks for another sanctioned
capture. A material loss or any quality/identity/fallback/memory/floor failure
is a regression. Non-pass verdicts include a sorted suspect module set and a
`fak-native-performance-bisect-packet/v1` with the known-good/bad revisions,
lane-acquisition command, rerun command, and witnessed stop condition. The
packet never switches engines or normalizes across devices.

## Cadence and evidence

Local presubmit runs the deterministic package/CLI tests only; hardware is not a
universal PR blocker. The scheduled/manual `native-performance-regression`
workflow dispatches the public request artifact to the private sanctioned
runner bridge documented in `docs/private-comms-channel.md`. The private side
must run the pinned Metal or CUDA campaign, retain raw logs privately, and
return only the scrubbed request and verdict. A scheduled result is accepted
only when the receipt names `fak-native`, the exact forward path, quality,
zero fallbacks, memory, repetitions, and module revisions.

Override requires a maintainer-owned issue containing: the scrubbed candidate
receipt, repeated rerun showing the classification, quality review, reason the
floor changes, new envelope-specific policy, and rollback revision. Rollback is
the last accepted revision from the verdict. Never weaken a different envelope,
waive the gate with self-authored output, or publish hosts, credentials, private
paths, or raw profiler logs.

Focused witness:

```powershell
$env:FAK_FAST='0'; .\test.ps1 -count=1 ./internal/nativeperf
$env:FAK_FAST='0'; .\test.ps1 -count=1 -run '^TestNativePerformance' ./cmd/fak
```
