---
title: "Native-performance profile v1 witness status"
description: "This directory records the acceptance boundary for issue 8760. The committed JSON under internal/nativeperf/testdata/native-performance-profile/ is synthetic..."
---
# Native-performance profile v1 witness status

This directory records the acceptance boundary for issue #8760. The committed
JSON under
[`internal/nativeperf/testdata/native-performance-profile/`](../../../internal/nativeperf/testdata/native-performance-profile/)
is **synthetic fixture data**. It proves strict decoding, validation,
backend-specific counter preservation, deterministic classification, and
`--profile-next` refusal/override behavior. It is not a profiler capture,
hardware measurement, throughput result, or evidence that either backend was
run.

## Committed deterministic fixtures

- `synthetic-metal-launch-bound.json` exercises the pinned Metal envelope and
  deterministically classifies a synthetic capture as launch-bound.
- `synthetic-cuda-bandwidth-bound.json` exercises the pinned CUDA envelope and
  deterministically classifies a synthetic capture as bandwidth-bound.
- `synthetic-metal-bandwidth-override.json` exercises a genuine same-envelope
  graph-order contradiction and preserves its synthetic issue-backed override
  in `--profile-next` output.
- `synthetic-metal-attribution-unavailable.json` exercises the closed typed
  unavailable state without claiming that the real Metal backend lacks
  attribution.
- `reject-*.json` fixtures fail closed on mixed envelopes, missing/overlapping
  phases, missing or non-finite/negative counters, invalid native identity,
  absent dispatch-attribution state, unknown or mixed levers, unsupported
  counter comparisons, and a contradiction without an issue-backed reason.

The package witness is:

```powershell
$env:FAK_FAST='0'; .\test.ps1 -count=1 ./internal/nativeperf
```

The CLI fixture tests are selected by:

```powershell
$env:FAK_FAST='0'; .\test.ps1 -count=1 -run '^TestNativePerformance' ./cmd/fak/native_performance.go ./cmd/fak/native_performance_test.go
```

## Real sanctioned capture acceptance

| Backend | Status on August 25, 2026 | Acceptance still required |
|---|---|---|
| Metal | **OPEN — no real bundle committed here** | Capture on the sanctioned Apple path, retain the raw private profiler output outside the public repository, scrub the manifest, validate it against the exact Metal envelope, and review the artifact before adding it here. |
| CUDA | **OPEN — no real bundle committed here** | Capture on the sanctioned CUDA path, retain the raw private profiler output outside the public repository, scrub the manifest, validate it against the exact CUDA envelope, and review the artifact before adding it here. |

Do not create placeholder `sanctioned/metal` or `sanctioned/cuda` bundles. Those
paths become evidence only after a genuine capture exists. Public additions must
contain relative scrubbed references and no credentials, hostnames, private
paths, or raw internal logs.
