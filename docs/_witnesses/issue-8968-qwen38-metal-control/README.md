---
title: "Issue 8968 — Qwen3.8 Metal control-token profile"
description: "Reference documentation for Issue 8968 — Qwen3.8 Metal control-token profile, preserving the page's implementation details, evidence, and operating context."
---

# Issue 8968 — Qwen3.8 Metal control-token profile

This bundle is the control-only input that selects the next optimization boundary for #8324. It
does not claim a candidate, retained performance, or completion of the resident hybrid forward.

No model was run again for #8968. `profile.json` and `profile.receipt.json` are byte-for-byte copies
of the live M3 artifacts landed by prerequisite #9020 at
`f83e759240c7fe1cb25cbe51d05fb0b10fe21c3e`. That capture already used the exact envelope and
acceptance path required here; reusing its bound raw receipt avoids fabricating or perturbing a
second control measurement.

## Bound control envelope

- Model: `unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- Artifact: 17,106,775,008 bytes; SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- Host: Apple M3 Pro, 18 GPU cores, 36 GiB unified memory
- Shape: P=32 prefill and T=64 decode in one owned model session
- Warm-up/reset: the profile branch runs before modelbench's generic warm-up, creates a fresh
  session and fresh lifecycle recorder, then closes that session in the recorded teardown phase
- Execution: `engine=fak-native`, `backend=metal`,
  `forward_path=metal/qwen35-hybrid-session-v1`, zero promised CPU fallback
- Enabled controls: `FAK_METAL_STREAM_Q4K=1`, `FAK_Q4K=1`
- Explicitly absent: `FAK_Q4K_FREE_CPU`, forced Q8 upload, candidate selectors, group overrides,
  and worker/budget overrides

The receipt binds the artifact, loaded model configuration, host, clean source revision, capture
binary, and full relevant environment. Its 14,833 ordered raw command-buffer events all report
commit, completed wait, host readback, and finite positive timing. Recomputing those events yields
14,833 command buffers, 23,025 encoders, 15,322.027541660646 ms GPU duration, and
39,772.66442775726 ms host wait. The fallback event stream is empty and recomputes to zero.

Per-lever dispatch attribution is not represented by invented numeric zeros. The profile carries
the schema-defined typed reason `capture-tool-does-not-export-dispatch-attribution`; the lifecycle
counters above remain real full-session observations.

## Deterministic model-free readback

From repository root:

```console
go run ./cmd/modelbench \
  -native-performance-readback docs/_witnesses/issue-8968-qwen38-metal-control/profile.json
native performance readback: PASS docs/_witnesses/issue-8968-qwen38-metal-control/profile.json

go run ./cmd/fak native-performance \
  --profile docs/_witnesses/issue-8968-qwen38-metal-control/profile.json
{
  "schema": "fak-native-performance-bottleneck/v1",
  "envelope_id": "qwen38-27b-q4km-m3pro-p32-t64",
  "class": "synchronization-bound",
  "confidence": "medium",
  "recommended_lever_id": "metal.command-buffer-amortization"
}
```

The first command validates both files, recomputes the raw event and fallback digests and scalar
totals, and checks every identity binding without loading a model. The classifier then selects one
typed bottleneck without treating sampling overhead as an accepted throughput result.

The deterministic tamper suite covers the artifact, model configuration, host, source, binary,
raw event, fallback aggregate, forced-Q8 control, and worker budget bindings:

```console
go test ./cmd/modelbench \
  -run 'TestNativeProfileReceiptBindsAllEvidence|TestNativeProfileReadbackRecomputesCompanion|TestNativeProfileRefusesAnyPromisedMetalFallback' \
  -count=1
```

As a direct bundle check, the first event's encoder count was incremented in allocated scratch;
model-free readback exited 1 on the broken receipt binding. The scratch copy was then reaped.

## Digests

- `profile.json`: `2c20871e18bba94de44274a6f5803ace9eab109f082985db857624d4d5fea9f1`
- `profile.receipt.json`: `6ef2454f956cae42baefbcdea733a3f2e28c5085866b50835b9dbded561359c8`
- Raw execution events: `fc5a52c4ab0e605b38831ca26f277240241b236e0136f47586076f62a9f4ce47`
- Typed fallback events: `74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b`
- Receipt binding: `5c0d8322540b6772dad708efd2b22d6ef16370dd723b6c16cdc25974476bf10b`

Private artifact, binary, and service paths are intentionally absent.
