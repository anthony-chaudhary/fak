# Applied OSS runtime learnings: complete prefix tails and V4.1 activation limits

Observed at 2026-09-29T21:04:11Z. This is a bounded source refresh and native-runtime
adaptation record, not an exhaustive upstream study or a hardware benchmark.

## Source evidence and disposition

| Source | Event / state | Native adaptation | License / portfolio |
| --- | --- | --- | --- |
| [oMLX #3835](https://github.com/jundot/omlx/pull/3835), `omlx/scheduler.py` at `8288884d9b4f6db7b547633a94d36794c6b1d52d` | Merged 2026-09-22; released in v0.7.0rc1, 2026-09-24 | Preserve complete recurrent state at an exact stable shared-prefix boundary, including the partial 64-token block | Apache-2.0; ADAPT; DEFAULT within the existing native prefix-reuse path |
| [oMLX V4.1 expert activation](https://github.com/jundot/omlx/blob/3f2d07e8dff257119329e0a2e9821df81182f05d/omlx/patches/deepseek_v41/language.py#L551), lines 551–558 | Source at 2026-09-24 revision `3f2d07e8dff257119329e0a2e9821df81182f05d` | Upper-bound gate and symmetrically bound up before SwiGLU when the model config supplies a positive limit | [MIT, Copyright (c) 2023 DeepSeek](https://github.com/jundot/omlx/blob/3f2d07e8dff257119329e0a2e9821df81182f05d/omlx/patches/deepseek_v41/LICENSE); ADAPT; DEFAULT for configured V4.1 models |
| [oMLX #4031](https://github.com/jundot/omlx/pull/4031), `batch_generator.py` at `65515c3c9c3b6d884b3a4cb828ae6e4c86e84e3c` | Merged 2026-09-29, unreleased at observation | One-token cache handoff on simultaneous batch completion and late join | WATCH; current native fixed cohorts lack that transition |

Official oMLX HEAD at observation was `d6b2b92b11ebcbf4a4c1f002b0d590a3117212a8`.
Latest release was [v0.7.0rc1](https://github.com/jundot/omlx/releases/tag/v0.7.0rc1),
tag target `35be079d8a86a44dc2c6d485fbfbf43754e66298`. Refresh these observations
on the next cache/MTP release or a native scheduler cohort change.

The Go changes adapt the source mechanisms to existing interfaces; they do not
copy upstream Python code, tests, or comments. The V4.1 subtree's MIT license is
distinct from oMLX's root Apache-2.0 license.

## Native seams and limits

`internal/agent/inkernel_decode.go` already snapshots complete native KV and
recurrent state and admits it through existing cache authorization and budgets.
Its adaptive checkpoint rounded the structurally common prefix down to a
64-token boundary. Selecting the exact shared prefix removes redundant tail
prefill on later sibling requests. The cold-miss grid fallback and one
intermediate snapshot per request remain in place. This transfers the complete
tail-state principle; it does not introduce oMLX's SSD tail index or block format.
The backend-free native Metal GDN sequence route retains its historical grid:
that route admits only a fresh prompt and cannot consume a restored prefix.

`internal/model/v41_forward.go` already has the model's `SwigluLimit` and shared
and routed expert projections. Both activation paths need the configured clamp:
`gate = min(gate, limit)`, `up = clamp(up, -limit, limit)`, then the existing
`act(gate) * up`. A zero limit preserves the prior arithmetic. Negative gate
values have no lower clamp. Accelerated experts keep compressed weights and
gate/up matrix multiplies on the device. As of 2026-10-07, Vulkan's optional
configured SwiGLU operation applies the positive finite clamp and SiLU on the
device, reads one intermediate-width activation row, and feeds the existing
device down projection. Backends without that operation retain the host
activation fallback, which reads both gate and up rows. Nonpositive and NaN
limits retain the legacy fused operation; positive infinity uses the host path.
The default phase ledger and agent reader report activation device/host calls,
readback bytes, and elapsed time.

A physical Radeon 8060S test of a small routed-expert fixture witnesses 50%
less activation readback, one-third less total expert readback, and matching
decode and suffix-prefill outputs. This is a byte-transfer result, not a
full-checkpoint serving or latency qualification. Vulkan rejects stale or
malformed SwiGLU push-constant layouts at initialization; the legacy C entry
point keeps its zero-limit arithmetic. The broader V4.1 qualification hold
remains in force.

Both changes serve existing native inference users without adding dependencies,
persisted state, alternate selectors, or a separately maintained cache backend.
Their support cost is the existing code path plus focused behavioral regressions.

## Acceptance witnesses

An independent test-author context establishes failing baselines before the
implementation changes. Required post-edit commands, run in the public module:

```text
go test ./internal/agent ./internal/radixkv -count=1
go test -race ./internal/model -run 'TestV41SwigluLimit|TestV41Q2KGateUpHAL' -count=1
go test ./internal/model/... -run V41 -count=1
```

The prefix witness uses a small synthetic recurrent hybrid through the real
planner: three sibling prompts, complete non-grid prefix restoration, cache
accounting, cold continuation parity, and tenant isolation. The activation
witness exercises shared and routed host projections and the resident Q2_K device
callback against an independent formula, including saturation, negative gate
values, zero-limit identity, compressed staging, and full-session host parity.
Broader validation also repairs two existing test fixtures: warm-claim expiry is
introduced after a healthy warm setup, and the latent-norm attention witness
magnifies pre-norm projections while preserving the ordinary post-norm query
projection. The latter retains full-forward oracle and omitted-norm controls.

These are software correctness and avoided-work witnesses. They do not establish
physical TTFT or throughput; off-grid device chunking can change floating-point
reduction order. Hardware promotion requires a separately captured matched native
device run.

## V4.1 session dense projections

The dense projection bridge reuses Fak's existing GLM DSA projection-panel and
session HAL mechanisms for seven leaves: query down/up, KV, router gate, and
shared-expert gate/up/down. Cold query panels, decode, and suffix prefill select
device dispatch when the actual backend supports the resident dtype; unsupported
representations retain the existing host path. Session ownership preserves
installed callbacks and rebinds restored continuation data to its target session.
Selected operation failures roll back continuation state without host retry.
Prism transforms activation copies; LoRA consumes the original activation rows.

The default phase ledger and bounded agent reader expose
`dense_projection_device_calls`, `dense_projection_host_calls`,
`dense_projection_device_rows`, `dense_projection_host_rows`,
`dense_projection_activation_upload_bytes`, `dense_projection_readback_bytes`,
and `dense_projection_nanos`. Calls record attempts, rows record completed
projections, and bytes record successful activation uploads and returned
readbacks. Weight staging remains outside the activation byte counters.

On 2026-10-07, a physical Radeon 8060S RADV run of a small all-Q2_K seven-leaf
fixture completed cold prefill (three tokens), decode (one token), and suffix
prefill (two tokens). It observed 17/7/14 dense device calls and 21/7/14 completed
rows, with zero dense host calls or rows. Phase deltas recorded
21504/7168/14336 uploaded activation bytes and 18048/6016/12032 readback bytes;
the default ledger matched the invoked operations. The maximum absolute logit
difference against the matched whole-history CPU oracle was 2.563e-6. The
existing physical activation ablation also retained its 50% readback reduction
and identical logits. Build the current native Vulkan library and shader bundle
first, then run the physical component witness from the public repository root:

```text
FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^(TestV41DenseProjectionHalo|TestV41ClampedDeviceSwiGLUHalo)$' -count=1 -timeout=5m
```

These observations cover the reduced projection components. The grouped output
bridge below extends device selection; other host stages remain. The strict
whole-V4.1 device guard remains unchanged; full-checkpoint serving, device prefix
restoration, combined Engram execution, TTFT, and throughput require separate
qualification. No latency or market-leadership claim follows from this fixture.

## V4.1 grouped attention output

Fak adapts the group-preserving output algebra in [oMLX V4.1 language.py,
lines 348–350](https://github.com/jundot/omlx/blob/3f2d07e8dff257119329e0a2e9821df81182f05d/omlx/patches/deepseek_v41/language.py#L348)
at `3f2d07e8dff257119329e0a2e9821df81182f05d`; the DeepSeek subtree is MIT,
Copyright (c) 2023 DeepSeek. Each attention group selects a contiguous compressed
`wo_a` slice, projects that group's activation, and contributes its ordered rank
row to `wo_b`. The intermediate rank join remains on the host. Immutable packed
weights reuse the existing model cache. The bridge validates both siblings before
selection and retains the original host path for unsupported backend dtypes.
Selected failures propagate without host retry and roll back continuation state.
The raw grouped algebra retains its existing adapter semantics.

Nine default phase fields and the bounded agent reader expose grouped device/host
calls and completed rows, matrix operation attempts, activation upload/readback
bytes, elapsed nanoseconds, and actual host float weight materialization bytes.
Cold host materialization is charged once per layer; incremental steps charge
actual returned blocks. Activation counters exclude weight staging.

On 2026-10-07 (2026-10-08 UTC), source
`794065b83f8e08a427dc17821f54d716847928ae` passed the physical Radeon 8060S RADV
component witness using a reduced all-Q2_K payload (eight groups, 64 heads,
head dimension 32, rank 32, hidden width 256). All seven ordinary dense projections
and grouped output were enabled together. Cold prefill of three tokens, one decode
token, and suffix prefill of two tokens observed 27/9/18 grouped matrix calls,
3/1/2 completed grouped rows, and zero grouped host calls. Default activation
upload bytes were 27648/9216/18432 and readback bytes were 6144/2048/4096.
Against the actual CPU-session control, whole host float weight materialization
fell from 524288/524288/1048576 bytes to zero, a 100% reduction for these grouped
weights. Compressed device weight buffers were reused through continuation.
Maximum logit difference was 6.557e-7, cosine exceeded 0.9999999999999, and greedy
outputs matched. The preceding dense projection and clamped activation physical
witnesses also passed in the same serial run.

Build the native Vulkan library and shader bundle before this command from the
public repository root:

```text
FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^(TestV41GroupedOutputVulkan|TestV41DenseProjectionHalo|TestV41ClampedDeviceSwiGLUHalo)$' -count=1 -timeout=5m
```

This is a reduced component byte-volume and parity observation. It does not prove
full-checkpoint serving, overall latency or throughput improvement, combined
Engram execution, or market leadership. The strict whole-V4.1 device guard remains.


## V4.1 packed Engram projection

Fak extends its existing Engram port of [ds4 at
`bd66c402070042bf0a79ad6ece8242de4c93680c`](https://github.com/antirez/ds4/blob/bd66c402070042bf0a79ad6ece8242de4c93680c/metal/dsv41.metal#L85).
That project is MIT licensed. The raw projection and per-HC BF16 gating order
remain unchanged. The new bridge reuses native packed-weight validation,
immutable Session weight staging, and the existing HAL matrix API. It applies
neither Prism nor LoRA to the raw Engram mixing tensors. Unsupported backends
retain the existing host path; selected operation failures propagate without a
host retry and restore continuation state.

Model ingress now resolves the three GGUF Engram mixing names into their forward
names and retains supported packed KV weights through ordinary quantized loading.
Packed tables keep their original namespace. Both the standard rectangular
weight shape and the established fixture orientation address the same
output-major bytes without transposition. Attachment preserves existing flat
forward settings when initializing nested Engram metadata. A complete reduced
GGUF load through the default backend Session passes cold prefill, Step, and
suffix prefill; this witness uses the ordinary loader rather than transplanted
weights or configuration.

The default prefill/decode phase JSON includes nine `engram_projection_*` fields:
`device_calls`, `host_calls`, `device_rows`, `host_rows`, `matmul_calls`,
`activation_upload_bytes`, `readback_bytes`, `nanos`, and
`host_weight_f32_bytes`. The bounded agent reader includes an
`engram_projection` clause for both phases. Calls include selected attempts;
rows count completed projections. Transfer bytes describe successful activation
uploads and output reads, excluding immutable weight staging. Host weight bytes
count actual returned float payloads, including partial failure results, rather
than estimated allocations. These observations are separate from dense and
grouped projection counters and require no collection flag.

On 2026-10-08 UTC, source `53386da5f33e3cac1ebcf9c42052ae252ce7b734`
passed the physical Radeon 8060S RADV Engram component test with all seven ordinary
dense projections and grouped output enabled. The synthetic HC4 payload has
hidden width 256 and a Q2_K KV projection of shape `[1280,6144]`. Its immutable
packed KV device buffer was staged once and reused through continuation.

| Phase | Tokens | Engram matrix calls / completed rows | Activation upload / readback bytes | Actual host-control float weight bytes | Device-path host float weight bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold prefill | 3 | 3 / 3 | 73728 / 15360 | 31457280 | 0 |
| Decode | 1 | 1 / 1 | 24576 / 5120 | 31457280 | 0 |
| Suffix prefill | 2 | 2 / 2 | 49152 / 10240 | 62914560 | 0 |

Whole host float weight materialization fell by 100% in each phase. The default
nine-field ledger matched the actual API observations, finite logit cosine was
at least 0.9999999999999529, maximum logit difference was 5.0664e-7, and greedy
outputs matched in all three phases. The dense, grouped, and clamped activation
physical tests also passed in the same serial run.

Build the native Vulkan library and shader bundle before this command from the
public repository root:

```text
FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^(TestV41EngramProjectionVulkan|TestV41GroupedOutputVulkan|TestV41DenseProjectionHalo|TestV41ClampedDeviceSwiGLUHalo)$' -count=1 -timeout=5m
```

This single physical correctness run establishes reduced component execution,
materialization byte volume, and output parity. It does not qualify a complete
Flash checkpoint, overall TTFT/prefill/decode speed, or market leadership.
The official-checkpoint and whole-architecture device-only guards remain intact.
Full-suite qualification is pending; three older expert component fixtures
reproduce unchanged aggregate counter failures on the parent and have a separate
native measurement-repair ticket, `v41-expert-component-counter-isolation`.
