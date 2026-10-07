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
