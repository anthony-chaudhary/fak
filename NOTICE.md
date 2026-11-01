# Third-party notice

This notice supplements [existing third-party notices](THIRD_PARTY_NOTICES.md).
See [LICENSE](LICENSE) for the project license.

## DeepInfer causal multi-position verification schedule

The native V4.1 speculative target panel in `internal/model/speculative.go`,
`v41_suffix_ubatch.go`, and the step transaction/head helpers adapts the causal
multi-position, layer-major schedule and explicit per-query publication view
from deepseek-ai/DeepInfer at revision
`774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` (Apache-2.0). Reference files are
`deepinfer/worker/model_runner.py`, `deepinfer/model_executor/models/deepseek.py`,
`deepinfer/model_executor/input_metadata/input_metadata.py`,
`deepinfer/model_executor/models/v4_modules/attn.py`,
`deepinfer/model_executor/layers/sampler.py`, and
`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`.

Changed for Fak: no upstream code is copied. The implementation reuses Fak's
existing same-session suffix driver, native arithmetic, bounded ring/compressor
journals and source publication stores. It returns every target-logit row and
retains journal ownership through the caller's emitted-prefix decision. It does
not import DeepInfer's stochastic sampler, DFlash drafter, CUDA runtime, cache
encoding or block-size constants. The preserved target acceptance is Fak's
existing temperature-zero tripwire. One logical target panel is not one GPU
kernel; actual head projections/tiles are accounted separately. This attribution
is not a hardware, full-checkpoint parity or throughput claim.

Licensed under the Apache License, Version 2.0 (upstream `LICENSE`); the upstream
repository has no NOTICE file. License text:
https://www.apache.org/licenses/LICENSE-2.0

