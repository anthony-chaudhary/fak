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

## ggml Q8 block reconstruction in packed KV attention

`internal/model/quant_attn_q8_packed.go` adapts the blockwise reconstruction
traversal of `dequantize_row_q8_0` from official ggml-org/llama.cpp revision
`83078fec0db82d6b5a00d9599062c38c39145755`:
https://github.com/ggml-org/llama.cpp/blob/83078fec0db82d6b5a00d9599062c38c39145755/ggml/src/ggml-quants.c

Changed for Fak: signed code times block scale reconstructs one bounded F32
head-row tile per worker, reused across sharing query heads and then values.
There is no full-layer F32 K/V materialization; this is not zero F32 scratch.
Fak's cache scales are already F32, unlike the donor's F16 scales. Queries
remain F32; existing native dot/value kernels, worker scheduling, cache
ownership, softmax and observer copies remain Fak-owned.
This is a block-reconstruction adaptation, not a wholesale upstream attention
kernel, SIMD port or Q8-query conversion. Different scalar/SIMD reduction orders
require numerical qualification; this notice makes no bit-parity, model-quality
or throughput claim.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## llama.cpp bounded mapped expert prefetch

The sorted range-union control flow in
`internal/model/expert_checkpoint_readahead.go` is adapted from
`llama_prefetch` in `src/llama-mmap.cpp`, ggml-org/llama.cpp revision
`74bca3950054eed1795f525cb9cf06bfe796492d` (b11870):
https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/src/llama-mmap.cpp

FAK groups by checkpoint shard and unions exact selected payload ranges before
the existing mapped reader performs platform page alignment. It does not copy
the donor's page-rounded gap merging or change packed weight bytes. New formats
and coalescing require `fak_expert_readahead_qualification`; backend qualification
and ordinary demand-error/ownership contracts remain separate.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

