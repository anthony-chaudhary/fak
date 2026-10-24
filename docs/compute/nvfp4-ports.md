# Composable NVFP4 primitives

The compute package supplies independent NVFP4 quantization, scale-layout,
packed CUDA W4A16, and explicit prequantized W4A4 operations. A ModelOpt
safetensors source preserves packed checkpoint weights and both scale levels.

## Composition

1. `QuantizeNVFP4Reference(rows, cols, values)` returns packed E2M1 bytes,
   one E4M3FN scale byte per 16 K values, and an FP32 tensor scale.
2. `QuantizeNVFP4(backend, shape, values)` constructs an owned `NewNVFP4`
   tensor. Already-quantized callers can use `NewNVFP4` directly, or read a
   weight with `model.OpenModelOptNVFP4Source(path, backend)` and
   `source.Weight("exact.checkpoint.base", compute.NVFP4)`.
3. `backend.Upload(tensor, NVFP4)` preserves the canonical packed values and
   linear scale bytes in one resident allocation. Uploads own independent
   buffers. Ordinary `MatMul` and `BatchedMatMul` use the CUDA W4A16 baseline.
4. To multiply two prequantized tensors, upload W[out,k] and A[rows,k], then
   call `compute.MatMulNVFP4(backend, W, A)`. The optional
   `NVFP4MatMulBackend` capability returns F32[rows,out] or an explicit error.
   The CPU implementation is a numerical oracle. CUDA uses cuBLASLt on the
   architecture/runtime combinations below.
5. `SwizzleNVFP4Scales` and `UnswizzleNVFP4Scales` expose the 128x4 scale
   permutation separately. Explicit CUDA W4A4 prepares its padded scale
   scratch on the device; ordinary uploads retain their compact layout.

Even K occupies the low nibble. Reconstruction is E2M1 times the per-block
scale times the global scale. NVFP4 differs from MXFP4, which uses E8M0
scales per 32 values.

The reference quantizer adapts Model Optimizer's dynamic recipe with RNE
conversion and float64 intermediates. It validates shapes and finite inputs,
canonicalizes zero tensors with zero global scale and unit block scales, and
rejects nonzero tensors whose global scale underflows FP32. It does not
promise calibrated accuracy or byte identity with every upstream recipe.

The layout adapter pads rows to 128 and scale columns to four with zeros.
Its inverse requires canonical zero padding and strips it. Both adapters
permute bytes without interpreting their numeric format.

## ModelOpt source

`OpenModelOptNVFP4Source` reads one safetensors file through the existing
`compute.WeightSource` interface and parser. For a base name it requires:

| Tensor | Dtype | Shape |
|---|---|---|
| `.weight` | U8 | [out,k/2] |
| `.weight_scale` | F8_E4M3 | [out,k/16] |
| `.weight_scale_2` | F32 | scalar or [1] |

TensorRT-LLM's `.weights_scaling_factor` and
`.weights_scaling_factor_2` sibling pair is also accepted. Mixed,
incomplete, malformed, or ambiguous pairs are rejected. Returned tensors
own their bytes and remain valid after the source is closed.

This is a composable single-file source. Sharded checkpoint indices, model
configuration, architecture-specific name translation, activation calibration,
and complete model-forward wiring remain outside this primitive.

## CUDA operations

W4A16 rounds F32 activations to BF16 RNE, decodes packed local weights,
performs a serial FP32 K reduction with separately rounded products and
additions, then applies the tensor scale once. Its numerical domain is finite
activations whose BF16 rounding, FP32 intermediates, and output remain finite.
Element values are a caller precondition on the device path; metadata and
geometry are validated before launch. This CUDA-core baseline establishes
neither FP4 tensor-core execution nor a throughput improvement.

Explicit W4A4 uses E2M1 operands, UE4M3 block scales, FP32 accumulation and
output, and a final `alpha = float32(weightGlobal * activationGlobal)`. It
uses the resident packed operands without host readback or widening. Geometry
requires k divisible by 32 and out divisible by four; rows may be arbitrary
positive values. Graph capture is currently refused. Unsupported architectures,
runtimes, or unavailable algorithms return `ErrNVFP4GEMMUnsupported` without
silently switching to W4A16 or a different backend.

| Architecture | Minimum official cuBLAS runtime |
|---|---|
| SM100 | 12.8.3 |
| SM103 | 12.9.0 |
| SM120 | 12.8.4 |

Other SM values are unqualified and refused. On SM103, algorithm 66 is
conservatively excluded in cuBLAS 13.5.1 through 13.6.1 and 13.7.x, following
NVIDIA's documented NVFP4 defects and fixes. cuBLAS runtime version queries
omit the fourth build component, so the policy assumes official NVIDIA
release builds. Compile-time headers older than CUDA 12.8 retain an explicit
unsupported stub.

The CPU W4A4 oracle rounds serial products/additions separately before the
global epilogue. Tensor-core reductions can use a different accumulation
order; GPU parity therefore requires a stated numerical tolerance. A successful
cuBLASLt enqueue counter witnesses the API path, not a particular assembly
instruction or performance result.

## Source provenance and licenses

- Quantization adapts NVIDIA Model Optimizer
  [`nvfp4_tensor.py@0638a274`](https://github.com/NVIDIA/Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py).
  Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  Apache-2.0. Changes: standalone Go API, strict validation, explicit
  zero/extreme handling, and dependency-free scalar RNE conversion.
- The scale permutation adapts NVIDIA TensorRT-LLM
  [`quantization.h@ecae3c97`](https://github.com/NVIDIA/TensorRT-LLM/blob/ecae3c97e2b646580415518a45a72bc5a5a67075/cpp/tensorrt_llm/kernels/quantization.h).
  Copyright (c) 2019-2023 NVIDIA CORPORATION. All rights reserved.
  Apache-2.0. Changes: independent Go forward/inverse adapters, checked
  geometry, and canonical zero padding.
- CUDA W4A16 preserves the numerical staging already adapted from FlashInfer
  by [the existing attribution](../../THIRD_PARTY_NOTICES.md#flashinfer-nvfp4-w4a16-numerical-staging).
- ModelOpt sibling names and linear layout are pinned to
  [`unified_export_hf.py@0638a274`](https://github.com/NVIDIA/Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/export/unified_export_hf.py)
  and its TensorRT-LLM export schema. The source reuses FAK's safetensors
  parser; it does not vendor an upstream loader implementation.
- cuBLASLt integration is independently implemented from NVIDIA's documented
  API. The [BSD-3 library sample@a94482e](https://github.com/NVIDIA/CUDALibrarySamples/blob/a94482ebecf8b16d5b83ab276b7db3a84979f0e5/cuBLASLt/LtNvfp4Matmul/sample_cublasLt_LtNvfp4Matmul.cu)
  was consulted as an API cross-check; no sample source is copied. No CUTLASS,
  CuTe DSL, Marlin, or TensorRT runner is vendored.

The full Apache License, Version 2.0 text is reproduced in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md#tensorfold-benchmark-evidence).
Modified source files retain upstream copyright and SPDX notices.

Primary runtime policy references are NVIDIA's release notes for
[CUDA 12.8 GA](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html),
[12.8 Update 1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html),
[12.9 GA](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html),
[the cuBLAS 13.6.2 patch](https://docs.nvidia.com/cuda/cublas-patch-release-notes/#cublas-patch-release-13-6-2),
and [CUDA 13.4 Update 1](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-toolkit-release-notes/index.html).
