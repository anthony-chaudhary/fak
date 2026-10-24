# Composable NVFP4 primitives

The compute package now supplies independent quantization, scale-layout and packed
CUDA W4A16 operations. They extend the existing NVFP4 tensor and CPU references.

## Composition

1. `QuantizeNVFP4Reference(rows, cols, values)` produces row-major packed E2M1
   bytes, one E4M3FN byte per 16 K values, and an FP32 tensor scale.
2. `QuantizeNVFP4(backend, shape, values)` constructs the existing owned
   `NewNVFP4` tensor from those bytes. Already-quantized weights can call
   `NewNVFP4` directly.
3. `backend.Upload(weight, NVFP4)`, `MatMul` and `BatchedMatMul` preserve the
   packed weights on CUDA and return F32 device tensors.
4. `SwizzleNVFP4Scales` and `UnswizzleNVFP4Scales` adapt linear scale bytes to
   and from the 128x4 tensor-core layout. The CUDA W4A16 baseline consumes linear
   scales; swizzling is a separate operation for future consumers.

Even K occupies the low nibble. Reconstruction is E2M1 times the per-block scale
times the global scale. NVFP4 is distinct from MXFP4's E8M0 scales per 32 values.

The quantizer adapts Model Optimizer's dynamic recipe using RNE conversion and
float64 intermediate calculations. It explicitly validates shapes and nonfinite
inputs, canonicalizes an all-zero tensor with zero global scale and unit block
scales, and rejects nonzero tensors whose global scale underflows FP32. It is a
reference implementation, not a calibrated quantization or byte-identity promise
for every upstream recipe.

The layout adapter pads rows to 128 and scale columns to four with zeros. Its
inverse requires canonical zero padding and strips it. It permutes bytes without
interpreting their numeric format.

## CUDA scope

The W4A16 kernel rounds F32 activations to BF16 RNE, decodes packed local weights
in the kernel, performs a serial FP32 K reduction with separately rounded products
and additions, then applies the FP32 tensor scale once. Its supported numerical
domain is finite activations whose BF16 rounding, FP32 intermediates and output
remain finite. Element values are a caller precondition on the device path;
metadata and geometry are validated before launch.

This portable CUDA-core implementation is a correctness baseline. It does not
execute Blackwell FP4 tensor-core MMA or establish a throughput improvement.
A native W4A4 path additionally needs activation quantization and an
architecture-specific block-scaled MMA implementation. A model loader must still
preserve packed weights and both scale levels before an NVFP4 checkpoint can use
these primitives through a complete model forward.

## Source provenance and licenses

- Quantization adapts NVIDIA Model Optimizer
  [`nvfp4_tensor.py@0638a274`](https://github.com/NVIDIA/Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/quantization/qtensor/nvfp4_tensor.py).
  Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  Apache-2.0. Changes: standalone Go API, strict validation, explicit zero/extreme
  handling, and dependency-free scalar RNE conversion.
- The 128x4 scale permutation adapts NVIDIA TensorRT-LLM
  [`quantization.h@ecae3c97`](https://github.com/NVIDIA/TensorRT-LLM/blob/ecae3c97e2b646580415518a45a72bc5a5a67075/cpp/tensorrt_llm/kernels/quantization.h).
  Copyright (c) 2019-2023 NVIDIA CORPORATION. All rights reserved.
  Apache-2.0. Changes: separate Go forward/inverse adapters, checked geometry
  and canonical zero padding.
- CUDA W4A16 carries the numerical staging already adapted from FlashInfer by
  [the existing attribution](../../THIRD_PARTY_NOTICES.md#flashinfer-nvfp4-w4a16-numerical-staging).
  No CUTLASS, CuTe DSL, Marlin or TensorRT runner implementation is vendored.

The full Apache License, Version 2.0 text is reproduced in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md#tensorfold-benchmark-evidence).
The modified Go source files retain their upstream copyright and SPDX notices.

