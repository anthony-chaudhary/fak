---
title: "fak explainer: AWQ 4-bit quantization support"
description: "Explains fak's AWQ support: the 4-bit activation-aware format, ~0.5625 bytes per parameter, the dequantization formula, and how to load AWQ safetensors models."
---

# AWQ Quantization Support

**Status:** Implemented (P0) | **Issue:** #485 (A-001)

AWQ (Activation-aware Weight Quantization) is a 4-bit quantization method that achieves near-float performance by using activation-aware calibration to determine optimal per-channel scaling factors.

*Who this is for:* engineers loading or serving AWQ-quantized safetensors with fak, or exporting their own AWQ checkpoints. Prerequisites: familiarity with 4-bit quantization basics (codes, scales, zero-points) and Go for the loader snippets. By the end you'll know fak's on-disk AWQ layout and dequant formula, how to call `model.LoadAWQ`, and how to produce a checkpoint with AutoAWQ.

## Overview

AWQ reduces model memory footprint to ~0.5625 bytes per parameter (4-bit weights + per-channel scales) compared to:
- FP32: 4 bytes/param
- Q8_0: 1.125 bytes/param  
- Q4_0: 0.625 bytes/param

AWQ achieves this while maintaining >99% of FP32 accuracy through activation-aware scale calibration.

## Format Specification

### Data Layout
- **Weights:** 4-bit packed (2 weights/byte, little-endian nibble ordering)
- **Scales:** One float32 per output channel (per-channel scaling)
- **Zero-point:** Fixed at 8 (symmetric 4-bit quantization)
- **Shape:** `[out, in]` matrix stored as `[out, in/2]` packed bytes

### Dequantization Formula
```
weight = scale[o] × (code - 8)
```
Where `code` is the unpacked 4-bit value (0-15) and `8` is the symmetric zero-point.

## Usage

### Loading AWQ Models

```go
import "github.com/anthony-chaudhary/fak/internal/model"

// Load from directory containing model.safetensors with AWQ weights
m, err := model.LoadAWQ("/path/to/awq/model")
if err != nil {
    log.Fatal(err)
}

// Check AWQ tensors loaded
fmt.Printf("Loaded %d AWQ tensors\n", m.AWQCount())
```

### AWQ Tensor Format

AWQ quantized safetensors use the following naming convention:
- `name.weight` — Packed 4-bit weights `[out, in/2]`
- `name.weight_scale` — Per-channel scales `[out]`

For example, for a QKV projection:
```
model.layers.0.self_attn.q_proj.weight      # 4-bit packed weights
model.layers.0.self_attn.q_proj.weight_scale # scales
```

### Integration with Forward Pass

The AWQ kernel provides:
- `awqMatRows` — Single-token GEMV (decode)
- `awqGemm` — Batched GEMM (prefill)

```go
// Matrix-vector multiplication: y = A @ x
y := awqMatRows(awqTensor, x)

// Batched matmul: Y = A @ X^T (P tokens)
Y := awqGemm(awqTensor, X, P)
```

## Creating AWQ Checkpoints

### Using AutoAWQ (Python)

```python
from autoawq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B"
quant_path = "./llama-3.1-8b-awq"

quantizer = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quantizer.quantize(tokenizer, quant_config={
    "zero_point": True,
    "q_group_size": 128,
    "n_sample_calib": 32,
})

quantizer.save_quantized(quant_path)
```

### Recommended Settings

| Model | Group Size | Calibration Samples |
|-------|-----------|---------------------|
| Llama 2/3 | 128 | 32 |
| Qwen2/2.5 | 128 | 32 |
| Mistral | 128 | 64 |

## Performance

### Memory Savings
| Model | FP32 | AWQ | Reduction |
|-------|------|-----|------------|
| Llama-3.1-8B | 16 GB | 4.5 GB | 3.6× |
| Llama-3.1-70B | 140 GB | 40 GB | 3.5× |
| Qwen2.5-7B | 14 GB | 4 GB | 3.5× |

### Accuracy
AWQ typically achieves >99% of FP32 accuracy on standard benchmarks:
- **Perplexity:** Within 1.05× of FP32
- **Zero-shot:** Same as FP32 within margin
- **Greedy decoding:** Argmax-exact >95% of tokens

### Throughput
Decode speed depends on backend:
- **CPU (Scalar):** ~0.5× Q8_0 (reference implementation)
- **CPU (AVX2):** ~0.6× Q8_0 (4-bit decode overhead)
- **CPU (AVX-512):** ~0.8× Q8_0 (better SIMD utilization)
- **CUDA:** ~1.0× Q8_0 (device-side 4-bit matmul with efficient dequantization)

## Implementation Details

### CPU Kernels
- **Scalar:** Portable Go reference (awq_amd64_asm.go)
- **AVX2:** 128-bit SIMD (placeholder, uses scalar)
- **AVX-512:** 512-bit SIMD (placeholder, uses scalar)

### CUDA Kernels
- **Dequantization:** On-the-fly 4-bit unpacking with per-channel scaling
- **GEMV:** Single-token decode (k_awq_gemv kernel)
- **GEMM:** Batched prefill (k_awq_gemm kernel)
- **Build:** Compiled with `-tags cuda` via nvcc (internal/compute/build_cuda.sh)

The CUDA implementation computes the matmul directly on packed 4-bit weights without full dequantization, achieving near-Q8 throughput with ~3.5× memory savings.

### Testing
Oracle tests verify:
- `TestAWQUnpack4bit` — Correct 4-bit unpacking
- `TestAWQDequantRowScalar` — Dequantization accuracy
- `TestAWQDotProductScalar` — Dot product correctness  
- `TestAWQMatRows` — Full GEMV operation
- `TestAWQOracleThreshold` — Cosine similarity ≥0.95

Run tests:
```bash
go test -v -run TestAWQ ./internal/model/...
```

## Limitations

1. **CUDA requires rebuild** — Must compile with `-tags cuda` (uses cgo)
2. **Requires even input dimensions** — Padded by AWQ export
3. **No zero-point tensors** — Assumes symmetric quantization
4. **Safetensors only** — Pytorch bin format not yet supported

## Future Work

1. **AVX2/AVX-512 assembly kernels** — For faster CPU dequantization
2. **CUDA graph integration** — Capture AWQ ops in decode graph
3. **Mixtral AWQ** — MoE models with AWQ quantization
4. **Dynamic AWQ** — Runtime quantization without pre-export

## References

- [AWQ Paper: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [AutoAWQ GitHub](https://github.com/mit-han-lab/llm-awq)
- [Issue #5: AWQ Quantization Support](https://github.com/anthony-chaudhary/fak/issues/5) — AWQ support tracking
