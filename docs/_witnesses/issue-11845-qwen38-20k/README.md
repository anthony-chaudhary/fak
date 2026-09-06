# Witness Artifact: Issue #11845 — FAK Native Kernel 20k Context Qwen3.8-27B-UD-Q2_K_XL on GCP

## Summary

This witness proves the first live, empirical run of the FAK native CUDA inference kernel executing a **20,480-token (20k context)** prompt with the two-bit Unsloth Dynamic quantization `Qwen3.8-27B-UD-Q2_K_XL.gguf` on an NVIDIA A100-SXM4-40GB GPU (`fak-qwen-serve`, GCP `us-central1-f`).

## Key Empirical Findings

1. **Native Execution**:
   - Engine: `fak-native`
   - Backend: `cuda`
   - Forward path: `cuda/qwen35-gdn-ssm-decode-v1`
   - Zero llama.cpp or external runtime delegation.
2. **20k Context Prefill & Memory Scaling**:
   - Prompt length: **20,480 tokens**
   - Prefill duration: **218.69 s**
   - Prefill throughput: **93.6 tok/s**
   - Decode throughput: **1.42 tok/s** (16 completion tokens)
   - Total TTFT: **230.196 s**
3. **Memory Footprint on 40GB A100**:
   - Weights footprint: ~9.83 GB (`9,828,981,664 bytes`)
   - Peak VRAM during 20k run: **35,246 MiB / 40,960 MiB**
   - Headroom preserved: **5,714 MiB** (zero OOM, zero thrashing)
   - Verified that 2-bit UD-Q2_K_XL leaves ample headroom for 20k+ context on a single 40GB accelerator.

## Exact In-Kernel Log Witness

```
2026/09/06 08:34:15 inkernel_chat model=Qwen3.8-27B-UD-Q2_K_XL backend=cuda forward_path=cuda/qwen35-gdn-ssm-decode-v1 q4k=true q8dec=avx512+fused/12w prompt=20480tok cacheable=0tok reused=0tok prefill=20480tok/218.69s/93.6tok/s decode=16tok/11.26s/1.4tok/s
```

## Captured Files

- `qwen38_20k_witness.json`: Raw telemetry and cryptographic verification record (SHA-256: `2d1e99750e716137a8de74738291e119c1af8f67e109d7cd909f180513f83e8e`).
