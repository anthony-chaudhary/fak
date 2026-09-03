# Issue 9666: GCP NVIDIA datacenter GPU Control Benchmark Receipt

**Title:** `bench(qwen38): capture server AMD accelerator receipt with GCP control`  
**Parent Issue:** [#9666](https://github.com/anthony-chaudhary/fak/issues/9666)  
**Date:** 2026-09-03  
**Hardware:** NVIDIA datacenter GPU (GCP `us-central1-f`)  
**Workload:** `Qwen3.8-27B` identical prompt, 128 completion tokens, greedy decoding (`temperature=0.0`)  

## Measured Head-to-Head Arms

| Metric | `fak-cuda` (Arm 1) | `vllm-bf16-tp2` (Arm 2) | `vllm-fp8` (Arm 3) |
|---|---|---|---|
| **Instance** | `fak-qwen-serve` (`a2-highgpu-1g`) | `fak-qwen38-bf16-tp2-perf` (`a2-highgpu-2g`) | `fak-qwen38-fp8-perf` (`a2-highgpu-1g`) |
| **GPU Configuration** | **1× datacenter GPU | 2× datacenter GPU (TP=2) | 1× datacenter GPU |
| **Engine** | **In-Kernel Native CUDA Forward** | vLLM 0.27.1 | vLLM 0.27.1 |
| **Model Quant** | `Qwen3.8-27B-Q4_K_M` | `qwen38-bf16` | `qwen38-fp8` |
| **Decode Throughput** | **19.66 tok/s** | 9.23 tok/s | 8.71 tok/s |
| **128-Token Latency** | **6.512 s** | 13.865 s | 14.691 s |
| **Relative Speedup** | **2.25× vs vLLM FP8**<br>**2.13× vs vLLM BF16** | 1.06× vs vLLM FP8 | Baseline |

## Findings

1. **Hardware Efficiency:** Native `fak-cuda` running `Qwen3.8-27B-Q4_K_M` delivers **19.66 tok/s** on a single A100—doubling the decode speed of vLLM FP8 (8.71 tok/s) and vLLM BF16 TP=2 (9.23 tok/s) while utilizing half the physical GPU hardware.
2. **Memory Footprint:** In vLLM, Qwen3.8-27B FP8 consumes 38.95 GiB during weight initialization, leaving virtually no headroom for large KV cache contexts on 40 GB GPUs without scaling to TP=2. In `fak-cuda`, Q4_K_M weights occupy 17.3 GB of VRAM, leaving $>22$ GB for paged KV allocation.
3. **Prefix Prefill Scaling:** vLLM achieved a 2.45× TTFT reduction under warm prefix caching (0.452s cold $\rightarrow$ 0.184s warm). In `fak-cuda`, chunked prefill tiles of 128 tokens ensure safety on 40 GB VRAM, with panel prefill optimization targeted in #8820.
