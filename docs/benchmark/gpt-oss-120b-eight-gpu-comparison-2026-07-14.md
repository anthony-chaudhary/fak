# GPT-OSS-120B on eight GPUs: observed reference and theory context

**Reference date:** 2026-07-14  
**Evidence class:** observed campaign measurements plus modeled hardware ceilings  
**Purpose:** a comparison baseline for later `fak` self-hosted model runs

## Verdict

- The served model was `openai/gpt-oss-120b`, an approximately 117–120B-parameter mixture-of-experts model with about 5.1B parameters active per token.
- The reference deployment used SGLang `0.5.13.post1`, tensor parallelism 8, and eight 40 GB datacenter GPUs.
- The best point in the stepped sweep was **3,964.7 completion tok/s at concurrency 64**, with **4.162 s p95** and no request errors in that sweep point.
- A separate 10-minute concurrency-48 run sustained **3,146.7489 completion tok/s**, completing **9,860/9,860 requests** without an error.
- These are observed serving results, not a tuned `fak`-versus-direct comparison and not a production-default recommendation.
- Against theory, dense BF16 peak is the wrong headline denominator for autoregressive MoE decode. An intentionally optimistic active-weight bandwidth model is more informative, but remains nonliteral because batching reuses weights and real serving moves much more than active expert weights.

## Reference deployment

| Field | Reference value |
|---|---|
| Model | `openai/gpt-oss-120b` |
| Architecture scale | approximately 117–120B total parameters; approximately 5.1B active parameters/token |
| Runtime | SGLang `0.5.13.post1` |
| Accelerators | 8× 40 GB datacenter GPUs |
| Parallelism | tensor parallelism 8 |
| Compute datatype context | BF16 |
| KV cache | FP8 E5M2 |
| Attention backend | Triton |
| Chunked prefill | 4,096 tokens |
| CUDA graphs | enabled |
| Orchestration | detached headless workers; 12 initial concurrent workers and at least 10 maintained through the primary run |

Twenty-five roles were launched during the campaign. An initial provider-backed worker wave was abandoned after rate-limit/context exhaustion; the successful measurement wave used detached Codex workers. This orchestration fact is context, not evidence that worker count caused the serving result.

## Observed results

### Stepped concurrency sweep

| Concurrency | Completion tok/s | p95 end-to-end latency | Errors |
|---:|---:|---:|---:|
| 8 | 908.8 | 2.361 s | 0 |
| 16 | 1,536.9 | 2.671 s | 0 |
| 32 | 2,530.3 | 3.248 s | 0 |
| 64 | 3,964.7 | 4.162 s | 0 |

This is one observed sweep, not proof that 64 is the maximum stable concurrency. Throughput rose over the tested range while p95 latency also rose.

### Ten-minute sustained point

| Field | Observed value |
|---|---:|
| Concurrency | 48 |
| Duration | 601.6114 s |
| Successful requests | 9,860 / 9,860 |
| Errors | 0 |
| Completion tokens | 1,893,120 |
| Completion throughput | 3,146.7489 tok/s |
| p50 / p95 / p99 | 2.9627 / 3.1317 / 3.2586 s |
| Artifact size | 843,499 bytes |
| Artifact SHA-256 | `bd3bf5bf322ce1a397f6b752cb832d32db15fbd5860ff95250428bac9fd8fc6a` |

### Concurrency-64 replicate

| Field | Observed value |
|---|---:|
| Concurrency | 64 |
| Output cap | 256 tokens |
| Duration | 304.7477 s |
| Successful requests | 3,136 / 3,136 |
| Errors | 0 |
| Completion tokens | 802,816 |
| Completion throughput | 2,634.3628 tok/s |
| p50 / p95 / p99 | 6.3273 / 7.8824 / 7.9504 s |
| Artifact size | 267,142 bytes |
| Artifact SHA-256 | `9ffa727ee2c2106a6de788252a72cf0074d97ea3e60b8934b46c6869ea9c2312` |

An independent validator exactly recomputed the replicate's row count, success/error counts, completion-token sum, and percentiles from the artifact. During this run, all eight GPUs were observed at 94–100% utilization, about 249–268 W each, and 37–41 °C.

**Comparability warning:** the sustained concurrency-48 point and concurrency-64 replicate used different prompts, output caps, and shared-service conditions. Their throughput difference is not a controlled concurrency result. Use the stepped sweep—not the cross-run delta—when discussing observed scaling, and still label that sweep as a single replicate.

## Comparison with theory

The following calculations are modeled context. They are not measured device FLOPs, HBM bytes, or `fak` efficiency.

### Hardware ceilings

Using nominal per-GPU reference accelerator values of approximately 312 TFLOP/s BF16 tensor-core compute, 1.55–1.56 TB/s HBM bandwidth, and a 400 W board-power ceiling gives these eight-GPU ceilings:

| Resource | Ideal aggregate ceiling |
|---|---:|
| BF16 tensor-core compute | approximately 2.50 PFLOP/s |
| HBM bandwidth | approximately 12.4 TB/s |
| GPU board power | 3.2 kW |

These are component specifications, not simultaneously attainable application throughput.

### Compute-oriented estimate

A deliberately rough active-parameter estimate is:

```text
2 FLOP/parameter × 5.1B active parameters ≈ 10.2 GFLOP/token
3,964.7 token/s × 10.2 GFLOP/token ≈ 40.4 TFLOP/s
40.4 TFLOP/s ÷ 2,496 TFLOP/s ≈ 1.6%
```

The **1.6% figure is not a useful efficiency score**. It compares an autoregressive, routed MoE decode workload with a dense tensor-core peak. Decode also pays for sequential dependencies, memory movement, attention/KV work, expert routing, inter-GPU collectives, and many operations that do not continuously occupy dense GEMM units. The small fraction therefore does not show that 98.4% of practical performance is recoverable.

### Idealized active-weight bandwidth estimate

For intuition only, assume each generated token streams 5.1B active parameters once at an idealized 4-bit payload:

```text
5.1B parameters × 0.5 byte ≈ 2.55 GB/token
12,400 GB/s ÷ 2.55 GB/token ≈ 4,860 token/s
```

| Observed point | Completion tok/s | Fraction of optimistic 4,860 tok/s bound |
|---|---:|---:|
| Stepped sweep, concurrency 64 | 3,964.7 | approximately 82% |
| Sustained, concurrency 48 | 3,146.7489 | approximately 65% |
| Replicate, concurrency 64 | 2,634.3628 | approximately 54% |

Call these **fractions of an idealized bound**, never measured bandwidth efficiency. The approximation omits attention and KV traffic, expert-routing imbalance, tensor-parallel collectives, activations and non-MoE layers, quantization metadata and dequantization, scheduler/HTTP work, padding, variable sequence lengths, and shared-service traffic. Conversely, batching permits weight reuse across tokens, so the naïve `2.55 GB/token` assumption is itself nonliteral. The bound is useful as an order-of-magnitude check, not as a roofline proof.

### Power context

The concurrency-64 replicate's per-GPU observations imply roughly 2.0–2.1 kW aggregate GPU draw, or about 64–67% of the 3.2 kW board-power ceiling. High reported GPU utilization with sub-maximum board power is consistent with a decode workload that keeps devices busy without continuously exercising peak dense tensor-core throughput. It is not, by itself, a causal diagnosis or an energy-efficiency result.

## How a future `fak` run should compare

Use three distinct comparators:

1. **Observed reference:** the measurements above show what this model/runtime/hardware campaign actually delivered.
2. **Theory context:** report compute, bandwidth, and power ratios as bounded diagnostics, with all assumptions visible.
3. **Real alternative:** for a `fak` performance claim, compare with tuned direct serving under the same workload and environment. The present campaign does not supply that matched A/B.

A future run is directly comparable only when it records or controls all of the following:

- identical model revision and weight/quantization digest;
- identical runtime version and serving configuration;
- identical hardware count/type, tensor parallelism, and clocks/power policy;
- identical prompt corpus and input/output-length distributions;
- identical concurrency or arrival-rate schedule and output cap;
- the same warm-up, prefix/cache state, batching policy, and service-contention conditions;
- separate prefill, decode, and end-to-end measurements where the runtime exposes them;
- completion tok/s, request throughput, p50/p95/p99 latency, error/timeout counts, and duration;
- GPU utilization, memory use, power, and—when available—energy per completed request/token;
- an immutable raw-artifact digest and an independent recomputation of headline aggregates.

If these controls differ, mark the comparison as directional and enumerate the mismatches. Do not turn a cross-workload ratio into a speedup claim.

### Machine-readable comparison record

Future evidence can copy this shape. `null` means not measured, not zero.

```json
{
  "schema": "fak-model-serving-comparison/1",
  "reference_date": "2026-07-14",
  "provenance": "observed",
  "model": {
    "id": "openai/gpt-oss-120b",
    "revision_or_digest": null,
    "total_parameters_approx": 117000000000,
    "active_parameters_per_token_approx": 5100000000
  },
  "runtime": {
    "name": "sglang",
    "version": "0.5.13.post1",
    "tensor_parallel": 8,
    "compute_dtype": "bf16",
    "kv_cache_dtype": "fp8_e5m2",
    "attention_backend": "triton",
    "chunked_prefill_tokens": 4096,
    "cuda_graphs": true
  },
  "hardware": {
    "accelerator": "40 GB datacenter GPU",
    "count": 8
  },
  "workload": {
    "concurrency": 48,
    "duration_seconds": 601.6114,
    "prompt_corpus_digest": null,
    "max_output_tokens": null
  },
  "results": {
    "requests_successful": 9860,
    "requests_error": 0,
    "completion_tokens": 1893120,
    "completion_tokens_per_second": 3146.7489,
    "latency_seconds": {
      "p50": 2.9627,
      "p95": 3.1317,
      "p99": 3.2586
    },
    "gpu_utilization_percent_range": null,
    "gpu_power_watts_each_range": null,
    "energy_joules_per_request": null
  },
  "witness": {
    "artifact_bytes": 843499,
    "sha256": "bd3bf5bf322ce1a397f6b752cb832d32db15fbd5860ff95250428bac9fd8fc6a"
  },
  "comparability": {
    "same_model_digest": null,
    "same_runtime_config": true,
    "same_prompt_corpus": null,
    "same_length_distribution": null,
    "same_service_contention": null,
    "controlled_ab": false
  }
}
```

## What is not yet established

- **Not yet:** a tuned, workload-matched direct-serving versus `fak` A/B speedup or savings claim.
- **Not yet:** maximum stable concurrency or saturation throughput; the stepped sweep ended at 64 and has one replicate.
- **Not yet:** measured achieved FLOP/s, HBM bandwidth, communication time, or kernel-level roofline attribution.
- **Not yet:** energy per token/request; sampled board power is not an integrated energy measurement.
- **Not yet:** a production service-level objective or recommended default configuration.

## Public evidence and specification sources

- [Campaign issue #4367](https://github.com/anthony-chaudhary/fak/issues/4367)
- [High-utilization checkpoint](https://github.com/anthony-chaudhary/fak/issues/4367#issuecomment-4974597504)
- [Concurrency-64 replicate and independent validation](https://github.com/anthony-chaudhary/fak/issues/4367#issuecomment-4974771437)
- [Runtime observations](https://github.com/anthony-chaudhary/fak/issues/4367#issuecomment-4974535580)
- [OpenAI GPT-OSS model card](https://huggingface.co/openai/gpt-oss-120b)
- [Vendor accelerator specifications](https://www.nvidia.com/en-us/data-center/a100/)

Raw campaign evidence is retained in the private campaign archive. This public reference intentionally excludes endpoints, host identifiers, credentials, process commands, and private control-channel details.
