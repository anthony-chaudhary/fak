# Witness: AgentX-Style Benchmark on GCP A100 Qwen3.8-27B Serving

**Date**: 2026-09-05  
**Verdict**: `VERIFIED_PASS`  
**Schema**: `fak.agentx-benchmark-receipt.v1`  
**Execution Plane**: GCP Cloud GPU Infrastructure (`fak-qwen-serve`, 1× NVIDIA A100-SXM4-40GB)  
**Model & Engine**: `Qwen3.8-27B-Q4_K_M` via `fak-inkernel-cuda`  
**Witness Test**: `go test -v ./docs/_witnesses/agentx-gcp-a100-qwen38/...`

---

## 1. Overview & Borrowed AgentX Methodology

Following the SemiAnalysis InferenceX AgentX borrow study (`docs/notes/BORROW-BENCHMARK-SERVING-METRICS-INFERENCEX-STUDY-2026-07-13.md`), this benchmark treats multi-agent execution as a continuous serving workload. It captures:
1. **Client/Server Request Lifecycle**: Explicit phase decomposition across `queue_wait`, `agent_execution`, and `evaluation` with nested interval verification.
2. **Streaming Interactivity**: Full-response inter-token latency (ITL) quantiles (p50, p90, p95, p99, max) and normalized generation interactivity.
3. **Prefix Caching Speedup**: Quantitative TTFT acceleration comparing cold first turns with warm subsequent turns sharing multi-turn history.
4. **Strict Fail-Closed Validation**: Rejects missing metrics, non-monotonic token arrival timestamps, or synthetic zeros.

---

## 2. Key Measured Performance Highlights

- **Concurrency Grid**: 5 concurrent agents running 5 multi-turn steps each (**25 total requests**), achieving **100.0% success rate (25/25)** with zero dropped tokens or server crashes.
- **Cold vs Warm TTFT**:
  - Cold Turn Mean TTFT: **1,184.55 ms**
  - Warm Turn Mean TTFT: **246.75 ms**
  - **Prefix Reuse Speedup**: **4.80×** reduction in Time-to-First-Token.
- **Inter-Token Latency (ITL)**:
  - ITL p50: **15.13 ms** (~66.1 tok/s decode rate)
  - ITL p95: **17.68 ms**
  - Normalized Interactivity: **65.77 tok/s**
- **Throughput Under Load**:
  - Request Throughput: **5.30 req/s**
  - Completion Token Throughput: **169.48 tok/s**
  - Cluster Token Throughput: **8,304.56 tok/s** (including prompt prefill and reused prefix tokens)

---

## 3. Artifact Integrity

- `receipt.json`: SHA-256 `1af84415c89f94df31214b0970a9a95befc0c9e2afbea9febf40a5ca1e74ae9d`
