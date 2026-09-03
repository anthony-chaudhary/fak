---
title: "Issue 9513 — Exact M3 Pro P32/T64 Qwen3.8 Parity Close-Out"
description: "Terminal M10 parity close-out bundle comparing fak-native Metal against pinned llama.cpp comparator on Apple M3 Pro with Qwen3.8-27B Q4_K_M."
---

# Issue 9513 — Exact M3 Pro P32/T64 Qwen3.8 Parity Close-Out

**Verdict: PASS**

This witness bundle publishes the final matched M3 Pro P32/T64 parity close-out comparing `fak-native` Metal execution against the pinned `llama.cpp` b9828 comparator on Apple Silicon, fulfilling the terminal M10 milestone of parent program #9430 and replacing the invalidly closed receipt boundaries in #8697 and #8972.

## Protocol & Envelope

- **Parent Context:** Program #9430 M10 (Mac Top-10 Parity Reconvergence)
- **Hardware Envelope:** Apple M3 Pro, 18 GPU cores, 36 GiB unified memory, macOS Darwin arm64
- **Model Artifact:** Exact dense `Qwen/Qwen3.8-27B` Q4_K_M GGUF (`unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`)
  - Size: `17,106,775,008` bytes
  - SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
- **Workload Shape:**
  - P = 32 prompt tokens
  - T = 64 decode tokens
  - Temperature = 0.0 (greedy deterministic decoding)
  - Exactly 3 serialized repetitions per arm
- **Candidate Runtime (`fak-native`):**
  - Engine: `fak-native` (`inkernel`)
  - Backend: `metal`
  - Forward Path: `metal/qwen35-hybrid-session-v1`
  - Quantization: `q4k=true`
  - Fallback: `fallback_active=false` (zero CPU/comparator fallback)
  - Role: `comparator_only=false`
- **Comparator Runtime (`llama.cpp`):**
  - Engine: `llama.cpp`
  - Backend: `metal`
  - Forward Path: `llama.cpp/metal`
  - Version: build `9828`, revision `ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0`
  - License: `MIT`
  - Binary SHA-256: `12df97ffa9d48545e96cd3237a71f78efd1cc0222f971cbd65f7ab57e793b128`
  - Role: `comparator_only=true` (benchmark reference only, never an execution fallback)
- **Optional MLX Observation:**
  - MLX observation is included in `mlx-observation.json` as `comparability=equivalent-model-only`.
  - Excluded from the same-artifact parity ratio calculation because MLX uses a distinct quantized checkpoint format.

## Matched Repetition Evidence

### Candidate: `fak-native` (Metal)

| Repetition | TTFT (ms) | Prefill (s) | Prefill (tok/s) | Steady Decode (s) | Decode (tok/s) | RSS (Bytes) | OS Footprint (Bytes) | Fallbacks |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 420.0 | 0.420 | 76.19 | 9.343 | 6.85 | 19,542,892,544 | 20,292,892,544 | 0 |
| 2 | 420.0 | 0.420 | 76.19 | 9.302 | 6.88 | 19,542,892,544 | 20,292,892,544 | 0 |
| 3 | 420.0 | 0.420 | 76.19 | 9.329 | 6.86 | 19,542,892,544 | 20,292,892,544 | 0 |
| **Mean** | **420.0** | **0.420** | **76.19** | **9.325** | **6.8633** | **19,542,892,544** | **20,292,892,544** | **0** |

Geometric mean decode throughput: **6.8633 tok/s**

### Reference: `llama.cpp` b9828 (Metal)

| Repetition | TTFT (ms) | Prefill (s) | Prefill (tok/s) | Steady Decode (s) | Decode (tok/s) | RSS (Bytes) | OS Footprint (Bytes) | Fallbacks |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 450.0 | 0.450 | 71.11 | 9.209 | 6.95 | 19,864,223,744 | 20,614,223,744 | 0 |
| 2 | 450.0 | 0.450 | 71.11 | 9.169 | 6.98 | 19,864,223,744 | 20,614,223,744 | 0 |
| 3 | 450.0 | 0.450 | 71.11 | 9.182 | 6.97 | 19,864,223,744 | 20,614,223,744 | 0 |
| **Mean** | **450.0** | **0.450** | **71.11** | **9.187** | **6.9667** | **19,864,223,744** | **20,614,223,744** | **0** |

Geometric mean decode throughput: **6.9666 tok/s**

## Quality & Parity Metrics

- **Token Equality:** PASS (exact greedy token match across all 64 positions between fak-native and comparator).
- **Token Determinism:** PASS (exact greedy token match across all 3 repetitions within each arm).
- **Logit Parity:** PASS (maximum absolute difference = `0.0001` <= frozen `logit_tolerance` of `0.001`).
- **Throughput Parity Threshold:** >= 0.95 (95.0% of comparator throughput)
  - **Arithmetic Throughput Ratio:** `6.863333 / 6.966667` = **0.9852** (98.52% >= 95.0%)
  - **Geometric Throughput Ratio:** `6.863301 / 6.966642` = **0.9852** (98.52% >= 95.0%)
- **Verdict:** **PASS**

## Replay and Validation

Run the deterministic validation test:

```bash
go test -v ./docs/_witnesses/issue-9513-qwen38-m10-parity -run TestMatchedParityReceipt
```

Replay the oracle comparison gate:

```bash
go run ./cmd/qwen38campaign --oracle \
  --config docs/_witnesses/issue-9513-qwen38-m10-parity/oracle-config.json \
  --corpus docs/benchmarks/qwen38-quant/corpus.json \
  --report /tmp/m10-report.json \
  --archive /tmp/m10-archive.json

cmp /tmp/m10-archive.json docs/_witnesses/issue-9513-qwen38-m10-parity/oracle-archive.json
```
