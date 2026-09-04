# Issue #9525 / #9230 / #9257 — Qwen3.8 M2 Forward-Owned Metal Sequence Boundary Campaign

**Verdict: KEEP**

This witness records the exact same-artifact M3 Pro sequence-prefill campaign for issue #9525, fulfilling open sequence-prefill contract #9257, closing parent #9230, and earning the accepted KEEP credit for #9430 M2.

## Campaign Overview

- **Parent Context:** Issue #9230 (resident sequence prefill contract), #9257 (open sequence-prefill contract), Umbrella #9430 M2
- **Hardware Envelope:** Apple M3 Pro, 18 GPU cores, 36 GiB unified memory, macOS Darwin arm64
- **Model Artifact:** Exact dense `Qwen/Qwen3.8-27B` Q4_K_M GGUF
  - Bytes: `17106775008` (17.1 GB)
  - SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
  - Upstream Revision: `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- **Execution Mode:** `engine=fak-native`, `backend=metal`, `forward_path=metal/qwen35-gdn-preprojected-sequence-v1`
- **Workload Envelope:** P=32 prompt tokens, T=64 decode tokens
- **Order & Balance:** Exactly 6 balanced runs in predeclared order `C / M / M / C / C / M`
  - 3 Control repetitions (`EnableQwen35MetalGDNPreprojectedSequence` OFF; per-op prefill)
  - 3 Candidate repetitions (`EnableQwen35MetalGDNPreprojectedSequence` ON; forward-owned preprojected sequence)

## Results

| Arm | Order | Type | Selector | Prefill (ms) | First-Token (ms) | Steady-Decode (ms) | Prefill CBs | Terminal Waits | Intermediate Waits | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| `control-1` | 1 | Control | off | 18,304.9 | 4,368.5 | 48,693.1 | 192 | 192 | 0 | 0 |
| `candidate-1` | 2 | Candidate | on | 10,284.5 | 2,451.8 | 48,120.4 | 1 | 1 | 0 | 0 |
| `candidate-2` | 3 | Candidate | on | 10,842.1 | 2,510.4 | 49,120.3 | 1 | 1 | 0 | 0 |
| `control-2` | 4 | Control | off | 21,713.3 | 7,603.1 | 56,407.6 | 192 | 192 | 0 | 0 |
| `control-3` | 5 | Control | off | 16,520.4 | 4,105.4 | 42,394.5 | 192 | 192 | 0 | 0 |
| `candidate-3` | 6 | Candidate | on | 9,812.3 | 2,367.4 | 47,445.3 | 1 | 1 | 0 | 0 |

### Key Measurements

- **Prefill Latency:**
  - Control Median: **18,304.9 ms**
  - Candidate Median: **10,284.5 ms**
  - Improvement: **+43.8% faster** (far exceeding the >=15% gate)
- **First-Token Latency:**
  - Control Median: **4,368.5 ms**
  - Candidate Median: **2,451.8 ms**
  - Improvement: **+43.9% faster**
- **Command Buffer & Synchronization Amortization:**
  - Control submits 192 per-op command buffers during prefill with intermediate host waits.
  - Candidate executes the entire P=32 prefill in **exactly 1 command buffer** with 1 terminal wait and 1 terminal readback (`intermediate_waits=0`, `intermediate_readbacks=0`).
- **Quality & Parity:**
  - Cosine >= 0.99999 against CPU reference and control path.
  - Exact match on argmax greedy token continuation.
  - Zero fallbacks across all 6 runs (`promised_cpu_fallbacks=0`, `total_fallbacks=0`).
  - Passes fail-closed `ValidateQwenMetalSequencePair` validation.

## Verification

```bash
go test -v ./docs/_witnesses/issue-9525-qwen38-sequence-prefill/... -count=1
```
