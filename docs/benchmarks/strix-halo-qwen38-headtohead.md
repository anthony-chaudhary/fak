---
title: "AMD Strix Halo Qwen3.8 fak-native vs Pinned llama.cpp A/B Benchmark"
description: "Physical execution baseline, prompt-token packet attestation, and head-to-head comparison status between fak-native and pinned llama.cpp on AMD Ryzen AI MAX+ 395 (Radeon 8060S / gfx1151 / 64GB UMA)."
---

# AMD Strix Halo Qwen3.8 fak-native vs Pinned llama.cpp A/B Head-to-Head

> **Status:** `NOT_COMPARABLE` / Hardware Validation `PENDING` (Honest Fail-Closed Accounting per P4 and #12098)  
> **Audience:** Serving runtime engineers, kernel authors, accelerator architects, benchmark auditors  
> **Machine-Readable Receipt:** [`docs/benchmarks/strix-halo-qwen38-headtohead-latest.json`](strix-halo-qwen38-headtohead-latest.json)  
> **Hardware Profile:** AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (`strix1` / `gfx1151` / 40 CUs / 64GB LPDDR5X UMA)  
> **Primary Envelope:** Dense Qwen3.8-27B Q4_K_M, 32,768 context, temperature 0, reasoning off, cold-per-request  

---

## 1. Executive Summary & Verdict

This benchmark tracks the paired same-token head-to-head comparison between **fak-native** (Vulkan/RADV compute pipeline) and an explicit **pinned llama.cpp comparator** on the physical local AMD Strix Halo appliance.

In accordance with repository governance (`BENCHMARK-GOVERNANCE.md`, `CLAIMS.md`), closed #9832 AMD scoreboard rules, and Issue #12098 Done Condition:
- **Verdict:** `NOT_COMPARABLE` (Scoreboard schema: `fak.qwen38.amd-scoreboard-report.v1`).
- **Hardware Validation:** `PENDING` (Timed A/B trials gated behind pinned comparator deployment and active service maintenance window).
- **Ratios Emitted:** `NONE` (Strictly zero ratio emitted for incomplete or pending comparator rows; ratio requires matched physical runs under exclusive GPU lease).

The first checkable step is **COMPLETE**: the canonical #9883 prompt-token packet (`fak.qwen38.prompt-token-packet.v1`) is materialized, bound to the exact physical artifact on the appliance (`sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`), sealed with canonical digest `f97bd502f5120620f5ec2b82f43921ebbb7d1c2988c5b09dd2ea5f53bf58b6ec`, and verified against template drift.

---

## 2. Physical Appliance Profile (`strix1`)

Physical hardware discovery was conducted via `fak-dev amd-strix-probe`:

| Dimension | Physical Specification | Observed Telemetry |
|---|---|---|
| **Host Appliance** | `strix1` (`strix-halo-fak.local`) | Measured SSH probe RTT: `431.80 ms` |
| **CPU Architecture** | AMD Ryzen AI MAX+ 395 | 16 physical Zen 5 cores, 32 threads, AVX-512 |
| **GPU Silicon** | AMD Radeon 8060S Graphics (`RADV STRIX_HALO`) | Target ISA: `gfx1151` (RDNA 3.5 APU) |
| **Compute Units (CUs)** | 40 CUs (2,560 Stream Processors) | 80 Matrix Cores (WMMA / Wave32) |
| **Physical Memory (RAM)** | 64 GB (67,028,504,576 bytes physical) | 256-bit LPDDR5X-8533 UMA bus |
| **UMA Allocatable Buffer** | 58,985,084,026 bytes (~54.94 GiB GTT ceiling) | Linux TTM dynamic aperture |
| **Vulkan ICD** | Mesa RADV Vulkan | `/usr/share/vulkan/icd.d/radeon_icd.json` |
| **Power Management (DPM)** | `power_dpm_level=manual` | Sustained APU performance governor |
| **Kernel Watchdog** | `amdgpu.lockup_timeout=-1` | Timeout disabled for long-context kernel compute |

### Bounded Device Correctness Witness
Under shared lease access on canonical lock `/tmp/fak-gpu.lease`:
- **Executable on Device:** `/usr/local/bin/fak` (Digest `sha256:ba4c3455d81da2eefde29f2572baa8ede55e97119d4b43d9a48181a3e0b47773`, build ref `fdf3597a11ba+uncommitted`).
- **Sub-Kernel Verification:** `fak-dev amd-strix-validate --subkernels=all --ablate=all --json` executed across all 21 Vulkan compute sub-kernels (`argmax`, `matmul_f32`, `matmul2_f32`, `matmul3_f32`, `q8_matmul`, `q8_matmul_wide`, `q8_matmul_vocab`, `q4k_matmul`, `q2k_matmul`, `rmsnorm`, `rmsnorm_matmul`, `rmsnorm_matmul2`, `rmsnorm_matmul3`, `swiglu`, `swiglu_matmul_add`, `rope`, `attention`, `qwen35_gdn_decode`, `qwen35_gdn_preprojected`, `qwen35_sequence_prefill`, `f16_kv_contiguize`).
- **Sub-Kernel Verdict:** `21/21 PASS` (Bit-exact or float32 logit cosine similarity >= 0.999999 against CPU reference oracle).

### Active Service Witness
- **Systemd Service:** `fak.service` (Main PID: `149670`, status `active (running)`).
- **Target Artifact:** `/var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf` (16 GB, SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`).
- **Active Memory:** 28.7 GB resident (peak 45.8 GB).
- **Live Inference Request Witness:** Bounded curl to `http://localhost:8080/v1/chat/completions` returned `chatcmpl-fak-1788840337978088257` (16 prompt tokens, 5 completion tokens).

---

## 3. First Checkable Step: Prompt-Token Packet Attestation (#9883)

To eliminate tokenizer drift, chat template discrepancies, and prefix-cache misattribution between fak-native and comparator arms, prompt token IDs are frozen prior to timed trials:

```json
{
  "schema": "fak.qwen38.prompt-token-packet.v1",
  "packet_id": "strix-halo-qwen38-27b-q4km-c1",
  "artifact_sha256": "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
  "tokenizer_identity": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "tokenizer_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "prompt_token_ids": [151644, 872, 198, 2610, 525, 264, 10925, 13, 151645, 198],
  "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
  "stop_token_ids": [151645, 151643],
  "context_budget": {
    "context_tokens": 32768,
    "context_budget_bytes": 34359738368
  },
  "generation_controls": {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "max_output_tokens": 128,
    "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
    "stop_token_ids": [151645, 151643]
  },
  "packet_digest": "f97bd502f5120620f5ec2b82f43921ebbb7d1c2988c5b09dd2ea5f53bf58b6ec"
}
```

Attestation validation:
- Verified via `qwen38quantrun.ValidatePromptPacketAttestation`: any divergence in token IDs, artifact hash, tokenizer digest, stop tokens, context budget, or generation controls rejects execution and halts comparison.

---

## 4. Typed Blockers & Fail-Closed Disposition

The head-to-head evaluation currently reports `NOT_COMPARABLE` with hardware validation `PENDING` due to four explicit, typed blockers:

1. **`PINNED_COMPARATOR_UNAVAILABLE`**:
   - The reference comparator binary (`llama.cpp` Vulkan/RADV build, e.g., `llama-bench` or `llama-server`) is not installed or present in `$PATH` on `strix1` (`which llama-bench` returned empty).
   - Under repository doctrine, comparator execution must be real and local to the physical appliance under the same driver, OS, and thermal conditions; external or modeled timings are disallowed.

2. **`SERVICE_PROTECTION_LOCK`**:
   - `fak.service` (PID 149670) is actively serving `Qwen3.8-27B-Q4_K_M.gguf` under 28.7 GB resident memory.
   - Operating instructions (`AGENTS.md`) strictly mandate: *"Do not stop or restart a managed inference server during active sessions. Before stopping, restarting, or reconfiguring any locally or remotely managed model inference endpoint or server, verify that all dependent sessions have completed or ownership has been explicitly handed off."*
   - Running full-model multi-slot interleaved trials while `fak.service` occupies half of the 64GB UMA memory would induce memory thrashing or OOM panics. A dedicated maintenance window with ownership handoff is required.

3. **`DEPENDENCY_OPEN_12096`**:
   - Issue #12096 (*"bench(modelbench): emit live canonical Qwen3.8 Vulkan receipts from raw-decode"*) remains OPEN.
   - Live canonical Vulkan decode receipt binding from `cmd/modelbench/raw_decode.go` into `internal/compute/qwen38_vulkan_decode_receipt.go` is required for end-to-end trace validation.

4. **`DEPENDENCY_OPEN_9883`**:
   - Issue #9883 (*"bench(qwen38): freeze a cross-engine prompt-token packet for AMD product-path trials"*) attestation logic is committed, but live multi-slot cross-engine execution on Strix Halo requires comparator presence.

---

## 5. Frozen Evaluation Envelope & Target Concurrency Matrix

When the timed A/B trial is executed under exclusive lease (`flock -w 30 -x /tmp/fak-gpu.lease`), it evaluates concurrency slots $c \in \{1, 4, 8\}$ against predeclared challenge bars:

| Concurrency ($c=np$) | External Challenge Bar (128GB Envelope) | fak-native Target Envelope (64GB Appliance) | Pinned llama.cpp RADV Reference | Comparison Metric | Statistical Gate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$c=1$** | 16.65 tok/s | Q4_K_M + Vulkan RADV (64GB) | Pinned Vulkan build (`GGML_VULKAN=ON`) | Aggregate accepted output tok/s | $\ge 5$ trials, CV $\le 5\%$, 95% LCB > 1.00 |
| **$c=4$** | 24.13 tok/s | Q4_K_M + Vulkan RADV (64GB) | Pinned Vulkan build (`GGML_VULKAN=ON`) | Aggregate accepted output tok/s | $\ge 5$ trials, CV $\le 5\%$, 95% LCB > 1.00 |
| **$c=8$** | 29.42 tok/s | Q4_K_M + Vulkan RADV (64GB) | Pinned Vulkan build (`GGML_VULKAN=ON`) | Aggregate accepted output tok/s | $\ge 5$ trials, CV $\le 5\%$, 95% LCB > 1.00 |

*Note on Envelopes:* The external challenge points from `nabe2030/qwen38-evo-x2` were observed on a 128GB Strix Halo platform. Exceeding those bars on the local 64GB appliance is an achievement, but local engine superiority is judged solely by the paired local ratio against the pinned comparator arm.

### Statistical & Quality Gates:
- 3 uncounted warmups followed by $\ge 5$ uncontended measured repetitions per cell.
- Alternating A/B order to isolate thermal and DPM drift.
- Aggregate accepted output tok/s: accepted completion tokens divided by total wall-clock duration from first admission through final completion.
- Zero corrupt/non-UTF-8 tokens, finite logits, solve rate non-inferior to reference.

---

## 6. Runbook & Exact Next Steps

To transition this benchmark from `NOT_COMPARABLE` / `PENDING` to a verified paired speedup verdict:

1. **Deploy Pinned Comparator on Strix Halo:**
   ```bash
   ssh strix1 "flock -w 30 -x /tmp/fak-gpu.lease bash -c 'git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && cmake -B /tmp/llama.cpp/build -DGGML_VULKAN=ON -S /tmp/llama.cpp && cmake --build /tmp/llama.cpp/build --config Release -j32 --target llama-bench && sudo cp /tmp/llama.cpp/build/bin/llama-bench /usr/local/bin/'"
   ```

2. **Schedule Maintenance Window & Quiesce Active Service:**
   ```bash
   ssh strix1 "sudo systemctl stop fak.service"
   ```

3. **Acquire Exclusive Lease & Run A/B Matrix:**
   ```bash
   ssh strix1 "flock -w 60 -x /tmp/fak-gpu.lease /usr/local/bin/fak bench qwen38-strix-headtohead --model /var/lib/fak/models/Qwen3.8-27B-Q4_K_M.gguf --packet docs/benchmarks/strix-halo-qwen38-headtohead-latest.json --concurrency 1,4,8 --trials 5 --out /tmp/strix-headtohead-runs.json"
   ```

4. **Restore Serving Service:**
   ```bash
   ssh strix1 "sudo systemctl start fak.service"
   ```

5. **Evaluate & Seal Scoreboard:**
   ```bash
   qwen38campaign --amd-scoreboard --config /tmp/strix-headtohead-runs.json --report docs/benchmarks/strix-halo-qwen38-headtohead-latest.json
   ```
