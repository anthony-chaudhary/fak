# Terminal-Bench 4 Reproduction Manual & Parity Envelope Specification

> **Audience:** Researchers, benchmark operators, and engineers replicating or auditing the Terminal-Bench 4 (TB4) evaluation comparing `fak harness` (in-kernel model engine) against `opencode` (`llama-server` baseline reference).

---

## 1. Executive Summary & Objective

This document provides the authoritative reproduction manual and parity envelope specification for running Terminal-Bench 4 on `fak`. Terminal-Bench 4 evaluates autonomous coding agents executing complex shell, file manipulation, build, test, and git operations inside isolated container sandboxes.

The evaluation compares two distinct execution arms under identical model weights, hyperparameters, and task environments:
1. **Arm A (fak Native):** `fak harness` executing directly against fak's native **in-kernel model engine** (`internal/model`), featuring zero-IPC inference, in-kernel KV cache tracking, and kernel security policy adjudication.
2. **Arm B (Baseline Reference):** `opencode` agent harness communicating with an external `llama-server` reference daemon serving the identical GGUF weights over `/v1/chat/completions`.

---

## 2. Hardware & Environment Requirements

### Recommended Hardware Envelope
- **GPU Acceleration:** NVIDIA GPU with $\ge$ 24GB VRAM (e.g. RTX 4090, A10, A100, L4) or Apple Silicon with unified memory $\ge$ 32GB (M3 Max / M4 Pro).
- **CPU:** 8+ physical cores (e.g. AMD Ryzen 9, Intel Core i9, or Apple M-series).
- **RAM:** Minimum 32GB system memory.
- **Disk Storage:** $\ge$ 100GB fast NVMe SSD storage (for OCI container layers and GGUF checkpoints).

### Operating System & Container Engine
- **Linux:** Ubuntu 22.04 LTS / Debian 12 with Docker Engine 24.0+ or Podman 4.5+.
- **macOS:** macOS 14+ with Docker Desktop or Colima.
- **User Permissions:** The executing user must have access to `/var/run/docker.sock` without sudo prompts (`sudo usermod -aG docker $USER`).

---

## 3. Strict Parity Invariants

To guarantee scientific neutrality and prevent benchmark gaming, both evaluation arms adhere to the strict parity invariants defined in `fak.tb4-run-contract.v1`:

| Axis | Arm A: fak Native Harness | Arm B: OpenCode Baseline Reference | Parity Invariant |
|---|---|---|---|
| **Model Weights** | Pinned Qwen3.8 GGUF (`Q4_K_M`) | Identical GGUF file | **Exact binary SHA-256 match** |
| **Decoding Temp** | `temperature = 0.0` (greedy argmax) | `--temp 0.0` | **Bit-exact greedy decoding** |
| **RNG Seed** | `seed = 42` | `--seed 42` | **Zero stochastic jitter** |
| **Sampling Bounds** | `top_p = 1.0`, `top_k = 1` | `--top-p 1.0 --top-k 1` | **No nucleus / stochastic sampling** |
| **Context Window** | `32,768` tokens | `--ctx-size 32768` | **Identical context capacity** |
| **Serving Concurrency** | Single tenant / single slot | `--parallel 1` | **Zero cross-request cache pollution** |
| **Network Isolation** | `--network none` (airgapped) | `--network none` (airgapped) | **Zero external internet access** |
| **Verification Oracle** | Dedicated post-run container oracle | Dedicated post-run container oracle | **Independent test oracle (neither agent grades itself)** |

---

## 4. Step-by-Step Reproduction Guide

### Step 1: Clone and Build fak
From the repository root, build the `fak` binary:
```bash
go build -o fak ./cmd/fak
```

### Step 2: Download Model Weights and Verify SHA-256
Download the sanctioned Qwen3.8-Coder GGUF checkpoint (e.g. from HuggingFace) and verify its cryptographic hash:
```bash
# Example model verification
sha256sum models/Qwen3.8-Coder-7B-Instruct-Q4_K_M.gguf
# Confirm hash matches the run contract pin
```

### Step 3: Run Preflight Checks
Run `fak bench tb4 preflight` to confirm that all prerequisites (container engine, model file, llama-server, and opencode) are present:
```bash
./fak bench tb4 preflight --model models/Qwen3.8-Coder-7B-Instruct-Q4_K_M.gguf --sha256 sha256:<expected-hash>
```

### Step 4: Execute Benchmark Run
Execute both arms across the task suite:
```bash
./fak bench tb4 run \
  --arm both \
  --model models/Qwen3.8-Coder-7B-Instruct-Q4_K_M.gguf \
  --dataset testdata/tb4bench/synthetic_suite.json \
  --out experiments/benchmarks/tb4-run-01
```

### Step 5: Evaluate Workspaces
Grade each task workspace using the cryptographically verified oracle test scripts:
```bash
./fak bench tb4 eval --run-dir experiments/benchmarks/tb4-run-01
```

### Step 6: Generate Comparative Report
Synthesize dual-arm metrics and generate authoritative comparison artifacts:
```bash
./fak bench tb4 compare \
  --fak-dir experiments/benchmarks/tb4-run-01/fak \
  --opencode-dir experiments/benchmarks/tb4-run-01/opencode \
  --out-json experiments/benchmarks/tb4-run-01/compare.json \
  --out-md experiments/benchmarks/tb4-run-01/compare.md
```

### Step 7: Inspect Run Trajectories
Use the trace inspector to replay turns or compare divergence side-by-side:
```bash
# Inspect single task run
./fak bench tb4 replay --run-dir experiments/benchmarks/tb4-run-01/fak --task tb4-synth-01-syntax-fix

# Side-by-side comparative replay
./fak bench tb4 replay --compare \
  --fak-dir experiments/benchmarks/tb4-run-01/fak \
  --opencode-dir experiments/benchmarks/tb4-run-01/opencode \
  --task tb4-synth-01-syntax-fix
```

---

## 5. Separation of Official Benchmark Scores from Telemetry

To ensure provenance honesty, metrics are partitioned into two strictly decoupled tiers:

1. **Official Benchmark Tier (Authoritative):**
   - `SolveRate`: Percentage of tasks marked `SOLVED` by the independent test oracle.
   - `TaskPassMap`: Map of `task_id -> bool`.
   - `MeanTaskDurationSeconds`: Average wall-clock execution time per task.
2. **Harness & Engine Telemetry Tier (Informational / Operational):**
   - `TotalPromptTokens` & `TotalCompletionTokens`.
   - `TokenEfficiency`: Tokens consumed per solved task.
   - `vDSOHits`: Invocations avoided via vDSO caching.
   - `CompactedTokens`: Tokens shed via context compaction.
   - `PolicyBlocks`: Actions blocked by kernel security policy.

Internal telemetry never modifies, inflates, or substitutes for official solve rates.

---

## 6. Troubleshooting Common Issues

| Symptom | Probable Cause | Corrective Action |
|---|---|---|
| `Cannot connect to the Docker daemon` | `/var/run/docker.sock` missing or permission denied | Start Docker service (`sudo systemctl start docker`) and verify user is in `docker` group. |
| `image does not contain a valid pinned digest` | Task manifest specifies mutable tag like `:latest` | Use immutable digest format: `image@sha256:<64-hex>`. |
| `Network unreachable` during execution | Expected behavior under `--network none` | Tasks must not rely on live internet; all dependencies must be pre-baked into image. |
| `port already in use` during llama-server start | Port collision on default port | Supervisor automatically picks ephemeral port via `FindFreePort()`; pass explicit `--port` if needed. |
| `TIMEOUT_AGENT` token emitted | Agent harness exceeded task turn budget | Increase `budget_turns` in task manifest or tune task prompt clarity. |

---

## 7. BENCHMARK-AUTHORITY.md Promotion Gate

No benchmark result from Terminal-Bench 4 may be cited publicly in `README.md`, marketing materials, or research papers until it has satisfied the following promotion criteria:
1. **Committed Run Contract:** An official contract (`fak.tb4-run-contract.v1`) with `LOCKED_EVALUATION` status is committed.
2. **Exact Weight Hashes:** Model file SHA-256 matches the pinned value.
3. **Reproducible Oracle Receipts:** Every task has a signed `GradingReceipt` verifying exit code 0.
4. **Committed Comparative Artifacts:** `tb4-compare-*.json` and `tb4-compare-*.md` are committed to `experiments/benchmarks/`.
5. **No UNCLASSIFIED Failures:** All failures must map to the closed taxonomy in `internal/tb4bench/taxonomy.go`.
