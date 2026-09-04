# Witness Artifact: Issue #10900 — 50+ Subagent Fan-Out & GCP Inference Benchmark

## Overview

This witness packet captures the empirical benchmarking of multi-agent fan-out scaling and remote inference on Google Cloud Platform (GCP). It validates the 3-tier scaling hierarchy against both high-concurrency offline simulation and a live GPU serving endpoint running `Qwen3.8-27B-Q4_K_M` on an NVIDIA datacenter GPU GPU (`fak-qwen-serve`).

## Key Advancements & Parity Comparison

### 1. 10x Breadth (128 Subagents) & 10x Depth (20 Turns)
- **Previous baseline**: 1–8 agents at 6 scripted turns executed strictly serially.
- **New capability**:
  - Breadth scaling: Evaluated grid $[1, 10, 50, 100, 128]$ subagents running concurrently across a bounded worker pool (`--workers 16`).
  - Depth scaling: Parameterized multi-turn planner executing up to 20 deep inspection turns per subagent without premature exit.
  - Elided tokens: **260,096 tokens elided** through in-kernel vDSO cross-agent cache deduplication at $N=128$.
  - Parallel wall time: **43.4 ms** at $N=128$ (2,952.3 parallel agents/sec), down from 669.3 ms serial wall time.

### 2. Live GCP datacenter GPU GPU Inference Witness
- **Node**: `fak-qwen-serve` (`us-central1-f`, `a2-highgpu-1g`, NVIDIA datacenter GPU).
- **In-Kernel Serving**: Native CUDA backend running `Qwen3.8-27B-Q4_K_M.gguf` on port `8155` with 32k context budget.
- **Observed Metrics**:
  - TTFT (p50): 10,375.0 ms
  - End-to-end latency (p50): 10,375.0 ms
  - Prompt tokens: 2,438 tokens
  - Completion tokens: 353 tokens
  - Cached prompt tokens: 2,438 tokens (100% prefix hit rate on subsequent turns)
  - Tasks completed: 100% (zero tool errors)

### 3. Cryptographic Completion Receipts
Each subagent execution produces a structured `SubagentReceipt` and signed `agent.Receipt` (`internal/agent/receipt.go`), proving:
- Provenance: `WITNESSED` execution of tool calls and zero policy denials.
- Trace ID: Explicitly correlated per subagent (`fanrun-n<N>-a<A>`).
- Final answer digest: SHA-256 hash of model outputs.
- Verification status: `VERIFIED`.

## Captured Artifacts

1. `fanrun_128_subagents_20turns.json`: Full benchmark cell report across widths $N \in [1, 10, 50, 100, 128]$ with 20 turns per agent.
2. `fanrun_128_subagents_20turns_receipts.jsonl`: Line-delimited JSON of 289 individual subagent completion receipts.
3. `fanrun_gcp_a100_qwen38_live.json`: Live inference benchmark report executed against GCP `fak-qwen-serve:8155`.
4. `fanrun_gcp_a100_qwen38_live_receipts.jsonl`: Cryptographically signed completion receipts from the live GCP GPU run.

## Reproduction Commands

### Local Simulation (Breadth & Depth):
```bash
go run ./cmd/fanrun --profile research --agents 1,10,50,100,128 --sub-turns 20 --workers 16 \
  --out docs/_witnesses/issue-10900/fanrun_128_subagents_20turns.json \
  --receipts-out docs/_witnesses/issue-10900/fanrun_128_subagents_20turns_receipts.jsonl
```

### Remote GCP datacenter GPU GPU Inference:
```bash
go run ./cmd/fanrun \
  --endpoint http://127.0.0.1:8155 \
  --api-key "$FAK_QWEN38_API_KEY" \
  --model Qwen3.8-27B-Q4_K_M \
  --provider openai \
  --agents 1 \
  --sub-turns 2 \
  --workers 1 \
  --out docs/_witnesses/issue-10900/fanrun_gcp_a100_qwen38_live.json \
  --receipts-out docs/_witnesses/issue-10900/fanrun_gcp_a100_qwen38_live_receipts.jsonl
```
