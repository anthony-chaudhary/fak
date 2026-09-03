---
title: "Popular local hardware setup guide for benchmarks"
description: "Setup prerequisites, execution workflow, and privacy expectations for running local benchmarks across Apple Silicon, NVIDIA RTX, AMD Radeon, and CPU systems."
---

# Popular local hardware setup guide for benchmarks

This guide covers preparing local developer machines and workstations to run and submit benchmarks using `fak bench local`. It covers hardware setup across Apple Silicon, NVIDIA GeForce/RTX, AMD Radeon, and CPU-only architectures.

Numbers are only authoritative in [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md); every claim carries a tag in [`../../CLAIMS.md`](../../CLAIMS.md). For engine and backend classification, see [`../HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md) and the [native inference invariant](../native-inference-goal.md).

## Platform support matrix

| Platform | Compute Backend | Support Level | Prerequisites | Typical Devices |
|---|---|---|---|---|
| **Apple Silicon** | Metal (`internal/metalgemm`) | Supported / Proven | macOS 14+, Xcode CLI tools (`xcode-select --install`) | M1/M2/M3/M4 (Base, Pro, Max, Ultra) |
| **NVIDIA GeForce/RTX** | CUDA (`internal/compute`) | Supported / Proven | CUDA Toolkit 12+, NVIDIA Driver 535+ | RTX 3060–4090, RTX 50-series, A100 |
| **AMD Radeon** | Vulkan (`internal/compute`) / ROCm | Experimental | Vulkan SDK / ROCm 6+, AMD Adrenalin / ROCm driver | Radeon RX 7600–7900 XTX |
| **CPU-Only** | Portable SIMD (`internal/compute`) | Reference-Only | Go 1.26+ toolchain | x86_64 (AVX-512/AVX2), arm64 (NEON) |

### Engine selection and native invariant

Under the [native inference invariant](../native-inference-goal.md), benchmark commands must explicitly specify the engine. CPU execution is an explicit choice (`--engine cpu-ref` or `--engine fak-native`), never an automatic silent fallback when an accelerator encounters errors.

## End-to-end benchmark workflow

The local benchmark flow consists of four stages: `inventory` → `run` → `verify` → `submit`.

### 1. Hardware inventory

Probe host CPU, unified memory / VRAM, detected accelerator devices, and local toolchains:

```bash
fak bench local inventory
fak bench local inventory --json
```

The output inspects physical RAM, operating system, chip architecture, and available GPU backends without initiating network traffic or executing models.

### 2. Run local benchmark

Run an official benchmark workload with explicit benchmark and engine labels:

```bash
# Apple Silicon (Metal)
fak bench local run --benchmark qwen-decode --engine fak-native --out receipt.json

# NVIDIA RTX (CUDA)
fak bench local run --benchmark qwen-decode --engine fak-native --backend cuda --out receipt.json

# CPU reference baseline
fak bench local run --benchmark qwen-decode --engine cpu-ref --out receipt.json
```

Every execution produces a structured `fak.local-hardware-benchmark.receipt/v1` artifact recording start time, duration, scrubbed output, SHA-256 integrity hash, hardware topology, and exact git revisions.

### 3. Verify receipt integrity

Inspect and validate the local receipt before sharing:

```bash
fak bench local verify --receipt receipt.json
```

This ensures the SHA-256 output digest matches the recorded stdout/stderr, execution exited cleanly (exit status 0), and all schema fields satisfy the specification.

### 4. Privacy review and submission

```bash
fak bench local submit --receipt receipt.json
```

#### Privacy expectations:
- **No secret upload:** `fak bench local submit` does not push data to any private server or telemetry endpoint.
- **Scrubbed output:** Environment paths, credentials, authorization tokens, internal hostnames, and private IP addresses are automatically scrubbed from the command output.
- **Operator control:** The command constructs a pre-filled GitHub issue URL and either launches the default browser or prints the URL to stdout. The operator inspects the text before submission.

## Troubleshooting

- **Metal device not detected on macOS:** Verify Xcode command-line tools are installed (`xcode-select -p`) and that the `fak` binary was compiled with cgo enabled on macOS (`darwin/arm64`).
- **CUDA memory allocation errors:** Ensure your card has sufficient free VRAM. If running large models on consumer cards (e.g. 8 GiB or 12 GiB VRAM), reduce the context window or use a more compact quantization format.
- **CPU throttling:** Ensure host power profiles and thermal conditions allow sustained compute without frequency degradation during multi-minute benchmarks.
