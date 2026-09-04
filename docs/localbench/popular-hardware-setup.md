---
title: "Popular local hardware benchmark setup guide"
description: "Step-by-step setup guide for configuring, running, and submitting local hardware benchmark receipts with fak bench local across Apple Silicon, NVIDIA RTX, AMD Radeon, and CPU-only environments."
---

# Popular local hardware benchmark setup guide

This guide is the task-oriented on-ramp for self-hosting operators and enthusiasts measuring local agent and model performance on hardware they own. It walks from initial platform setup through running benchmarks, verifying cryptographic receipts, and submitting to the moderated community scoreboard—without ever uploading data silently or falling back to external runtimes.

All benchmarks in `fak` follow the **native-inference invariant** ([`docs/native-inference-goal.md`](../native-inference-goal.md)): model execution is `fak-native` all the way, with transparent ownership of kernels, memory, and scheduling. There is no silent fallback to llama.cpp, Ollama, or external runtimes.

---

## 1. Supported Platform Matrix

| Platform | Recommended Engine / Backend | Key Toolchain Probe | Detection Command | Maturity Tier |
|---|---|---|---|---|
| **Apple Silicon (M1–M4)** | `fak-native` (Metal) | `xcrun metal --version` | `system_profiler SPDisplaysDataType` | **Tier 1 Supported / Proven** |
| **NVIDIA GeForce RTX (30/40/A-series)** | `fak-native` (CUDA) | `nvcc --version` | `nvidia-smi` | **Tier 1 Supported / Proven** |
| **AMD Radeon (Linux ROCm)** | `fak-native` (ROCm) | `rocminfo` | `rocminfo` | **Supported / Proven** |
| **AMD Radeon (Windows/Linux Vulkan)** | `fak-native` (Vulkan) | `vulkaninfo --summary` | `vulkaninfo` | **Experimental** |
| **CPU-Only (x86_64 / ARM64)** | `fak-cpu-q8` (Explicit CPU) | Standard C/Go toolchain | `/proc/cpuinfo` or `sysctl` | **Supported Baseline** |

---

## 2. Platform Prerequisites & Setup

### 2.1 Apple Silicon (Metal)

Apple Silicon Macs (M1, M2, M3, M4 across Base, Pro, Max, and Ultra configurations) utilize unified memory where the CPU and GPU share the same high-bandwidth LPDDR5/LPDDR5X memory pool (100 to 800 GB/s).

#### Prerequisites
1. **Operating System:** macOS 14.0 (Sonoma) or macOS 15.0+ (Sequoia).
2. **Xcode Command Line Tools:** Install with:
   ```bash
   xcode-select --install
   ```
3. **Verify Toolchain:**
   ```bash
   xcrun metal --version
   ```
   Ensure it outputs the Apple Metal compiler version (e.g. `Apple metal version 32023.98`).

#### Hardware Inspection & Knobs
- Check detected display and GPU model:
  ```bash
  system_profiler SPDisplaysDataType -detailLevel mini
  ```
- Check unified physical memory:
  ```bash
  sysctl -n hw.memsize
  ```
- **Large Model Headroom Knob:** When running quantized models (such as Qwen3.8-27B Q4_K_M) near the machine's memory boundary, opt into weight streaming:
  ```bash
  export FAK_METAL_STREAM_Q4K=1
  ```

---

### 2.2 NVIDIA GeForce RTX & RTX A-Series (Windows & Linux)

NVIDIA GPUs (RTX 30-series Ampere, RTX 40-series Ada Lovelace, RTX A-series, and Blackwell) run via `fak-native`'s custom CUDA kernels and CUDA graphs.

#### Linux Prerequisites
1. **NVIDIA Driver:** Version 535 or newer (recommended: 550+):
   ```bash
   nvidia-smi
   ```
2. **CUDA Toolkit:** Install CUDA 12.2 or newer. Verify `nvcc` is on your `$PATH`:
   ```bash
   nvcc --version
   ```
3. **Persistence Mode:** On dedicated Linux workstations or servers, enable driver persistence to eliminate initial CUDA context initialization delays:
   ```bash
   sudo nvidia-smi -pm 1
   ```
4. **Permissions:** Ensure your user is in the `video` group and has write access to `/dev/nvidia*`.

#### Windows Prerequisites
1. **NVIDIA Display Driver:** Install the latest Game Ready or Studio Driver (550+).
2. **Path Verification:** Confirm `nvidia-smi.exe` is reachable in PowerShell:
   ```powershell
   Get-Command nvidia-smi
   nvidia-smi
   ```
   If not in PATH, add `C:\Program Files\NVIDIA Corporation\NVSMI` or `C:\Windows\System32` to your user PATH.
3. **Native vs. WSL2:**
   - **Native Windows:** Recommended for developer ergonomics and native PowerShell integration.
   - **WSL2:** Supported under Ubuntu 22.04/24.04 WSL2 distros with CUDA on WSL enabled. Note that WSL2 virtualization adds ~2–4% kernel launch dispatch overhead.
4. **Antivirus / Microsoft Defender:** As noted in [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](../notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md), Microsoft Defender real-time scanning can quarantine newly compiled Go test binaries. Use `fak-dev windows-setup --apply` or run long test suites through WSL.

---

### 2.3 AMD Radeon (Linux ROCm & Windows/Linux Vulkan)

AMD Radeon graphics cards provide two execution routes depending on operating system and architecture.

#### Route A: Linux ROCm (Recommended for RDNA2 & RDNA3)
- **Supported Hardware:** AMD Radeon RX 6800/6900 series (gfx1030), RX 7900 XT/XTX (gfx1100), Radeon PRO W6800/W7900.
- **Prerequisites:**
  1. Install ROCm 6.0+ via the official AMD package repository.
  2. Add your user account to the `render` and `video` groups:
     ```bash
     sudo usermod -aG render,video $USER
     ```
  3. Verify toolchain and target architecture:
     ```bash
     rocminfo
     ```
     Ensure `Marketing Name: AMD Radeon...` appears in the output.
- **TTM Memory Limits:** By default, the Linux kernel TTM memory manager may limit single-allocation buffer size to 50% of system RAM. For large model residency, adjust via module options or `fak amdgpu governor` configuration.

#### Route B: Windows & Linux Vulkan (Cross-Platform)
- **Supported Hardware:** All modern AMD Radeon GPUs supporting Vulkan 1.3+.
- **Prerequisites:**
  1. Install latest AMD Software: Adrenalin Edition driver.
  2. Install the LunarG Vulkan SDK or runtime.
  3. Verify with:
     ```bash
     vulkaninfo --summary
     ```
- **DriverStore Path Scrubbing:** `fak bench local` automatically scrubs sensitive Windows driver store paths (e.g. `C:\Windows\System32\DriverStore\...`) while retaining the canonical Vulkan instance version.

---

### 2.4 CPU-Only Explicit Selection

CPU execution is fully supported across x86_64 (AVX2, AVX-512) and ARM64 (NEON).

- **Explicit Selection Rule:** CPU execution must be explicitly chosen using an engine label such as `--engine fak-cpu-q8`.
- **No Automatic Fallback:** If a GPU or accelerator fails to initialize, `fak` will **never** silently fall back to CPU inference. The command fails closed and logs the exact error to ensure benchmark receipts remain honest evidence.

---

## 3. The `fak bench local` Workflow

The receipt lifecycle consists of four reproducible steps:

```console
fak bench local inventory
fak bench local run --benchmark <name> --engine <label> --out receipt.json -- <command> [args...]
fak bench local verify receipt.json
fak bench local attest --receipt receipt.json --key <key-file> [--out attested.json]
fak bench local submit receipt.json
```

### Step 1: Discover Local Capabilities (`inventory`)
Run `inventory` to inspect how `fak` detects your local hardware, detected accelerators, toolchains, and available benchmark catalog recipes:

```console
fak bench local inventory
```

Example JSON output:
```json
{
  "hardware": {
    "os": "darwin",
    "arch": "arm64",
    "cpu": "Apple M3 Max",
    "memory_bytes": 68719476736,
    "accelerators": [
      {
        "vendor": "Apple",
        "kind": "gpu",
        "model": "Apple M3 Max",
        "backend": "Metal"
      }
    ],
    "toolchains": {
      "metal": "Apple metal version 32023.98"
    }
  },
  "benchmarks": [
    {
      "name": "modelbench",
      "kind": "cmd",
      "need": "weights",
      "summary": "Full-model serving benchmark: prefill tok/s, decode tok/s...",
      "run": "go run ./cmd/modelbench -dir ..."
    }
  ]
}
```

### Step 2: Execute and Record a Sealed Run (`run`)
Execute your benchmark command behind `--`. Provide the explicit benchmark name and engine label:

```console
fak bench local run \
  --benchmark modelbench \
  --engine fak-native \
  --out local-receipt.json \
  -- fak modelbench --model qwen38-27b --quant Q4_K_M
```

What `fak bench local run` does:
- Starts child process execution, streaming stdout and stderr to the console.
- Records exact duration, start/finish UTC timestamps, and exit status.
- Computes SHA-256 over raw un-scrubbed output bytes.
- Scrubs credentials, home paths, local drive paths, and machine names from retained output.
- Captures normalized OS, CPU, memory, accelerator, toolchain, and Git provenance.
- Computes a cryptographic SHA-256 integrity seal over canonical receipt bytes.

### Step 3: Verify Receipt Integrity (`verify`)
Verify that a receipt file has not been modified or corrupted since creation:

```console
fak bench local verify local-receipt.json
```

Output:
```text
VERIFIED fak.local-hardware-benchmark.receipt/v1 6ca136ddeff03b5762cf... benchmark=modelbench engine=fak-native exit=0
```

Any modification to exit status, command, duration, hardware, or output causes verification to fail closed:
```text
error: receipt integrity mismatch (tampered or corrupted)
```

### Step 4: Cryptographic Attestation (`attest`)
Under issue #10446, operators can bind their receipt to model artifact digest, quantization parameters, quality verification results, and execution backend using an Ed25519 keypair:

1. **Generate a signing key (stored safely outside source trees):**
   ```console
   fak bench local attest --generate-key --key ~/.fak/operator.key
   ```
2. **Sign the receipt with identity bindings:**
   ```console
   fak bench local attest \
     --receipt local-receipt.json \
     --key ~/.fak/operator.key \
     --key-id operator-main \
     --model "unsloth/Qwen3.8-27B-GGUF" \
     --model-digest "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169" \
     --quant "Q4_K_M" \
     --bits 4.5 \
     --workload "prompt-heavy-e2e" \
     --backend "metal" \
     --runtime "metal/qwen35-hybrid-session-v1" \
     --fallback "none" \
     --eval-kind "exact_match" \
     --eval-threshold "pass" \
     --quality-pass \
     --out attested-receipt.json
   ```
3. **Verify the attested envelope:**
   ```console
   fak bench local verify attested-receipt.json
   ```
   Output:
   ```text
   VERIFIED fak.local-hardware-benchmark.receipt/v1 <SHA256> benchmark=modelbench engine=fak-native exit=0 attestation=self_signed
   ```

### Step 5: Privacy Review & Submission (`submit`)
Run `submit` to prepare a GitHub submission packet:

```console
fak bench local submit local-receipt.json
```

Output:
```markdown
## Local hardware benchmark receipt

- Schema: `fak.local-hardware-benchmark.receipt/v1`
- Receipt SHA-256: `6ca136ddeff03b5762cf...`
- Benchmark: `modelbench`
- Engine: `fak-native`
- Command: `["fak", "modelbench", "--model", "qwen38-27b"]`
- Exit status: `0`
- Hardware: `darwin/arm64`; CPU `Apple M3 Max`; memory `68719476736` bytes
- Accelerators: Apple Apple M3 Max (Metal)
- fak: `0.45.0` (`923309221e5b`)
- repo revision: `60b8081aea20`

---
Open this URL to review and explicitly submit (this tool never uploads):
https://github.com/anthony-chaudhary/fak/issues/new?title=...
```

**Privacy Guarantee:** `fak bench local submit` **never** uploads data or opens your browser automatically. It outputs the Markdown report and a prefilled URL for you to review before submitting.

---

## 4. Privacy Expectations & Scrubbing Guarantees

Every local benchmark receipt is scrubbed before serialization:

| Data Category | Handling in Receipt | Scrubbing Rule |
|---|---|---|
| **API Keys & Tokens** | **REDACTED** | Flags matching `--token`, `--api-key`, `--secret`, `--password` have values replaced with `[REDACTED]`. |
| **Bearer Headers** | **REDACTED** | Patterns matching `Bearer <token>` are replaced with `Bearer-[REDACTED]`. |
| **User Home Directories** | **REDACTED** | Paths matching `C:\Users\<username>` or `/home/<username>` are replaced with `[HOME]`. |
| **Local File Paths** | **REDACTED** | Windows absolute paths and UNC shares are replaced with `[LOCAL_PATH]`. |
| **Hostnames / Machine IDs** | **REDACTED** | Matches to `computername`, `hostname`, `machine-id` are replaced with `[REDACTED]`. |
| **Raw Terminal Output** | **Bounded & Scrubbed** | Retains up to 32 KB of scrubbed terminal text; raw un-scrubbed output is represented only by its cryptographic SHA-256 digest. |

---

## 5. Community Scoreboard Integration (#10445)

Submitted receipts are processed through `fak`'s moderated community hardware scoreboard (`internal/localbench/scoreboard.go`):

1. **Intake Validation:** Receipts are checked for schema compliance, trailing bytes, payload size limits (1 MiB max), and SHA-256 integrity.
2. **Moderation Queue:**
   - `pending`: Newly ingested receipts awaiting automated or maintainer review.
   - `accepted`: Verified against matched hardware envelopes and published to the public scoreboard.
   - `rejected`: Incompatible, non-reproducible, or malformed submissions with an auditable reason.
   - `quarantined` / `withdrawn`: Isolated or author-retracted runs.
3. **Deterministic Deduplication:** Receipts submitted multiple times are recognized idempotently via a deterministic hash of receipt integrity and benchmark identity (`ComputeDedupKey`), preserving historical review status without duplication.
4. **Privacy-Safe Public Projection:** The public scoreboard API projects only non-sensitive summary fields (`PublicProjection`), completely omitting raw command output, home directories, and user paths while escaping all text strings against HTML injection.

---

## 6. Troubleshooting & Diagnostics

### Issue: "accelerators: none reported"
- **NVIDIA:** Verify `nvidia-smi` is executable from the current shell. On Windows, check that the driver directory is in your System PATH.
- **AMD ROCm:** Verify `rocminfo` runs without error and your user is in the `video` and `render` groups.
- **Apple Silicon:** Verify Xcode Command Line Tools are active (`xcode-select -p`).

### Issue: "child command exited with status N; receipt was still written"
A non-zero exit code (e.g. status 1 or 137 OOM) still generates a valid, sealed receipt. `fak` preserves child process failures so negative evidence and crash diagnostics can be studied without hiding failures.

### Issue: "receipt integrity mismatch (tampered or corrupted)"
The receipt's JSON payload was modified after creation. Re-run the benchmark to generate a fresh receipt, or ensure your text editor does not alter line endings or JSON whitespace.

### Issue: Windows Defender Quarantines Test Binaries
On Windows, Microsoft Defender real-time protection may flag transient test executables. Do not disable antivirus; run full test suites under WSL2 as documented in [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](../notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md), or run `fak-dev windows-setup --apply` for vetted local exclusions.

---

## 7. Related Documents & Authority

- [`docs/native-inference-goal.md`](../native-inference-goal.md) — The native inference invariant and matched-envelope doctrine.
- [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) — Single source of truth for benchmark claims and rules.
- [`examples/local-hardware-benchmark/README.md`](../../examples/local-hardware-benchmark/README.md) — Walkthrough of the local receipt tool.
- [`docs/_witnesses/issue-10458-home-llm-consumer-hardware/README.md`](../_witnesses/issue-10458-home-llm-consumer-hardware/README.md) — Witnessed consumer hardware residency and composed-cache benchmark.
- Issues: #10421 (v1 receipt foundation), #10444 (CLI promotion), #10445 (moderated scoreboard), #10446 (attestation envelope), #10458 (consumer residency benchmark), #10470 (hardware setup guide).
