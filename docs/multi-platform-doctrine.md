---
title: "First-class multi-platform doctrine: any OS, any GPU"
description: "Canonical platform-support doctrine for fak: co-equal support across macOS, Linux, and Windows, and first-class native inference across heterogeneous GPUs (current top-3 focus: Apple Metal, NVIDIA CUDA, AMD ROCm/Vulkan) behind one compute HAL."
---

# First-class multi-platform doctrine: any OS, any GPU

Audience: maintainers, contributors, silicon partners, and agents designing, implementing,
benchmarking, or reviewing `fak`.

Authority: this is the canonical platform-support doctrine. It complements the
[fak-native inference doctrine](native-inference-goal.md) (engine ownership) and the
[local-agent milestone](local-agent-milestone.md) (product milestone). Measured capability
and performance claims remain governed by [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md),
[`CLAIMS.md`](../CLAIMS.md), and the [hardware matrix](HARDWARE-MATRIX.md).

> **TL;DR:** `fak` is built to run natively on **any OS** (macOS, Linux, Windows) and **any GPU**
> (current top-3 focus: Apple Metal, NVIDIA CUDA, and AMD ROCm/Vulkan) behind one `internal/compute.Backend`
> seam—unlike local inference engines that hyper-fit to a single OS or GPU vendor.

## The six invariants, in plain language

<!-- multi-platform-doctrine:any-os -->
**Any OS:** macOS, Linux, and Windows are co-equal first-class targets for the fak runtime, CLI, gateway, and agent boundary.

<!-- multi-platform-doctrine:any-gpu -->
**Any GPU:** fak is architected for first-class native inference across heterogeneous GPUs behind one internal/compute.Backend seam rather than hyper-fitting the engine to a single vendor.

<!-- multi-platform-doctrine:top-three-focus -->
**Top-3 focus:** Current top-3 GPU focus is Apple Metal, NVIDIA CUDA, and AMD (ROCm and Vulkan), with cpu-ref as the always-available correctness oracle and the HAL open for future silicon.

<!-- multi-platform-doctrine:no-hyper-fit -->
**No single-vendor hyper-fit:** Vendor-specific kernels, memory layouts, and fast paths stay behind internal/compute.Backend and never leak single-OS or single-vendor assumptions into the model loop, scheduler, KV/context MMU, policy floor, or user contract.

<!-- multi-platform-doctrine:honest-evidence -->
**Honest evidence boundary:** Multi-platform doctrine states engineering intent and architectural rules, never an unwitnessed claim that every OS, GPU, and model cell is already qualified.

<!-- multi-platform-doctrine:no-silent-degradation -->
**Explicit capability reporting:** fak runtime-capabilities reports what actually registered and ran; an unavailable or unsupported GPU backend returns an explicit status or policy-selected fallback rather than silently substituting an external engine or claiming GPU execution on CPU.

## How `fak` differs from single-platform local inference engines

Local inference engines commonly trade portability for short-term velocity by baking one
vendor's memory model, driver stack, or OS assumptions directly into the model loop:

| Approach | Typical shape | Where it breaks |
|---|---|---|
| **Apple / Metal-only runtimes** | Tightly coupled to macOS unified memory and Metal/MLX primitives. | Leaves Linux and Windows NVIDIA/AMD workstations and servers outside the runtime. |
| **NVIDIA / CUDA-only serving stacks** | Multi-process Python + CUDA toolchains assuming Linux datacenter drivers. | Treats macOS, Windows, and AMD GPUs as unsupported or second-class wrappers. |
| **Single-box forks and wrappers** | Fast to prototype on one card; vendor types leak into scheduling and KV layout. | Adding a second GPU vendor forces a parallel model loop or brittle `#ifdef` sprawl. |
| **`fak` (first-class multi-platform)** | One Go binary and one agent-kernel boundary above `internal/compute.Backend`. | Exposes **native, first-class kernels** per top-3 GPU while keeping the model loop, KV/context MMU, scheduler, policy floor, and CLI identical across OSes and vendors. |

### Multi-platform does not mean lowest common denominator

Portability in `fak` is not a lowest-common-denominator subset that ignores hardware strengths.
Instead, `internal/compute.Backend` pairs a uniform whole-operation contract (`MatMul`,
`BatchedMatMul`, `Attention`, `RMSNorm`, `RoPE`, `Argmax`) with explicit optional capabilities
(`Caps`: `Async`, `FusedAttn`, `FusedFFN`, `GraphCompile`, `UploadDtype`, `DeviceMemory`,
`Collective`, `CapacityProbe`).

- **Apple Metal** can exploit unified memory, zero-copy residency, and Apple Silicon neural/GPU paths.
- **NVIDIA CUDA** can exploit native `sm_*` tensor cores, NVFP4/FP8 paths, and CUDA graphs.
- **AMD ROCm & Vulkan** can exploit native HIP kernels on Linux (`rocm`) and portable Wave32/SPIR-V compute across Windows and Linux (`vulkan`).

Each backend exposes its native fast paths through `Caps` and `internal/compute` without changing
the upper kernel layers or breaking parity against `cpu-ref`.

## Operating System Contract (Any OS)

| OS Target | Architectures | First-Class Surface | Legitimate OS-Specific Boundary |
|---|---|---|---|
| **macOS** | `arm64` (Apple Silicon) | Full `fak` CLI, gateway, policy floor, `cpu-ref`, and Metal backend. | cgo Metal framework bindings, Darwin wired-memory primitives, macOS release archive. |
| **Linux** | `x86_64`, `arm64` | Full `fak` CLI, gateway, policy floor, `cpu-ref`, CUDA, ROCm, and Vulkan backends. | Native CUDA/HIP/Vulkan driver bindings, systemd/container deployment integration. |
| **Windows** | `x86_64` | Full `fak` CLI, gateway, policy floor, `cpu-ref`, CUDA, and Vulkan backends. | Native Windows CUDA/Vulkan driver bindings, Windows process/path handling. |

**OS anti-patterns (not allowed):**

1. Treating Windows as "WSL-only" for core CLI, gateway, policy, or supported native GPU paths.
2. Baking POSIX shell pipelines, `/tmp`-only paths, or Linux-only `sysfs` assumptions into cross-platform runtime packages without an OS-abstracted seam.
3. Treating a single-OS convenience entry point (such as an Apple-Silicon-only quickstart helper) as permanent OS privilege rather than a named parity gap to close on Linux and Windows.

## GPU Support Tiers & Current Top-3 Focus (Any GPU)

| Tier | Backends / Scope | Role in `fak` |
|---|---|---|
| **Top-3 Active Focus** | **1. Apple Metal** (`metal`, macOS arm64)<br>**2. NVIDIA CUDA** (`cuda`, Linux & Windows)<br>**3. AMD ROCm & Vulkan** (`rocm` native HIP on Linux; `vulkan` on Windows & Linux) | Primary native GPU targets receiving active kernel optimization, memory-residency work, and hardware-witnessed qualification. |
| **Always-On Correctness Floor** | **`cpu-ref`** (`Reference` class, pure Go across all OSes/ISAs) | Bit-exact / deterministic reference oracle that every `Approx` GPU backend is tested against; also serves explicit policy-controlled `local_cpu_degraded` envelopes. |
| **Open Extension Path (Beyond Top 3)** | Intel GPUs (Vulkan or dedicated backend), NPUs, dataflow/custom silicon, WASM | Supported through the same `internal/compute.Backend` + `Caps` registration seam and verified via `cmd/modelbench -backend <name> -require-non-reference` ([supported silicon backends](supported/silicon-backends.md)). |

## Architectural Rules

1. **Keep vendor specifics below the HAL seam.** No CUDA, HIP, Metal, or Vulkan headers, pointer assumptions, or device-specific scheduling hacks may leak above `internal/compute.Backend`.
2. **Dispatch on `Caps`, not vendor names.** Upper layers (`internal/model`, `internal/agent`, `internal/gateway`, `internal/ctxmmu`) branch on advertised `Caps` and `CorrectnessClass`, never on `if backend == "cuda"` sprawl.
3. **Hold every GPU backend to `cpu-ref`.** Every `Approx` device backend proves correctness against the `cpu-ref` `Reference` floor and must pass `cmd/modelbench -backend <name> -require-non-reference` (fail closed if it silently falls back to CPU).
4. **Land fast paths asymmetrically without locking in.** A new optimization (for example, a quantized KV layout or fused attention kernel) may ship on one of the top-3 GPUs first, provided the contract stays vendor-neutral and the parity gap is recorded for the remaining top-3 backends.
5. **Scope every claim to its witness.** A benchmark or qualification on one OS + GPU + model cell (for example, RTX 5080 CUDA or M3 Pro Metal or RX 7600 Vulkan) never implies qualification on another OS, GPU generation, or model shape.

## Review Gate

Before accepting a runtime, kernel, or platform-support change, ask:

1. Does it preserve identical CLI, gateway, policy, and KV/context semantics across macOS, Linux, and Windows?
2. Are any vendor-specific kernels, layouts, or driver calls strictly contained behind `internal/compute.Backend`?
3. If a capability is added for one of the top-3 GPUs (Metal, CUDA, ROCm/Vulkan), is the upper-layer contract expressed via `Caps` so the other two can implement it cleanly?
4. Does `fak runtime-capabilities` report what actually registered and ran without silent fallback?
5. Is any performance or support claim scoped strictly to the witnessed OS, GPU, build tags, and model?

## Deterministic Docs Guard

This docs-only guard pins the six required invariant sentences and the nine inbound links
that make the doctrine discoverable. Run it from the repository root:

```powershell
$canonical = 'docs/multi-platform-doctrine.md'
$requiredPhrases = @(
  'macOS, Linux, and Windows are co-equal first-class targets for the fak runtime, CLI, gateway, and agent boundary.'
  'fak is architected for first-class native inference across heterogeneous GPUs behind one internal/compute.Backend seam rather than hyper-fitting the engine to a single vendor.'
  'Current top-3 GPU focus is Apple Metal, NVIDIA CUDA, and AMD (ROCm and Vulkan), with cpu-ref as the always-available correctness oracle and the HAL open for future silicon.'
  'Vendor-specific kernels, memory layouts, and fast paths stay behind internal/compute.Backend and never leak single-OS or single-vendor assumptions into the model loop, scheduler, KV/context MMU, policy floor, or user contract.'
  'Multi-platform doctrine states engineering intent and architectural rules, never an unwitnessed claim that every OS, GPU, and model cell is already qualified.'
  'fak runtime-capabilities reports what actually registered and ran; an unavailable or unsupported GPU backend returns an explicit status or policy-selected fallback rather than silently substituting an external engine or claiming GPU execution on CPU.'
)
$requiredLinks = [ordered]@{
  'AGENTS.md'                               = 'docs/multi-platform-doctrine.md'
  'llms.txt'                                = 'docs/multi-platform-doctrine.md'
  'docs/index.md'                           = 'multi-platform-doctrine.md'
  'docs/architecture.md'                    = 'multi-platform-doctrine.md'
  'docs/native-inference-goal.md'           = 'multi-platform-doctrine.md'
  'docs/supported/backends.md'              = '../multi-platform-doctrine.md'
  'docs/supported/silicon-backends.md'      = '../multi-platform-doctrine.md'
  'docs/explainers/hardware-portability.md' = '../multi-platform-doctrine.md'
  'docs/HARDWARE-MATRIX.md'                 = 'multi-platform-doctrine.md'
}
$failures = @()
$body = (Get-Content -Raw -LiteralPath $canonical) -replace '(?s)\x60{3}.*?\x60{3}', ''
foreach ($phrase in $requiredPhrases) {
  if (-not $body.Contains($phrase)) { $failures += "missing phrase: $phrase" }
}
foreach ($entry in $requiredLinks.GetEnumerator()) {
  if (-not (Select-String -Quiet -SimpleMatch -LiteralPath $entry.Key -Pattern $entry.Value)) {
    $failures += "missing link: $($entry.Key) -> $($entry.Value)"
  }
}
if ($failures.Count) {
  $failures | ForEach-Object { Write-Error $_ }
  exit 1
}
"PASS multi-platform-doctrine phrases=$($requiredPhrases.Count) inbound_links=$($requiredLinks.Count)"
```

Expected output:

```text
PASS multi-platform-doctrine phrases=6 inbound_links=9
```

## Related Authorities

- [Fak-native inference doctrine](native-inference-goal.md) — engine ownership, matched-envelope rule, and external-reference boundary.
- [Useful local agents, accelerated automatically](local-agent-milestone.md) — first product milestone and automatic eligibility rules.
- [External system architecture](architecture.md) — host-side kernel boundary and trust surfaces.
- [Choose a model-execution backend](supported/backends.md) — operator backend selection and `fak runtime-capabilities`.
- [Supported silicon backends](supported/silicon-backends.md) — `compute.Backend`, `Caps`, conformance, and non-reference gates.
- [Hardware portability via the compute HAL](explainers/hardware-portability.md) — the seven host-CPU assumptions neutralized by `internal/compute`.
- [Hardware matrix](HARDWARE-MATRIX.md) — machine-specific coverage and support-maturity ledger.
