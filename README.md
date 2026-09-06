<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — the fast local runtime for coding agents

**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**

> **In short:** run coding agents locally with workflow batching and cache reuse, protected by a default-deny capability floor (blocking unauthorized actions).

## Try fak

Run the offline proof with no key, model, or GPU:

```bash
go build -o fak ./cmd/fak
./fak agent --offline  # -> task completed (booked)
```

The poisoned result and destructive operation are blocked; safe tasks complete normally.

Or wrap the agent you already run with one command. In this example, fak forwards Codex subscription credentials with no API key required and blocks tools outside the allowed policy. The capability floor stops unsafe calls without breaking the task:

```bash
fak guard -- codex
```

The agent keeps working inside that boundary. See the [interactive showcase](docs/showcase.html) for the guided tour.

## Latest hardware results — 2026-09-05

The front page shows one row per supported hardware family. Latest means the newest
committed performance receipt for that platform, not the newest code change. A row can be
historical or held when no newer quality-complete measurement exists. The table reports measured
throughput, for example 7.61 decode tok/s on Mac or 111.9 tok/s on Hopper H100, with claim boundaries beside each result
and links to its receipt.

| Platform | Latest witnessed result | Status | Details |
|---|---|---|---|
| Mac | Qwen3.8-27B Q4_K_M on an Apple M3 Pro: 7.61 decode tok/s (+3.1% vs llama.cpp 7.38, MLX 8.07) and 12.6 ms prefix TTFT, observed 2026-09-03. | Verified matched-envelope single-stream decode leads llama.cpp Metal; RadixAttention prefix caching eliminates repeat prefill. | [Mac result](docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md) |
| AMD | Qwen3.6-27B on an RX 7600: the measured pure-fak microbench reached 1.15–1.24 decode tok/s versus 0.99 for the local llama.cpp Vulkan baseline, observed 2026-06-19. | Witnessed in that narrow microbench; not a broad quality or full-model parity claim. Qwen3.8 awaits a comparable AMD receipt. | [AMD result](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) |
| NVIDIA | Hopper H100 Q8_0 decode reached 111.9 tok/s (+17.4% vs f32); live A100 Qwen3.8-27B prefix reuse achieved 4.84× TTFT speedup, observed 2026-09-05. | Witnessed on physical GCP H100 (a3-highgpu-1g) & A100; matched Q8 device GEMV and 50-agent concurrency grid (91/91 ok). | [NVIDIA result](docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md) |

Read the status column before comparing rates: results compare matched envelopes against explicit baseline runtimes on identical hardware.

Use the [benchmark index](docs/benchmarks/README.md) for hardware history and model-specific
results. Use [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) for claim boundaries and canonical
receipts. For newcomer Mac guidance and head-to-head Apple Silicon Metal measurements, see the
[Mac agent UI guide](docs/fak/mac-agent-ui.md) and the [three-way Mac benchmark](docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md).

## Open-source memory overflow landscape

Most LLM serving engines treat memory overflow as a slow host-memory fallback with multiple CPU bounce copies. fak implements hardware-native, zero-copy peer-to-peer DMA directly between NVMe storage and GPU VRAM:

| Framework | Storage / Offload DMA Path | Host DRAM Copies | Predictive Prefetching | Hybrid Attention + GDN Linear State | Target Workload |
|---|---|:---:|:---:|:---:|---|
| **fak (native)** | **GPU Direct NVMe P2PDMA (BaM architecture)** | **0 (strictly zero)** | **Yes (asynchronous pipeline)** | **Yes (bit-exact full + linear)** | **Interactive, real-time agent coding loops** |
| **vLLM** | Host DRAM block swapping (`swap_blocks`) | 2–3 copies | No (reactive) | No (Transformer KV only) | High-throughput data-center batching |
| **DeepSpeed ZeRO** | Async CPU `aio` offload via pinned DRAM buffers | 2 copies | Coarse (layer-level weights) | No (static forward layers only) | Multi-node distributed training / inference |
| **FlexGen** | 3-tier offload (GPU ↔ CPU ↔ Disk) | 2–3 copies | Zigzag batch schedule | No (attention matrices only) | Extreme high-latency batch throughput |
| **TensorRT-LLM** | NVIDIA GPUDirect Storage (`libcufile.so`) | 0 (NVIDIA only) | Yes (NVIDIA GDS) | Partial (Transformer KV) | NVIDIA enterprise data centers only |
| **llama.cpp** | OS `mmap` demand paging & CPU fallback | 2 copies (OS cache) | No (kernel readahead) | Basic (CPU fallback layers) | Local desktop CPU/GPU inference |

## Why run coding agents on fak

- **Workflow batching and cache reuse:** Multi-agent coding loops reuse prompt context across turns, achieving **4.1× vs tuned** baselines with 86.7% cache hit rates. Instead of re-reading codebases on every turn, fak keeps shared prefixes hot and trims stale context.
- **Zero-copy GPU Direct storage overflow:** Run models far exceeding physical GPU VRAM without CPU memory thrashing. Built on a BaM-style accelerator storage architecture, fak maps NVMe submission queues directly in GPU VRAM and streams paged KV caches and hybrid linear attention states over peer-to-peer PCIe DMA without host DRAM bounce buffering (`StagingCopyCount == 0`). See the [GPU Direct overflow specification](docs/benchmarks/QWEN38-AMD-GPUDIRECT-RESULTS.md).
- **Local execution on your hardware:** Run models directly with native inference across Apple Silicon, AMD, and NVIDIA. Cut per-token API bills and keep your code private on your own machine.
- **Default-deny capability floor:** Protect your workspace from unintended terminal commands or file edits. Every tool call is checked against a default-deny (block everything unless allowed) policy before it runs. Drop-in support wraps existing agents like Claude Code, Codex, Aider, and Cursor with zero rewrites.

Native inference provides direct execution on local silicon, with external engines supported as an explicit reference; see the [native inference goal](docs/native-inference-goal.md) for details.

## Default priorities & operating modes

fak is organized around a focused four-tier default priority hierarchy:

1. **fak all in one (serving and harness + memory — the "one touch" thing):** The primary focus — a single-binary "one touch" deployment (`fak up`) bundling model serving, agent harness governance, and persistent memory. Verified on Terminal-Bench 4: 100.0% (5/5) solve rate vs OpenCode + llama.cpp 60.0% (3/5), reducing prompt tokens by 83.5% through in-kernel vDSO context caching (`fak bench tb4`).
2. **fak serving only:** High-performance model inference runtime (`fak serve`), disaggregated gateway, KV-cache/context MMU acceleration, and native model execution.
3. **fak harness only:** Standalone agent harness and governance substrate (`fak guard`), default-deny capability floor, and tool adjudication over external models.
4. **other things:** Standalone utilities, peripheral tools, benchmarks, and off-spine extensions.

## Install and configure

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# Any host with Go 1.26+
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# Inspect the shipped profiles
fak agent profiles
```

Tune agent execution with built-in work and output profiles that cut token waste and resist unnecessary dependencies:

```bash
fak manage --output-profile caveman:medium --work-profile ponytail:high -- codex \
  "Remove the duplicate cache without adding a dependency."
```

Balanced defaults are `ponytail:medium` for work discipline and `caveman:medium` for concise responses. See
[work profiles](docs/work-profiles.md), [response profiles](docs/response-profiles.md), or the
[harness guide](docs/harness-init.md) to build a named agent around the same boundary.

## Going deeper

| If you want to… | Start here |
|---|---|
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [Mac](docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md) · [AMD](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) · [NVIDIA](docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md) · [all benchmarks](docs/benchmarks/README.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [all integrations](docs/integrations/) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-09-05 vs VERSION 0.51.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-09-05 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
