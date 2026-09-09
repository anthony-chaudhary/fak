<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — the fast local runtime for coding agents

**fak is an agent runtime: one binary puts a fast, cache-accelerated boundary between your coding agent and every tool call.**

> **In short:** run coding agents locally with zero-cold-start subagent fanout and cache reuse, protected by a default-deny capability floor (blocking unauthorized actions).

## Try fak

Run the offline proof with no key, model, or GPU:

```bash
go build -o fak ./cmd/fak
./fak agent --offline  # -> task completed (booked)
```

The poisoned result and destructive operation are blocked; safe tasks complete normally.

### See raw speed and batched agents (Mac or any host)

Experience raw in-kernel inference speed and concurrent multi-agent fanout on any Mac:

1. Raw inference speed (`fak up`):
   Auto-probes unified memory with `macfit`, reserves headroom to prevent swapping, starts the local OpenAI-compatible endpoint on `:8080`, and opens an interactive chat REPL (use `fak up --mock` for instant zero-download verification):
   ```bash
   fak up
   ```
   ```
   [READY] fak up running on http://127.0.0.1:8080
     • Model: 27B (qwen3.8-27b-q4_k_m) | Context: 65536 tokens | Headroom: 33.3%
   you> Explain context caching in one line
   fak> The Context MMU shares paged KV blocks across runs for sub-millisecond reuse.
        [telemetry: 76.1 tok/s | 64 tokens | 840ms | context: 128/65536]
   ```
   Ask any question to observe raw Apple Silicon Metal generation speed with per-token telemetry.

2. Batched subagent speed (`fak opencode`):
   In another terminal (or backgrounding `fak up --headless`), launch OpenCode:
   ```bash
   fak opencode
   ```
   Give OpenCode a parallel multi-agent prompt:
   ```
   "Using parallel subagents, audit the packages under internal/ and report their status"
   ```
   OpenCode spawns four concurrent subagents (`worker`, `researcher`, `explore`, `tester`). Instead of re-reading 25k tokens of repo rules (`AGENTS.md`) and tools sequentially (100k tokens of cold-start lag), `fak` warms the shared prefix once ($O(1)$ memory cloning). All four subagents decode co-batched in parallel with zero cold start.
   Live split-pane telemetry displays real-time agent fanout and cache reuse:
   ```
   fak-turn ok prov=24.8k tok (88% of prompt) fak=0 tok cache=healthy_cache
   [fak info] 4 active · 4 subagents · 4 in-flight · 88% x-agent reuse · 4.1× speedup
   ```

3. Deterministic verification benchmark:
   Measure the subagent fanout speedup directly on your host in seconds:
   ```bash
   fak bench subagent --concurrency=4
   ```
   Runs four co-batched subagents over a 30,000-token shared prefix, witnessing 18,000+ tokens/sec aggregate throughput and >95% cache hit rate with bit-exact logit parity (`cosine = 1.000000`).

Or wrap the agent you already run with one command. In this example, fak forwards Codex subscription credentials with no API key required and blocks tools outside the allowed policy. The capability floor stops unsafe calls without breaking the task:

```bash
fak guard -- codex
```

The agent keeps working inside that boundary. See the [interactive showcase](docs/showcase.html) for the guided tour.

## Latest hardware results — 2026-09-08

The front page shows one row per supported hardware family. Latest means the newest
committed performance receipt for that platform, not the newest code change. A row can be
historical or held when no newer quality-complete measurement exists. The table reports measured
throughput with claim boundaries beside each result and links to its receipt.

| Platform | Latest witnessed result | Status & Details |
|---|---|---|
| Mac | Qwen3.8-27B Q4_K_M on Apple M3 Pro: forward-owned Metal sequence prefill was 43.8% faster, 10,284.5 vs 18,304.9 ms, and used 1 command buffer instead of 192; observed 2026-09-03. | Accepted component-path result with exact greedy continuation and zero fallbacks; it is not a full-run throughput comparison. [Qwen result index](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) |
| AMD | Qwen3.6-27B on RX 7600: the pure-fak TG1 microbench measured 1.24 decode tok/s versus 0.99 for the local llama.cpp Vulkan baseline; observed 2026-06-19. | Narrow, older-model microbench. No accepted current Qwen3.8 AMD result exists. [AMD receipt](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) |
| NVIDIA | Qwen2.5-3B Q8_0 on a physical Hopper H100: fak reached 111.9 decode tok/s, 17.4% above its f32 path; observed 2026-09-05. | Native CUDA result; llama.cpp Q8_0 was 3.24× as fast at 362.7 tok/s in the same run. [H100 receipt](docs/benchmarks/GCP-H100-RESULTS.md) |

Read the status column before comparing rates: results compare matched envelopes against explicit baseline runtimes on identical hardware.

Use the [benchmark index](docs/benchmarks/README.md) for hardware history and model-specific
results. Use [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) for claim boundaries and canonical
receipts. For newcomer Mac guidance, running local models (Qwen3.8), and head-to-head Apple Silicon Metal measurements, see the
[Mac local models guide](docs/fak/mac-local-models.md), [Mac agent UI guide](docs/fak/mac-agent-ui.md), and the [three-way Mac benchmark](docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md).

## Open-source memory overflow landscape

Most LLM serving engines treat memory overflow as a slow host-memory fallback with multiple CPU bounce copies. fak implements hardware-native, zero-copy peer-to-peer DMA directly between NVMe storage and GPU VRAM:

| Framework | Storage / Offload DMA Path | Host DRAM Copies | Predictive Prefetching | Hybrid Attention + GDN Linear State | Target Workload |
|---|---|:---:|:---:|:---:|---|
| **fak (native)** | GPU Direct NVMe P2PDMA (BaM architecture) | 0 (strictly zero) | Yes (asynchronous pipeline) | Yes (bit-exact full + linear) | Interactive, real-time agent coding loops |
| vLLM | Host DRAM block swapping (`swap_blocks`) | 2–3 copies | No (reactive) | No (Transformer KV only) | High-throughput data-center batching |
| DeepSpeed ZeRO | Async CPU `aio` offload via pinned DRAM buffers | 2 copies | Coarse (layer-level weights) | No (static forward layers only) | Multi-node distributed training / inference |
| FlexGen | 3-tier offload (GPU ↔ CPU ↔ Disk) | 2–3 copies | Zigzag batch schedule | No (attention matrices only) | Extreme high-latency batch throughput |
| TensorRT-LLM | NVIDIA GPUDirect Storage (`libcufile.so`) | 0 (NVIDIA only) | Yes (NVIDIA GDS) | Partial (Transformer KV) | NVIDIA enterprise data centers only |
| llama.cpp | OS `mmap` demand paging & CPU fallback | 2 copies (OS cache) | No (kernel readahead) | Basic (CPU fallback layers) | Local desktop CPU/GPU inference |

## Why run coding agents on fak

- **Zero-cold-start subagent fanout:** Standard multi-agent swarms pay a heavy cold-start penalty on every spawned worker, re-ingesting 20k–30k tokens of prompts, tools, and repo context. fak warms this shared prefix once. Subagents inherit resident KV caches in milliseconds ($O(1)$ memory cloning), dropping Time-To-First-Token (TTFT) and achieving **4.1× vs tuned** baselines with 86.7% cache hit rates. In-kernel tool caching (vDSO) serves idempotent reads in sub-microsecond time.
- **Real-time multi-agent visibility:** Inspect live cross-agent reuse rates, per-subagent token breakdowns, and savings sparklines directly in your terminal overlay (`fak info` / `fak guard`) to see and verify the speedup as subagents execute concurrently.
- **Zero-copy GPU Direct storage overflow:** Run models far exceeding physical GPU VRAM without host memory thrashing. Built on a BaM accelerator storage architecture, fak maps NVMe queues directly in GPU VRAM. It streams paged KV caches and hybrid linear states over peer-to-peer PCIe DMA without DRAM bounce copies (`StagingCopyCount == 0`). See the [GPU Direct overflow specification](docs/benchmarks/QWEN38-AMD-GPUDIRECT-RESULTS.md).
- **Local execution on your hardware:** Run models directly with native inference across Apple Silicon, AMD, and NVIDIA. New work prioritizes Qwen3.8 with resident quantization and prefix reuse. Cut token bills and keep your code private on your own machine.
- **Default-deny capability floor:** Protect your workspace from unintended commands, path escapes, or tool poisoning. Every tool call is verified against a capability floor before execution. Drop-in wrappers protect existing agents like Claude Code, Codex, OpenCode, and Cursor with zero rewrites.

Native inference provides direct execution on local silicon, with external engines supported as an explicit reference; see the [native inference goal](docs/native-inference-goal.md) for details.

## Default priorities & operating modes

fak is organized around a focused four-tier default priority hierarchy:

1. **fak all in one (serving and harness + memory — the "one touch" thing):** The primary focus: a single-binary turnkey runtime (`fak up`) bundling model serving, agent harness governance, and persistent memory. Verified on Terminal-Bench 4: 100.0% (5/5) solve rate vs OpenCode + llama.cpp 60.0% (3/5), cutting prompt tokens by 83.5% via in-kernel vDSO context caching (`fak bench tb4`).
2. **fak serving only:** High-performance model inference runtime (`fak serve`), disaggregated gateway, KV-cache context acceleration, and native model execution.
3. **fak harness only:** Standalone agent governance substrate (`fak guard`), default-deny capability floor, and tool adjudication over external models.
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
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [Mac local models](docs/fak/mac-local-models.md) · [all integrations](docs/integrations/) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-09-08 vs VERSION 0.54.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-09-08 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
