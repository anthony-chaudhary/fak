<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — useful local agents, accelerated automatically

**Fak is building the open runtime that makes useful local agents practical on your own machine.**

Start locally, give an agent real work, and keep useful context across turns.
Our first breakthrough milestone combines native inference, speculative decoding,
and agentic caching into an experience whose qualified acceleration is automatic.
The capability floor bounds what tools the agent may execute.

**Status:** this is the product milestone we are working toward. Today,
automatic setup and cache reuse have specific model/backend limits; speculative
decoding and physical GPU Direct paths are not universally enabled or qualified.
See the [local-agent milestone](docs/local-agent-milestone.md) for the current
wiring, the meaning of automatic, and the evidence required to earn the claim.

## Try fak

Install with `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh` (or `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`).

Try the local workflow on a supported Apple Silicon configuration:

1. Start local inference (`fak up`):
   Probes unified memory with `macfit` to choose a model/context budget with
   headroom. Starts the local OpenAI-compatible endpoint on `:8080` and opens
   an interactive chat REPL. Use `fak up --mock` to inspect the workflow without
   a model or GPU; mock output is not inference performance evidence.
   ```bash
   fak up
   # -> [READY] fak up running on http://127.0.0.1:8080
   ```
   Inspect the selected model/backend and memory budget before interpreting
   results. Apple Metal selection is automatic when the device and build support it.

2. Run agent work (`fak opencode`):
   In another terminal (or backgrounding `fak up --headless`), launch OpenCode:
   ```bash
   fak opencode
   ```
   Give OpenCode a parallel multi-agent prompt:
   ```
   "Using parallel subagents, audit the packages under internal/ and report their status"
   ```
   Compatible agents can reuse shared instructions and repository context.
   Inspect actual cache reuse and task outcomes; a fresh prefix still requires
   prefill, and reuse depends on model state, backend support, and cache identity.

3. Inspect the subagent benchmark:
   ```bash
   fak bench subagent --concurrency=4
   ```
   This does not replace a real coding-task acceptance witness. Check the
   benchmark's engine, execution regime, and receipt before treating its output
   as hardware evidence; simulated output does not qualify a physical device.

> [!TIP]
> New to subagents? Follow the [Subagents Guide](docs/subagents-guide.md) to launch `fak up` and run parallel cohorts with shared-prefix cache reuse.

### Governance for external agents (`fak guard`)

Already running Claude Code or Codex? Wrap the agent you already run with one command to add a default-deny capability floor. fak forwards Codex subscription credentials with no API key required and blocks tools outside the allowed policy without breaking the task:

```bash
fak guard -- codex
```

In-kernel policy adjudication checks every tool call in under a microsecond before execution. See the [interactive showcase](docs/showcase.html) for a guided tour, or run `fak agent --offline` (# -> task completed) to inspect policy decisions with zero setup.

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
receipts. For Mac local model setup and head-to-head Apple Silicon Metal measurements, see the
[Mac local models guide](docs/fak/mac-local-models.md) and the [three-way Mac benchmark](docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md). For agent UI workflows, see the [Mac agent UI guide](docs/fak/mac-agent-ui.md).

## Why run coding agents on fak

- **Reuse the work behind each turn:** Compatible prefix/KV caching avoids
  rebuilding shared instructions and context. The product target is automatic
  reuse across turns and compatible agents, with correct invalidation and isolation.
- **Accelerate generation automatically:** Native kernels, memory sizing,
  quantization, and speculative decoding are parts of one local workflow. The
  milestone requires qualified defaults; current MTP decoding requires explicit
  selection. See the [implementation snapshot](docs/local-agent-milestone.md#current-implementation-is-narrower-than-the-milestone).
- **Real-time multi-agent visibility:** Inspect live cross-agent reuse rates, per-subagent token breakdowns, and savings sparklines directly in your terminal overlay (`fak info` / `fak guard`) to see and verify the speedup as subagents execute concurrently.
- **Keep reusable state close to compute:** Device-resident caching and direct
  GPU storage paths aim to reduce paging and copy overhead on supported hardware.
  GPU residency and physical NVMe-to-GPU DMA are different claims. The current
  [claim ledger](CLAIMS.md) and [milestone](docs/local-agent-milestone.md) explain
  the wiring and qualification limits; a default-valued flag alone proves neither.
- **Run on your own hardware:** Native backends target Apple Silicon, AMD, and
  NVIDIA with different support envelopes — and different delivery channels:
  The installer defaults to Apple Silicon Metal on darwin/arm64 and bundled Vulkan
  on linux/amd64, including AMD Strix Halo. NVIDIA users are directed to
  `ghcr.io/anthony-chaudhary/fak:cuda-latest` with `--gpus all`.
  CPU archives are secondary references selected with `--variant cpu`.
  These GPU-first archive changes require a release carrying the GPU assets;
  historical v0.54.0 lacks them. Missing GPU assets fail with an actionable message.
  Apple acceleration is fak-native Metal; MLX is a comparison runtime.
  New native-performance work prefers Qwen3.8. Choose a supported model/backend
  and measure the actual local workflow.
- **Default-deny capability floor:** Protect your workspace from unintended commands, path escapes, or tool poisoning. Every tool call is verified against a capability floor before execution. Drop-in wrappers protect existing agents like Claude Code, Codex, OpenCode, and Cursor with zero rewrites.

Native inference provides direct execution on local silicon, with external engines supported as an explicit reference; see the [native inference goal](docs/native-inference-goal.md) for details.

## Default priorities & operating modes

fak is organized around a focused four-tier default priority hierarchy:

1. **fak all in one (serving and harness + memory — the "one touch" thing):** The primary focus is the [automatic local-agent milestone](docs/local-agent-milestone.md): model serving, agent execution, capability-floor governance, and reusable context through one approachable runtime. Qualification requires a real task and independent acceptance evidence on a supported machine.
2. **fak serving only:** High-performance model inference runtime (`fak serve`), disaggregated gateway, KV-cache context acceleration, and native model execution.
3. **fak harness only:** Standalone agent governance (`fak guard`) with a default-deny capability floor and tool adjudication over external models.
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
fak guard --output-profile caveman:medium --work-profile ponytail:high -- codex \
  "Remove the duplicate cache without adding a dependency."
```

Balanced defaults are `ponytail:medium` for work discipline and `caveman:medium` for concise responses. See
[work profiles](docs/work-profiles.md), [response profiles](docs/response-profiles.md), or the
[harness guide](docs/harness-init.md) to build a named agent on the runtime.

## Going deeper

| If you want to… | Start here |
|---|---|
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [Mac](docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md) · [AMD](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) · [NVIDIA](docs/_witnesses/issue-10944-nvidia-gcp-overnight/README.md) · [all benchmarks](docs/benchmarks/README.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [subagents](docs/subagents-guide.md) · [Mac local models](docs/fak/mac-local-models.md) · [all integrations](docs/integrations/) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-09-09 vs VERSION 0.54.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-09-09 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
