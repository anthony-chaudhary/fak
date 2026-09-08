<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — a local runtime and safety boundary for coding agents

> **In short:** fak is one Go binary that sits between a coding agent, its model, and its tools. It can front a hosted or local model, reuse shared context, and judge tool calls before they run.

In a 50-turn × 5-agent Qwen2.5-1.5B Q8 session, fak's shared-context arm finished in about one quarter of the tuned per-agent warm-KV arm's time: **4.1× vs tuned**. Both arms ran live on the same kernel, so this measures reuse in this workload rather than universal serving speed. [Inspect the receipt](experiments/session/headline-qwen-50x5.json).

## Try fak

Run the deterministic offline proof. It needs no API key, model download, or GPU:

```bash
go build -o fak ./cmd/fak
./fak agent --offline  # -> poisoned result blocked; destructive op prevented; task completed
```

The report shows the poisoned result removed, the destructive operation prevented, and the flight-booking task still completed.

To wrap an agent you already use:

```bash
fak guard -- codex
```

The default posture is `default_open`. Unlisted benign calls are admitted after the guard checks them. Registered dangerous operations, explicit policy denies, protected self-modification, and frozen safety invariants still fail closed or require confirmation.

Use a strict allow-list when that is your policy:

```bash
fak guard --posture fail_closed -- codex
```

In `fail_closed`, a tool must be affirmatively allowed. Start with the [interactive showcase](docs/showcase.html), or inspect and edit the built-in floor with `fak guard --dump-policy`.

## Run a local model

Load a local GGUF checkpoint into fak's own engine and keep the same tool boundary:

```bash
fak guard --gguf /path/to/model.gguf -- codex
```

This path needs no API key or second model server. It is a correctness and cache-reuse reference, not a production throughput claim. If Ollama, LM Studio, or llama.cpp is already running, use `fak guard --local -- codex`. For serving-grade throughput, keep the tuned engine and front its OpenAI-compatible endpoint with `fak serve --base-url`.

## Latest hardware results — 2026-09-08

Latest means the newest committed performance receipt for that platform. These rows have different models and workloads; compare only within the stated envelope.

| Platform | Latest witnessed result | Boundary |
|---|---|---|
| Mac | Qwen3.8-27B Q4_K_M on Apple M3 Pro: forward-owned Metal sequence prefill was 43.8% faster, 10,284.5 vs 18,304.9 ms, and used 1 command buffer instead of 192; observed 2026-09-03. | Accepted component-path result with exact greedy continuation and zero fallbacks; it is not a full-run throughput comparison. [Qwen result index](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) |
| AMD | Qwen3.6-27B on RX 7600: the pure-fak TG1 microbench measured 1.24 decode tok/s versus 0.99 for the local llama.cpp Vulkan baseline; observed 2026-06-19. | Narrow, older-model microbench. No accepted current Qwen3.8 AMD result exists. [AMD receipt](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) |
| NVIDIA | Qwen2.5-3B Q8_0 on a physical Hopper H100: fak reached 111.9 decode tok/s, 17.4% above its f32 path; observed 2026-09-05. | Native CUDA result; llama.cpp Q8_0 was 3.24× as fast at 362.7 tok/s in the same run. [H100 receipt](docs/benchmarks/GCP-H100-RESULTS.md) |

Use the [benchmark index](docs/benchmarks/README.md) for history and [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) for claim boundaries. The [Qwen performance index](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) separates current, diagnostic, historical, and awaiting-remeasurement results.

## What fak provides

- A model gateway for hosted APIs, existing OpenAI-compatible servers, or an explicitly loaded local GGUF.
- A tool-call boundary with policies and structured denies. It redacts secrets, holds irreversible actions for confirmation, and writes an audit journal.
- Shared-prefix and result reuse for agent fleets. The 4.1× receipt above is a read-heavy, single-host workload; it does not imply faster raw token generation on every engine.
- Native CPU, Apple Metal, AMD Vulkan, and NVIDIA CUDA paths with platform-specific evidence and limits.

See the [native inference goal](docs/native-inference-goal.md) for the engine boundary and [Status](STATUS.md) for what is shipped, limited, or planned.

## Install and configure

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# Any host with Go 1.26+
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# Inspect the shipped profiles
fak agent profiles
```

Optional work and response profiles tune how an agent approaches a task and how much it says:

```bash
fak manage --output-profile caveman:medium --work-profile ponytail:high -- codex \
  "Remove the duplicate cache without adding a dependency."
```

The defaults are `ponytail:medium` and `caveman:medium`. Read the [work profiles](docs/work-profiles.md), [response profiles](docs/response-profiles.md), or [harness guide](docs/harness-init.md) for the full contract.

## Going deeper

| If you want to… | Start here |
|---|---|
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [Qwen results](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) · [all benchmarks](docs/benchmarks/README.md) · [benchmark authority](BENCHMARK-AUTHORITY.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [all integrations](docs/integrations/) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-09-08 vs VERSION 0.53.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-09-08 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
