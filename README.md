<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — configure your agents for the task at hand

fak is an agent kernel: one Go binary that manages context, models, tools, policy, and receipts.

> In short: see the latest Mac, AMD, and NVIDIA evidence first; open the indexes for history.

[Try fak](#try-fak), or see the [interactive showcase](docs/showcase.html) for the guided tour.

## Latest hardware results — 2026-08-28

The front page shows one row per supported hardware family. Latest means the newest
committed performance receipt for that platform, not the newest code change. A row can be
historical or held when no newer quality-complete measurement exists.

| Platform | Latest witnessed result | Status | Details |
|---|---|---|---|
| Mac | Qwen3.8-27B Q4_K_M on an Apple M3 Pro: 2.3–2.9 decode tok/s and 3.2–8.4 full-prefill tok/s, observed 2026-08-20. | Historical; its review window ended without a comparable replacement, so this is not a current parity claim. | [Mac result](docs/benchmarks/QWEN38-27B-LATEST.md) |
| AMD | Qwen3.6-27B on an RX 7600: the measured pure-fak microbench reached 1.15–1.24 decode tok/s versus 0.99 for the local llama.cpp Vulkan baseline, observed 2026-06-19. | Witnessed in that narrow microbench; not a broad quality or full-model parity claim. Qwen3.8 awaits a comparable AMD receipt. | [AMD result](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) |
| NVIDIA | Native Qwen3.8-27B CUDA: the cold arm produced 5/5 exact outputs at 11.8–12.1 decode tok/s; confirmed cache hits produced 0/5 exact at about 0.2 tok/s, captured 2026-08-25. | Hold: failed cache-hit quality excludes this from parity or improvement claims. | [NVIDIA result](docs/_witnesses/issue-8819-qwen38-cache-attribution/README.md) |

Use the [benchmark index](docs/benchmarks/README.md) for hardware history and model-specific
results. Use [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) for claim boundaries and canonical
receipts.

## Try fak

Wrap the Codex session you already use. fak forwards the existing subscription credential and
applies a capability allow-list, so tools outside the configured floor are blocked by default:

```bash
fak guard -- codex
```

Or run the offline proof with no key, model, or GPU:

```bash
go build -o fak ./cmd/fak
./fak agent --offline
```

Expected result: `task completed (booked)` while the poisoned result and destructive operation
are blocked.

## What fak manages

- **Context:** reuse shared setup, keep provider cache continuity, and shed old turns.
- **Models:** route each task to the configured provider or fak-native engine without a silent fallback.
- **Tools:** admit each call through a default-deny capability floor and record the verdict.

The native engine is the product and performance path. llama.cpp remains an explicit benchmark,
diagnostic, and interoperability reference; it is never a silent fallback. See the
[native inference goal](docs/native-inference-goal.md) for the full contract.

## Install and configure

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# Any host with Go 1.26+
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# Inspect the shipped profiles
fak agent profiles
```

Use Ponytail to control how strongly the agent resists avoidable machinery and Caveman to
control how compactly it reports:

```bash
fak manage --output-profile caveman:medium --work-profile ponytail:high -- codex \
  "Remove the duplicate cache without adding a dependency."
```

Balanced defaults are `ponytail:medium` and `caveman:medium`. See
[work profiles](docs/work-profiles.md), [response profiles](docs/response-profiles.md), or the
[harness guide](docs/harness-init.md) to build a named agent around the same boundary.

## Documentation

| If you want to… | Start here |
|---|---|
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [Mac](docs/benchmarks/QWEN38-27B-LATEST.md) · [AMD](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) · [NVIDIA](docs/_witnesses/issue-8819-qwen38-cache-attribution/README.md) · [all benchmarks](docs/benchmarks/README.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [all integrations](docs/integrations/) |
| Understand the kernel | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-08-28 vs VERSION 0.45.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-08-28 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
