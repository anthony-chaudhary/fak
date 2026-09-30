<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — useful local agents, accelerated automatically

**Fak is building the open runtime that makes useful local agents practical on your own machine.**

Start locally, give an agent real work, and keep useful context across turns.
Our first milestone is one useful native local coding workflow on one existing
supported device/model, with Strix Halo as the initial proving ground. OpenCode may
supply interaction and tools; Fak-native owns model loading and inference, and
Fak guard/context/lifecycle stay core. Study existing OSS and incorporate useful
techniques before reinventing them; direct OSS development, business proofs of
concept, and comparisons remain a separate evidence plane.

**Status:** this native workflow remains a milestone to qualify. Pin one supported
model/quantization by current compatibility evidence; earn a normal documented start,
an independently accepted repository change, restart/follow-up, context invalidation,
engine identity, and a current tuned OSS comparison within declared quality/time/memory
budgets. Count failures and all overhead. Optional speculative and direct-I/O paths
require their own qualification. Ship one notable milestone before compute expansion.
See the [milestone](docs/local-agent-milestone.md), [engine doctrine](docs/native-inference-goal.md),
and [product orientation](docs/project-orientation.md#primary-adaptation-index).

**Where it stands on speed:** on an Apple M3 Pro running Qwen3.8-27B,
fak's own Metal engine measured 6.86 decode tok/s, 0.985× a pinned llama.cpp
reference build on the same Mac, with token-for-token identical output
(observed 2026-09-03). That is
parity with the strongest local engine, not yet a lead; see the
[Qwen results](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) for the receipt.

**Pick your path:** run a local agent → [Try fak](#try-fak) · guard an agent you
already use → [`fak guard`](#governance-for-external-agents-fak-guard) · check the
evidence → [benchmarks](docs/benchmarks/README.md) and [claims](CLAIMS.md).

## Try fak

Install with `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh` (or `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`).

Try the local workflow on a supported Apple Silicon configuration:

1. Start local inference (`fak up`):
   Sizes a model/context budget from unified memory (`macfit`), holds new work
   back under memory pressure instead of swapping, and starts the local OpenAI-compatible endpoint on
   `:8080` and opens an interactive chat REPL. Use `fak up --mock` to inspect the workflow without
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
   The same local endpoint also backs Claude Code (`fak claude`, via an
   Anthropic Messages adapter), Codex (`fak codex`), and [Pi](docs/integrations/pi.md).
   Compatible agents reuse shared instructions and repository context; a fresh
   prefix still needs prefill, and reuse depends on model, backend, and cache identity.

3. Inspect the subagent benchmark:
   ```bash
   fak bench subagent --concurrency=4
   ```
   Check its engine, regime, and receipt before treating output as hardware
   evidence; it does not replace a real coding-task acceptance witness.

### Governance for external agents (`fak guard`)

Already running Claude Code or Codex? Wrap the agent you already run with one command to add a default-deny capability floor (only allowed tools run; everything else is blocked). fak forwards Codex subscription credentials with no API key required and blocks tools outside the allowed policy without breaking the task:

```bash
fak guard -- codex
```

In-kernel policy adjudication checks every tool call in under a microsecond before execution. See the [interactive showcase](docs/showcase.html) for a guided tour, or run `fak agent --offline` (# -> task completed) to inspect policy decisions with zero setup.

## Latest hardware results — 2026-09-25

One row per hardware family: the newest committed performance receipt for that platform,
with its claim boundary and receipt link. A row stays historical until a newer quality-complete
measurement exists.

| Platform | Latest witnessed result | Status & Details |
|---|---|---|
| Mac | Qwen3.8-27B Q4_K_M on Apple M3 Pro: forward-owned Metal sequence prefill was 43.8% faster, 10,284.5 vs 18,304.9 ms, and used 1 command buffer instead of 192; observed 2026-09-03. | Accepted component-path result with exact greedy continuation and zero fallbacks; it is not a full-run throughput comparison. [Qwen result index](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) |
| AMD | Qwen3.6-27B on RX 7600: the pure-fak TG1 microbench measured 1.24 decode tok/s versus 0.99 for the local llama.cpp Vulkan baseline; observed 2026-06-19. | Narrow, older-model microbench. No accepted current Qwen3.8 AMD result exists. [AMD receipt](docs/benchmarks/QWEN36-AMD-VULKAN-RESULTS.md) |
| NVIDIA | Qwen2.5-3B Q8_0 on a physical Hopper H100: fak reached 111.9 decode tok/s, 17.4% above its f32 path; observed 2026-09-05. | Native CUDA result; llama.cpp Q8_0 was 3.24× as fast at 362.7 tok/s in the same run. [H100 receipt](docs/benchmarks/GCP-H100-RESULTS.md) |

Read the status column before comparing rates. History: [benchmark index](docs/benchmarks/README.md);
claim boundaries: [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md); Mac setup:
[Mac local models](docs/fak/mac-local-models.md) and [Mac agent UI](docs/fak/mac-agent-ui.md).

## Why run coding agents on fak

- **Reuse the work behind each turn:** Compatible prefix/KV caching avoids
  rebuilding shared instructions and context. The product target is automatic
  reuse across turns and compatible agents, with correct invalidation and isolation.
- **Accelerate generation automatically:** Native kernels, memory sizing,
  quantization, and speculative decoding are parts of one local workflow. The
  acceleration target requires qualified defaults; current MTP decoding requires explicit
  selection. See the [implementation snapshot](docs/local-agent-milestone.md#current-implementation-is-narrower-than-the-milestone).
- **Real-time multi-agent visibility:** Inspect live cross-agent reuse rates, per-subagent token breakdowns, and savings sparklines directly in your terminal overlay (`fak info` / `fak guard`) to see and verify the speedup as subagents execute concurrently.
- **Keep reusable state close to compute:** Device-resident caching and direct GPU
  storage paths aim to cut paging and copy overhead; GPU residency and physical
  NVMe-to-GPU DMA are separate claims, qualified in the [claim ledger](CLAIMS.md).
- **Run on your own hardware:** Native backends target Apple Silicon (fak-native
  Metal), AMD and Strix Halo (bundled Vulkan), and NVIDIA
  (`ghcr.io/anthony-chaudhary/fak:cuda-latest` with `--gpus all`), each with its own
  support envelope; direct OSS development, business PoCs, and comparisons stay in
  a separate evidence plane and cannot close the native product milestone
  ([native inference goal](docs/native-inference-goal.md)).
- **Default-deny capability floor:** Protect your workspace from unintended commands, path escapes, or tool poisoning. Every tool call is verified against a capability floor before execution; subagents get their own narrower floor, and a circuit breaker stops an agent stuck retrying a failing tool. Drop-in wrappers protect existing agents like Claude Code, Codex, OpenCode, and Cursor with zero rewrites.

## Configure agent profiles

Built-in work and output profiles cut token waste and resist unnecessary dependencies:

```bash
fak agent profiles
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
| Is the cache paying off? (trend) | [Cache-value roll-up](docs/cache-value-rollup.md) — kernel reuse and provider-dollar savings kept in separate, unblended tracks |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [Pi](docs/integrations/pi.md) · [subagents](docs/subagents-guide.md) · [Mac local models](docs/fak/mac-local-models.md) · [all integrations](docs/integrations/) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |

## Commercial serving

We run one repetitive repo workload (test-candidate generation, codebase Q&A, or doc
maintenance) on a managed, metered inference route and prove it against your baseline
and acceptance criteria in a fixed-fee two-week pilot before ongoing metered operation.
Inference quality is [SW-VERIFIED] at raw-compute parity; production readiness is not yet
claimed. To start, open an issue describing your workload (channel=oss).

Apache-2.0 licensed.

<!-- readme-verified: 2026-09-27 vs VERSION 0.55.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-09-09 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
