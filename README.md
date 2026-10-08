<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — useful local agents, accelerated automatically

**Fak is building the open runtime that makes useful local agents practical on your own machine.**

Start locally, give an agent real work, and keep useful context across turns.
Our first milestone combines native inference, speculative decoding, and reuse of
the work-so-far (agentic *KV caching*) so qualified acceleration is automatic.
A capability floor bounds what tools the agent may execute.

**Status:** automatic setup and cache reuse have specific model/backend limits today;
speculative decoding and physical GPU Direct paths are not yet default-on or
universally qualified. See the [local-agent milestone](docs/local-agent-milestone.md)
and the [claims ledger](CLAIMS.md) for what is real and what is not.

**Where it stands on speed** ([Qwen results](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md)):

- Mac (M3 Pro, Qwen3.8-27B): fak's Metal engine measured 43.8% faster prefill than
  its own per-op reference path (observed 2026-09-03). That is a component result, not a full run.
- Versus llama.cpp, the strongest local engine: no accepted full-run Mac comparison
  exists; the earlier near-parity figure was withdrawn on 2026-09-27 pending remeasurement.
- NVIDIA H100: fak measured 111.94 decode tok/s; llama.cpp is still about 3× faster there.

llama.cpp is an explicit external reference for comparison only; fak never selects it
as a fallback ([native inference goal](docs/native-inference-goal.md)).

**Pick your path:** run a local agent → [Try fak](#try-fak) · guard an agent you
already use → [`fak guard`](#governance-for-external-agents-fak-guard) · check the
evidence → [benchmarks](docs/benchmarks/README.md) and [claims](CLAIMS.md).

## Try fak

Install with `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh` (or `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`).

1. Start local inference on a supported Apple Silicon Mac. `fak up` sizes the model
   and context to your memory, holds new work back under memory pressure instead of
   swapping, and serves an OpenAI-compatible endpoint on `:8080`. Add `--mock` to
   walk the workflow with no model and no GPU (mock output is not performance evidence).
   ```bash
   fak up
   # -> [READY] fak up running on http://127.0.0.1:8080
   ```
2. In another terminal, point an agent at it: `fak opencode`, `fak codex`,
   `fak claude`, or [Pi](docs/integrations/pi.md). Give it real work, e.g.
   *"Using parallel subagents, audit the packages under internal/ and report their status."*

### Governance for external agents (`fak guard`)

Already running Claude Code or Codex? Wrap the agent you already run with one command
to add a default-deny capability floor: only allowed tools run, everything else is
blocked. fak forwards your Codex subscription credentials (no API key required):

```bash
fak guard -- codex
```

The guard rules on every tool call before it runs. To see allow/deny decisions with
no key, no model, and no setup, run the offline agent or take the
[interactive showcase](docs/showcase.html):

```bash
fak agent --offline   # -> task completed
```

## What changed since v0.55.0 — 2026-10-03

- **`fak claude` routes through the live fak router by default** (`FAK_ROUTER_URL`,
  else `127.0.0.1:18101`); `--no-router` keeps the direct Mac setup. OpenCode and Pi
  default workers also route through it, so one paused provider no longer stops every worker.
- **`fak up` serves several agent sessions at once**, admitting work against capacity
  with backpressure instead of timing out.
- **Serving engine:** continuous batching is wired into decode, and concurrent cold
  requests share one prefill. Code-verified; no accepted throughput receipt yet.
- **Mac GPU keepalive (off by default):** a 2026-10-03 point-in-time run on a busy
  M3 Pro measured 19.7 vs 9.4 decode tok/s on the small Qwen3.5-0.8B model; prefill
  did not improve and 27B was not tested ([receipt](docs/benchmarks/receipts/mac-keepalive/20261003/REPORT.md)).
- **Guard hardening:** context changes, memory apply, and plugin writes are refused
  unless an act-bound lease fences them.

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
claim boundaries: [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md).

## Going deeper

| If you want to… | Start here |
|---|---|
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [all benchmarks](docs/benchmarks/README.md) · [cache-value roll-up](docs/cache-value-rollup.md) · [Mac local models](docs/fak/mac-local-models.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [Pi](docs/integrations/pi.md) · [subagents](docs/subagents-guide.md) · [all integrations](docs/integrations/) |
| Tune agents: work and output profiles | [Work profiles](docs/work-profiles.md) · [response profiles](docs/response-profiles.md) · [harness guide](docs/harness-init.md) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) · [Native inference goal](docs/native-inference-goal.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Build on fak, or run a commercial pilot | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) · [commercial serving](docs/README-legacy.md#commercial-serving) |
| Read sections moved off this page | [README overflow](docs/README-legacy.md) — why run on fak, agent profiles, subagent bench |

Apache-2.0 licensed.

<!-- readme-verified: 2026-10-03 vs VERSION 0.55.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-09-09 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py -->
