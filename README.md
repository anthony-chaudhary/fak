<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — useful local agents, accelerated automatically

**fak is the open runtime that makes useful local agents practical on your own machine.**

> In one line: fak serves local models to your coding agents on your own hardware, and refuses every tool call you did not allow.

```bash
# Apple Silicon Mac to start; more platforms and options in Try fak below
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh && fak up
```

Start locally, give an agent real work, and keep useful context across turns.
Our first milestone combines native inference, speculative decoding, and reuse of
the work-so-far (agentic *KV caching*) so qualified acceleration is automatic.
A capability floor bounds what tools the agent may execute.

Status: automatic setup and cache reuse have specific model/backend limits today;
speculative decoding and physical GPU Direct paths are not yet default-on or
universally qualified. See the [local-agent milestone](docs/local-agent-milestone.md)
and the [claims ledger](CLAIMS.md) for what is real and what is not.

## Latest hardware results — 2026-11-24

One row per hardware family: the newest committed performance receipt for that platform,
with its claim boundary and receipt link. A row stays historical until a newer quality-complete
measurement exists.

| Platform | Latest witnessed result | Status & Details |
|---|---|---|
| Mac | Inkling-2-Mini UD-Q3_K_XL on a 36 GB Apple M3 Pro (18-core GPU, Metal): fak-native decode with DFlash2 draft verification reached 97.5–107.4% of Splash 1.7.2 on natural prompts (40.3 vs 37.5 tok/s on 32-token capitals) and 148% on repeated text (177.1 vs 119.6); prefill was 98–108% of Splash; mainline llama.cpp b12681 decodes 18.9 tok/s on the same Mac and GGUF; observed 2026-11-24. | HW-witnessed ABBA, warm single request, same target and draft weights; 128-token Fibonacci trails Splash by 2.5% (63.6 vs 65.2). Short suites match token-for-token; one 128-token prompt diverges at near-ties in both engines. Needs `main` at or after 1f31533f2: **not in the v0.57.0 release**. [M3 Pro Inkling-2 receipt](docs/benchmarks/published/m3pro-inkling2-mini-metal/RESULTS.md) |
| AMD | DeepSeek-V4.1-Flash-Q2_K on one 128 GB AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S gfx1151, Vulkan): warm single-stream decode reached 4.61 tok/s, +29% over the same-session 3.57 base, with greedy output parity in 32/32 requests; observed 2026-10-22. | Physical APU result; the 30 tok/s decode target is not yet met, so this row stands until a newer accepted AMD measurement exists. [Strix Halo V4.1 Flash results](docs/benchmarks/STRIX-HALO-DEEPSEEK-V41-FLASH.md) |
| NVIDIA | Inkling-2-Mini UD-Q3_K_XL on an RTX 5080 16 GB (sm_120): fak-native resident CUDA decode reached 119.9 tok/s single-stream on the default path, about 1.48x upstream llama.cpp tg128 (81.0 on the same card and GGUF) and level with the fastest llama.cpp fork measured (119.5 in the same window); opt-in speculative decode (`FAK_INKLING2_SPEC=2`) reached 151.7 tok/s, +26.6% over plain decode in the same window; greedy output matches the llama.cpp oracle on 3 of 4 cases; observed 2026-11-24. | HW-witnessed ABBA, in-process decode bench, not a served-endpoint rate. Speculative decode is off by default because its batched verify is not bit-exact to the per-token step. The fourth oracle case diverges at a known near-tie, and at 1,946 tokens of context plain decode still trails the fork (89.0 vs 113.8 tok/s). [Plain decode receipt](docs/benchmarks/published/aegis-inkling2-fork-parity/RESULTS.md), [spec decode receipt](docs/benchmarks/published/aegis-inkling2-spec-decode/RESULTS.md) |

Read the status column before comparing rates. History: [benchmark index](docs/benchmarks/README.md);
claim boundaries: [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md).

## Try fak

Installed with the one-liner at the top? Prefer Go: `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`. More install options are in the [install-options matrix](INSTALL.md#install-options-at-a-glance); removal steps are in [Uninstall](INSTALL.md#uninstall).
The Inkling-2 Mac result above needs `main` until the next release: `go install github.com/anthony-chaudhary/fak/cmd/fak@main` (`fak version` shows the commit on its `build:` line).

Already run Claude Code or Codex? If you only want to guard it, skip to
[`fak guard`](#governance-for-external-agents-fak-guard).

1. Start local inference on a supported Apple Silicon Mac. `fak up` sizes the model
   and context to your memory and serves an OpenAI-compatible endpoint on `:8080`.
   Under memory pressure it holds new work back instead of swapping. Add `--mock` to
   walk the workflow with no model and no GPU (mock output is not performance evidence).
   ```bash
   fak up
   # -> [READY] fak up running on http://127.0.0.1:8080
   ```
2. In another terminal, point an agent at it. Use `fak opencode`, `fak codex`,
   `fak claude`, or [Pi](docs/integrations/pi.md). Give it real work, e.g.
   *"Using parallel subagents, audit the packages under internal/ and report their status."*
   `fak up` serves only OpenAI chat completions (enough for OpenCode and Pi). `fak claude` and
   `fak codex` need Messages and Responses: run `fak serve --gguf qwen38:27b-q4` for them.
   No separate router is needed: agents talk to the local endpoint directly.

### Governance for external agents (`fak guard`)

Already running Claude Code or Codex? One command adds a capability floor around it.
The kernel checks every tool call before it runs. Structural denies and danger
rules block by structure; the strict `fail_closed` posture runs only allow-listed tools.
fak forwards your Codex subscription credentials (no API key required):

```bash
fak guard -- codex
```

To see allow/deny decisions with
no key, no model, and no setup, run the offline agent or take the
[interactive showcase](docs/showcase.html):

```bash
fak agent --offline   # -> task completed
```

### Works with the agent you already use

| Agent | Command | Needs |
|---|---|---|
| Claude Code | `fak claude` | `fak serve` (Anthropic Messages, `/v1/messages`) |
| Codex | `fak codex`, or `fak guard -- codex` | `fak serve` (OpenAI Responses, `/v1/responses`) |
| OpenCode | `fak opencode` | `fak up` or `fak serve` (OpenAI chat completions) |
| Pi | `fak pi` · [Pi guide](docs/integrations/pi.md) | `fak up` or `fak serve` (OpenAI chat completions) |

## Your machine, your files

- Local by default. `fak serve` listens on `127.0.0.1:8080`. It refuses to start on an
  address reachable from off this host unless you name an inbound token
  (for example `--require-key-env`) or pass the deliberately named
  `--unsafe-allow-unauthenticated-bind`.
- No opaque model store. Downloaded models are plain files at
  `<cache>/fak-models/hub/<owner>/<repo>/<revision>/<file>` (your user cache directory,
  or `$FAK_MODELS_DIR/hub`). Copy, inspect, or delete them with ordinary tools.
- No phone-home. No analytics SDK is linked, and fak runs no update check on its own:
  `fak self-update` and `fak release-staleness` act only when you invoke them.
- Clean exit. [Uninstall](INSTALL.md#uninstall) covers the binary, services and shims, plus models and settings.
- Credit upstream. Parts of fak are adapted from ggml / llama.cpp;
  attributions and pinned revisions are in [NOTICE.md](NOTICE.md).

## Going deeper

| If you want to… | Start here |
|---|---|
| Understand why fak exists | [why fak, now](docs/explainers/why-fak-the-market-moment.md) |
| See what changed recently | [changes since 2026-11-10](docs/README-legacy.md#changes-since-2026-11-10) |
| Get the exact model files behind the results | [pinned Hugging Face sources](docs/supported/models.md#exact-hugging-face-pins-for-witnessed-models) |
| Sell tokens from your own GPUs | [token-factory operator route](docs/operator/token-factory.md) |
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [all benchmarks](docs/benchmarks/README.md) · [cache-value roll-up](docs/cache-value-rollup.md) · [Mac local models](docs/fak/mac-local-models.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [Pi](docs/integrations/pi.md) · [subagents](docs/subagents-guide.md) · [all integrations](docs/integrations/) |
| Tune agents: work and output profiles | [Work profiles](docs/work-profiles.md) · [response profiles](docs/response-profiles.md) · [harness guide](docs/harness-init.md) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) · [Native inference goal](docs/native-inference-goal.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Work on the fak codebase itself | [first-day engineer guide](docs/engineering/first-day.md) · [contributing](CONTRIBUTING.md) |
| Build on fak, or run a commercial pilot | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) · [commercial serving](docs/README-legacy.md#commercial-serving) |
| Read sections moved off this page | [README overflow](docs/README-legacy.md) — why run on fak, agent profiles, subagent bench, and the 2026-11-24 batch (why fak now summary, speed framing, changes since 2026-11-10) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-11-24 vs VERSION 0.57.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-11-24 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py + /refresh-readme -->
