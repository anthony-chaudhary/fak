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

## Why fak, now

Coding agents crossed from demo to daily driver, and iteration collapsed from
days to minutes. The bottleneck moved downstream with it: review throughput,
verification, and trust now cap what an agent fleet ships. The
faster the loop, the more PRs pile up at the review gate, and the more the human
becomes the serialization point in an otherwise parallel system.

The two answers the market offers are binary, and both fail at fleet scale.
**Gate every step by hand** and review load scales with agent output: the speed
the agent was bought for is thrown away at the review step. **Just trust the
AI** and the queue disappears along with the safety net: silent breakage, a
"tests pass" that was never run, no audit trail.

fak is the third option: trust as a definition you write down, not a leap.

| The binary | fak's middle path |
|---|---|
| A human approves every tool call ↔ the model calls anything | A capability floor on every tool call: structural denies and danger rules refuse by structure, not by a recognizer that can fail open — shipped `default_open`, strict `fail_closed` one flag away |
| Believe the agent's "done" ↔ verify every claim yourself | The kernel checks the claim against git evidence: the subject is forgeable, the diff is not |
| Hope parallel agents don't collide ↔ coordinate them by hand | Lane leases and arbitration decide who may run on what, and refuse with a structured reason a peer can act on |
| Silent overrides ↔ permanent human locks | Graduated exceptions: an attending operator approves one exact refused call; the agent cannot approve itself |

Every row is the scar of a failure observed running agent work for real. fak
is the recorded list of what broke. The same shift
reshapes serving: fleets need batching, cache reuse, failover, and preemption.
That is the runtime's other half. Full framing:
[why fak, now](docs/explainers/why-fak-the-market-moment.md).

**Status:** automatic setup and cache reuse have specific model/backend limits today;
speculative decoding and physical GPU Direct paths are not yet default-on or
universally qualified. See the [local-agent milestone](docs/local-agent-milestone.md)
and the [claims ledger](CLAIMS.md) for what is real and what is not.

**Speed:** the newest witnessed result for each platform, with its claim boundary
and receipt link, lives in the Latest hardware results section immediately below.
Versus llama.cpp on Mac there is no accepted full-run comparison yet. The earlier
near-parity figure was withdrawn on 2026-09-27 pending remeasurement. llama.cpp is
an explicit external reference for comparison only; fak never selects it as a
fallback ([native inference goal](docs/native-inference-goal.md)).

**Pick your path:** run a local agent → [Try fak](#try-fak) · guard an agent you
already use → [`fak guard`](#governance-for-external-agents-fak-guard) ·
sell tokens from your own GPUs → [token-factory operator route](docs/operator/token-factory.md) ·
understand why → [the market framing](docs/explainers/why-fak-the-market-moment.md) ·
check the evidence → [benchmarks](docs/benchmarks/README.md) and [claims](CLAIMS.md).

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

Install with `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh` (or `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`). More install options are in the [install-options matrix](INSTALL.md#install-options-at-a-glance); removal steps are in [Uninstall](INSTALL.md#uninstall).

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

The guard rules on every tool call before it runs. To see allow/deny decisions with
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

- **Local by default.** `fak serve` listens on `127.0.0.1:8080`. It refuses to start on an
  address reachable from off this host unless you name an inbound token
  (for example `--require-key-env`) or pass the deliberately named
  `--unsafe-allow-unauthenticated-bind`.
- **No opaque model store.** Downloaded models are plain files at
  `<cache>/fak-models/hub/<owner>/<repo>/<revision>/<file>` (your user cache directory,
  or `$FAK_MODELS_DIR/hub`). Copy, inspect, or delete them with ordinary tools.
- **No phone-home.** No analytics SDK is linked, and fak runs no update check on its own:
  `fak self-update` and `fak release-staleness` act only when you invoke them.
- **Clean exit.** [Uninstall](INSTALL.md#uninstall) covers the binary, services and shims, plus models and settings.
- **Credit upstream.** Parts of fak are adapted from ggml / llama.cpp;
  attributions and pinned revisions are in [NOTICE.md](NOTICE.md).

## What changed since 2026-11-10

- Inkling-2-Mini runs fak-native on Apple Metal: `fak run <path>/Inkling-2-Mini-UD-Q3_K_XL.gguf`
  picks Metal automatically and verifies DFlash2 draft tokens on the GPU when an
  `Inkling-2-Mini-DFlash2/` directory sits next to the GGUF (receipt in the table above).
  It landed after v0.57.0, so install from `main` (`go install github.com/anthony-chaudhary/fak/cmd/fak@main`)
  until the next release; `fak version` shows the commit on its `build:` line.
- A second model family runs fak-native on NVIDIA: Inkling-2-Mini decodes resident on
  CUDA with one replayed graph per token, 119.9 tok/s on an RTX 5080 (receipt in the
  table above). Speculative decode is opt-in (`FAK_INKLING2_SPEC=2` plus
  `FAK_INKLING2_MTP_DIR`): the checkpoint's own draft head proposes two tokens and one
  batched forward checks them, 151.7 tok/s on the same card. It stays off by default
  because the batched check is not bit-exact to one-token-at-a-time decode. Its tokenizer
  now matches Hugging Face on 46/46 corpus lines. A batched prompt prefill is opt-in
  behind `FAK_INKLING2_BATCHED_PREFILL=1`; per-token prefill stays the default.
- CUDA prefill defaults: dense K-quant prompt panels batch through the mmq kernel,
  and Q6_K batched matmul uses mmq at 8+ prompt tokens on sm_80 and newer. The Q4_K
  mmq default from 2026-11-07 stands.
- `fak serve` on CUDA falls back to fit-with-offload when a dense model does not fit
  in VRAM, and exits on a fatal warmup failure instead of serving a broken engine.
- Cache reuse: native child tasks share the parent's stable system and tools
  preamble, and the gateway exports `fak_gateway_fusion_cache_*` metrics that price
  the prefill saved against no-reuse and same-session-only baselines.
- Gateway: `GET` and `DELETE /v1/responses/{id}`, `service_tier` accepted, and a
  forced hosted tool refused with a typed 400.
- Guard policy: a `loop_depth` rule caps a looped model's recurrent depth per tool
  scope. Still current from the previous window: an attending operator approves one
  exact refused call with `fak guard allow --approve` (the agent cannot approve
  itself), and workspace admission defaults to the open posture.
- Docs: the router is optional (agents talk to `fak up` / `fak serve` directly), and
  GPU owners get a [token-factory operator route](docs/operator/token-factory.md).

## Models fak has actually run — exact Hugging Face pins

Every result above measured a pinned artifact, not "latest". The URLs below are the
sources behind those receipts; `fak ls` aliases resolve to the same pins
(`internal/modelreg`). Where a receipt records only a filename, the official model card
is linked instead of an unrecorded mirror.

| Model (artifact) | Pinned Hugging Face source | Witnessed on |
|---|---|---|
| Qwen3.8-27B Q4_K_M (`qwen38`, default) | [unsloth/Qwen3.8-27B-GGUF@f1bfb127](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe) · [official Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apple M3 Pro, Metal |
| Qwen3.6-27B Q4_K_M | [lmstudio-community/Qwen3.6-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.6-27B-GGUF) | AMD RX 7600, Vulkan |
| Qwen2.5-3B-Instruct Q8_0 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (receipt records the Q8_0 GGUF filename only) | NVIDIA H100, CUDA |
| Inkling-2-Mini UD-Q3_K_XL | [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (file under `Mainline_Support/`; receipt records its sha256) · Metal draft [incoai/Inkling-2-Mini-DFlash2@4f9088ea](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2) | NVIDIA RTX 5080, CUDA · Apple M3 Pro, Metal |
| DeepSeek-V4.1-Flash Q2_K (7 shards, 264,515,279,456 B) | [vcruz305/DeepSeek-V4.1-Flash-GGUF@c4a08554](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/c4a085541cb53f67ee5e57b63d255e80cef286e7) · [official deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Strix Halo, Vulkan |
| Qwen3.5-0.8B Q4_K_M | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Apple M3 Pro, Metal keepalive |
| Prism Bonsai-2-27B PQ2_0 | [prism-ml/Ternary-Bonsai-2-27B-gguf@b072e1d3](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/b072e1d3b35a0a630cece372c2127528e0994386) | Apple M3 Pro, CPU |
| SmolLM2-135M-Instruct Q8_0 | [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) | in-kernel bit-exact HF oracle fixture |

## Going deeper

| If you want to… | Start here |
|---|---|
| Check what is shipped, limited, or planned | [Status](STATUS.md) · [claims](CLAIMS.md) · [feature matrix](docs/supported/features.md) |
| Browse performance evidence | [all benchmarks](docs/benchmarks/README.md) · [cache-value roll-up](docs/cache-value-rollup.md) · [Mac local models](docs/fak/mac-local-models.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [Pi](docs/integrations/pi.md) · [subagents](docs/subagents-guide.md) · [all integrations](docs/integrations/) |
| Tune agents: work and output profiles | [Work profiles](docs/work-profiles.md) · [response profiles](docs/response-profiles.md) · [harness guide](docs/harness-init.md) |
| Understand the runtime | [Architecture](ARCHITECTURE.md) · [capability map](docs/CAPABILITIES.md) · [CLI reference](docs/cli-reference.md) · [Native inference goal](docs/native-inference-goal.md) |
| Learn in prerequisite order | [Start here](START-HERE.md) · [learning path](LEARNING-PATH.md) · [documentation index](docs/index.md) |
| Work on the fak codebase itself | [first-day engineer guide](docs/engineering/first-day.md) · [contributing](CONTRIBUTING.md) |
| Build on fak, or run a commercial pilot | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) · [commercial serving](docs/README-legacy.md#commercial-serving) |
| Read sections moved off this page | [README overflow](docs/README-legacy.md) — why run on fak, agent profiles, subagent bench |

Apache-2.0 licensed.

<!-- readme-verified: 2026-11-24 vs VERSION 0.57.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-11-24 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py + /refresh-readme -->
