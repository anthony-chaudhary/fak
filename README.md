# fak: the Fused Agent Kernel

**One binary in front of the agent you already run: a smaller token bill, the right model per call, and long sessions that pick themselves back up — with safe-by-default tool calls riding along on the same checkpoint.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Go Reference](https://pkg.go.dev/badge/github.com/anthony-chaudhary/fak.svg)](https://pkg.go.dev/github.com/anthony-chaudhary/fak) [![Release](https://img.shields.io/github/v/release/anthony-chaudhary/fak?color=blue&label=release&sort=semver)](https://github.com/anthony-chaudhary/fak/releases/latest) [![Go 1.26+](https://img.shields.io/badge/Go-1.26%2B-00ADD8.svg)](go.mod)

<!-- readme-verified: 2026-07-07 vs VERSION 0.37.0 + BENCHMARK-AUTHORITY -->

fak treats every tool call like a syscall: the model proposes, the kernel disposes — the stable setup (system prompt, tools, KV cache) is computed once and reused, repeated reads are served locally, and old turns are shed while the provider's cache stays alive, so the same loop comes out cheaper, faster, and longer-running. Each call also gets a verdict against a default-deny capability floor (a reviewable allow-list) on that same seam, so it stays controlled without a second component. It works with Claude Code, Codex, Cursor, and OpenAI / Anthropic / MCP clients; your model, IDE, and keys stay exactly as they are.

**Pick your path:** [wrap your agent now](#get-started-with-fak-guard) ·
[guided tutorial — 15 min, no key, no GPU](docs/fak/tutorial.md) ·
[fak in 41 seconds, on video](visuals/hero-video.mp4) ·
[Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb) ·
[tool-call controls](#tool-call-controls) · [install](#install).

## What you get, in numbers

Every figure traces to [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md); the honesty ledger is [CLAIMS.md](CLAIMS.md), where every claim carries exactly one tag: `[SHIPPED]`, `[SIMULATED]`, or `[STUB]`.

| Number | What it means | The fence |
|---|---|---|
| **~4.1× less work than a tuned warm-cache stack** | 50-turn × 5-agent run: the shared setup — system prompt, tools, the model's KV cache — is computed once and reused by every agent | reuse climbs to **6.95×** up the model ladder |
| **~107K tokens per trim** | history compaction on our longest measured session, firing repeatedly as the session grows | each trim saves on top of the provider's prompt-cache discount, not instead of it |
| **~120 tok/s** | in-kernel CUDA decode on a single RTX 4070 | inside llama.cpp's measured band; the 4-platform sweep: [docs/HARDWARE-MATRIX.md](docs/HARDWARE-MATRIX.md) |
| **~362 ns** per call | the guard tax: the allow/deny decision runs in-process (measured, Apple M3 Pro) | per-call overhead, no network hop |

The honest split most cost pitches skip: on the Claude Code route the provider's prompt-cache discount is the bigger number — fak keeps it alive (cache markers pass through byte-for-byte), and fak's own authored slice on top is the smaller one: fleet-wide ~0.3–16% of the total saved (`fak cachevalue report`), climbing on long sessions. Six token-savers are on by default, the first three lossless: [the scorecard + split](docs/serving/token-defaults-scorecard.md) · [full attribution](docs/notes/SESSION-CACHE-SAVINGS-ABLATION-2026-06-29.md).

## Get started with `fak guard`

Wrap the agent you already run in one command — no rewrite, no config edit, no second terminal:

```bash
fak guard -- claude                                   # your Claude Code, on your Pro/Max subscription; no API key needed
fak guard --api-key-env ANTHROPIC_API_KEY -- claude   # Anthropic API billing instead
fak guard --provider openai --api-key-env OPENAI_API_KEY -- opencode   # an OpenAI-compatible agent
```

The gateway starts in-process on loopback and your credential passes through untouched. Every tool call on that boundary receives a verdict, and on exit you get a decision summary. Walkthrough + proof a real `/v1/messages` turn crossed the gateway: [docs/integrations/claude.md](docs/integrations/claude.md).

**Try it first — no clone, no key, no model, no GPU** (install is one `go install`, [below](#install)):

```bash
fak routebench                                   # -> COST / LATENCY / QUALITY delta vs a one-model baseline
fak preflight --tool refund_payment --args "{}"  # -> DENY (DEFAULT_DENY): not on the allow-list, fail-closed
```

## More ways to run it

| You want | Run | See |
|---|---|---|
| A local model in the kernel | `fak guard --gguf qwen2.5:7b -- claude` — no key, no network, data stays on the box | honest fence: a small local model is a quality ramp, not a frontier coder — [head-to-head vs llama.cpp](docs/benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md) |
| An always-on gateway | `fak node` installs `fak serve` as a real system service | [docs/fak/node-setup.md](docs/fak/node-setup.md) |
| Crash recovery | `fak resume watchdog` relaunches dead sessions (dry-run by default); `fak resume plan` prices a warm vs cold resume | — |
| Codex / Cursor / MCP hosts | keep the normal model wire; ask the kernel for verdicts over MCP | [Codex](docs/integrations/openai-codex.md) · [Cursor](docs/integrations/cursor.md) · [examples/mcp](examples/mcp) |
| Any OpenAI/Anthropic-compatible client | `fak serve` in front of a model endpoint | [GETTING-STARTED.md](GETTING-STARTED.md) · [docs/fak/api-reference.md](docs/fak/api-reference.md) |
| Slack | a durable run-card per guard session; `fak chatrelay` bridges a served model into a channel | [docs/fak/slack-sessions.md](docs/fak/slack-sessions.md) |

Witnessed live in front of Claude Code, opencode, and Codex; 41 of 47 surveyed harnesses repoint with one base URL ([the catalogue](docs/supported/README.md)).

## Tool-call controls

The performance seam doubles as a safety floor at no extra cost: a tool call crosses the kernel before it runs, receives a verdict — ALLOW, DENY, TRANSFORM, or REQUIRE_WITNESS — and is recorded with a closed reason code you can assert on (`POLICY_BLOCK`, `SECRET_EXFIL`, …); distrusted result bytes are quarantined before they become the model's next instructions, and the model cannot widen its own authority by text alone. The floor is a deployable JSON manifest you copy, trim, and test — no model in the loop; starter floors cover coding, customer support, DevOps, trading, and clinical/PHI. Point an agent at one with `fak guard --policy examples/<file>`. More: [POLICY.md](POLICY.md) · [examples/README.md](examples/README.md) · [the tool call is a syscall](docs/explainers/tool-call-is-a-syscall.md).

## Install

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Go 1.26+ is required; no external Go dependencies, no `go.sum`. From a clone: `go build -o fak ./cmd/fak`. Prebuilt archives and containers: [INSTALL.md](INSTALL.md). Build/test/ship: [CONTRIBUTING.md](CONTRIBUTING.md).

## Boundaries

For raw token throughput use vLLM or SGLang — fak is the agent kernel around them. Injection classifiers help, but tool authority comes from policy. Provider cache hits are rebates, telemetry until you control the memory. Keep irreversible, exfil-shaped tools off the allow-list. The full fence line: [what fak is not](docs/explainers/what-fak-is-not.md).

## Docs map

Going deeper starts at the [front-page overflow](docs/README-legacy.md): why now, the per-domain use-case catalogue, vCache, model routing, and everything moved off this page.

| If you want... | Read |
|---|---|
| Guided first session (15 min) · absolute-beginner start | [docs/fak/tutorial.md](docs/fak/tutorial.md) · [START-HERE.md](START-HERE.md) |
| Install + the four usage tiers | [GETTING-STARTED.md](GETTING-STARTED.md) · [LEARNING-PATH.md](LEARNING-PATH.md) |
| Long sessions / cache · you never manage context | [docs/explainers/long-sessions-keep-the-cache-hit.md](docs/explainers/long-sessions-keep-the-cache-hit.md) · [docs/explainers/you-never-manage-the-context-window.md](docs/explainers/you-never-manage-the-context-window.md) |
| Capability floor (policy) · security model | [POLICY.md](POLICY.md) · [docs/fak/security.md](docs/fak/security.md) |
| CLI verbs · supported models / engines / harnesses | [docs/cli-reference.md](docs/cli-reference.md) · [docs/supported/README.md](docs/supported/README.md) |
| Benchmark authority · gallery · honesty ledger | [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) · [BENCHMARK-GALLERY.md](BENCHMARK-GALLERY.md) · [CLAIMS.md](CLAIMS.md) |
| Machine-readable map | [llms.txt](llms.txt) |

License: [Apache-2.0](LICENSE).
