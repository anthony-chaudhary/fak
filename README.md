<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg">
    <img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320">
  </picture>
</p>

# fak — the Fused Agent Kernel

**Same agent, smaller bill. Wrap Claude Code, Codex, or Cursor in one command — cheaper, self-resuming, and checked on every tool call.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Go Reference](https://pkg.go.dev/badge/github.com/anthony-chaudhary/fak.svg)](https://pkg.go.dev/github.com/anthony-chaudhary/fak) [![Release](https://img.shields.io/github/v/release/anthony-chaudhary/fak?color=blue&label=release&sort=semver)](https://github.com/anthony-chaudhary/fak/releases/latest) [![Go 1.26+](https://img.shields.io/badge/Go-1.26%2B-00ADD8.svg)](go.mod) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/anthony-chaudhary/fak)

<!-- readme-verified: 2026-07-12 vs VERSION 0.40.0 + BENCHMARK-AUTHORITY -->

fak is one small program you run in front of the agent you already use. Your model, IDE, and keys don't change.

```bash
fak guard -- claude   # run it in front of your agent: your Pro/Max plan, no API key, every tool call gets a verdict
```

**Pick your path:** [wrap your agent](#start-in-one-command) · [15-min tutorial — no key, no GPU](docs/fak/tutorial.md) · [see it on video](visuals/hero-video.mp4) · [install](#install)

> **Project truth default:** every dispatchable ticket estimates its work and share of its parent production scope; unqualified “complete” means production complete. See [Production completion and project scope](docs/project-production-completion.md).

## Why people run it

- 💸 **Pay less for the same run** — **~4.1× less work than a tuned warm-cache stack** (up to **6.95×** on bigger models). Every agent shares one cached setup instead of rebuilding it each turn, on top of your provider's cache discount.
- 🛟 **Sessions that don't die.** Long runs trim their own history (**~107K tokens** per trim at the high end) and restart themselves after a crash — you never babysit the context window.
- 🔒 **You stay in control.** Every tool call gets an ALLOW / DENY verdict against a reviewable allow-list before it runs: **362 ns**, in-process, no model in the loop, no network hop.
- 🔌 **Nothing to rewrite.** Drop it in front of Claude Code, Codex, Cursor, or any OpenAI / Anthropic / MCP client. Your model, IDE, and keys stay exactly as they are.

Every number here traces to [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md), and every claim in [CLAIMS.md](CLAIMS.md) carries exactly one tag — `[SHIPPED]`, `[SIMULATED]`, or `[STUB]`.

## Same agent, a fraction of the bill

<p align="center">
  <img src="visuals/75-token-savings-frontdoor.svg" alt="Token-economics card measured over 495 real fak guard sessions: 74.8% of API cost avoided versus a no-cache, no-compaction counterfactual priced at list (OBSERVED). The saving is shown as two independent tracks, side by side and never blended into one number — the provider's own prompt-cache rebate ($3,264 avoided, OBSERVED) and fak's compaction-shed ($3,404 avoided, WITNESSED, 686M context tokens shed). fak_share = 42.4% of the saving is fak-authored. SOTA companions: 4.1x less work than a tuned warm-cache stack, and an 86.7% KV-cache hit rate = 100% of optimal." width="900">
</p>

Across **495 real guarded sessions**, running your agent behind the kernel avoided **74.8% of the API cost** you'd otherwise pay to re-send the same context every turn (OBSERVED, versus a no-cache / no-compaction counterfactual priced at list). The card keeps the two sources honest and *side by side*, never merged into one number: the provider's own prompt-cache rebate — fak keeps the prefix byte-identical so it stays warm — and fak's own compaction-shed, the 686M context tokens it dropped before you paid to re-send them. Measured against the *best already-shipped* alternative, a tuned warm-cache stack, the reuse win is **~4.1×**; of the total saved, the provider's rebate is the provider's, and fak's own authored share is `fak_share = 42.4%`.

Every dollar traces to `fak cachevalue report` and its WITNESSED / OBSERVED ledgers; the reuse multipliers trace to [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md).

## Start in one command

Wrap the agent you already run — no rewrite, no config file, no second terminal:

```bash
fak guard -- claude                                   # your Claude Code on your Pro/Max plan — no API key needed
fak guard --api-key-env ANTHROPIC_API_KEY -- claude   # bill the Anthropic API instead
fak guard --provider openai --api-key-env OPENAI_API_KEY -- opencode   # any OpenAI-compatible agent
```

The gateway runs in-process on loopback and your credential passes through untouched. Every tool call on that boundary gets a verdict, and on exit you get a savings-and-decisions summary. Walkthrough + proof a real `/v1/messages` turn crossed the gateway: [docs/integrations/claude.md](docs/integrations/claude.md).

## Try it now — no key, no GPU

No clone, no model, no card. (Install is one line, [below](#install).)

```bash
fak routebench                                    # -> COST / LATENCY / QUALITY vs a one-model baseline
fak preflight --tool refund_payment --args "{}"   # -> DENY (DEFAULT_DENY): not on the allow-list, fail-closed
```

## More ways to run it

| You want | Run | See |
|---|---|---|
| A local model in the kernel | `fak guard --gguf qwen2.5:7b -- claude` — **~120 tok/s** on a single RTX 4070, no key, no network | a small local model is a quality ramp, not a frontier coder — [head-to-head](docs/benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md) · [hardware matrix](docs/HARDWARE-MATRIX.md) |
| An always-on gateway | `fak node` installs `fak serve` as a real system service | [docs/fak/node-setup.md](docs/fak/node-setup.md) |
| Crash recovery | `fak resume watchdog` relaunches dead sessions (dry-run by default); `fak resume plan` prices a warm vs cold restart | — |
| Codex / Cursor / MCP hosts | keep the normal model wire; ask the kernel for verdicts over MCP | [Codex](docs/integrations/openai-codex.md) · [Cursor](docs/integrations/cursor.md) · [examples/mcp](examples/mcp) |
| Any OpenAI/Anthropic client | `fak serve` in front of a model endpoint | [the serving side](docs/serving/README.md) · [GETTING-STARTED.md](GETTING-STARTED.md) · [docs/fak/api-reference.md](docs/fak/api-reference.md) |
| Slack | a durable run-card per session; `fak chatrelay` bridges a served model into a channel | [docs/fak/slack-sessions.md](docs/fak/slack-sessions.md) |

Tested live with Claude Code, opencode, and Codex; 41 of 47 surveyed harnesses connect with a single base-URL change ([the catalogue](docs/supported/README.md)).

## Every tool call gets a verdict

The speed seam doubles as a safety floor at no extra cost. A tool call crosses the kernel before it runs and comes back ALLOW, DENY, TRANSFORM, or REQUIRE_WITNESS — logged with a fixed reason code you can test against (`POLICY_BLOCK`, `SECRET_EXFIL`, …). Untrusted result bytes are quarantined before they can become the model's next instruction, and the model can't widen its own authority with text alone. The floor is a JSON file you copy, trim, and test — no model in the loop; starter floors ship for coding, customer support, DevOps, trading, and clinical/PHI. Point at one with `fak guard --policy examples/<file>`. More: [POLICY.md](POLICY.md) · [examples/README.md](examples/README.md) · [the tool call is a syscall](docs/explainers/tool-call-is-a-syscall.md).

<p align="center">
  <img src="visuals/74-session-effectiveness.svg" alt="One fak guard session, replayed from its decision journal: 1,720 hash-chained tool-call decisions over a 7h 11m unattended run — 99.77% allowed to proceed, the 4 dangerous calls blocked (an off-policy git action at the git gate, three tainted-data sinks at the IFC gate), zero hash-chain breaks, across 22 distinct tools. A timeline shows verdicts kept pace across the whole run; a work-mix bar breaks the calls into explore/execute/author/mcp/orchestrate/schedule." width="900">
</p>

This is one real `fak guard` run, replayed from its decision journal. An agent worked for **7 hours unattended** while the kernel put a verdict on **every one of 1,720 tool calls** — waving the work through, blocking only the four that were dangerous, and hash-chaining every decision into a record that can't be edited after the fact. Nothing about it is special: it's the default shape of running any agent behind the gate. As agents run longer and more autonomously, a real-time verdict on every call plus a tamper-evident trail stops being a nice-to-have — a long unattended run that blocks only what's dangerous and can *prove* what it did is where every session is heading.

Every number on the card is recomputed from the session's hash-chained journal by [`tools/gen_session_effectiveness_svg.py`](tools/gen_session_effectiveness_svg.py) into a checkable [stats sidecar](visuals/74-session-effectiveness.stats.json); the per-verdict kernel cost traces to [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md).

## Install

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Go 1.26+ is required (as of 2026-07-14; source: [`go.mod`](go.mod)); no external Go dependencies, no `go.sum`. From a clone: `go build -o fak ./cmd/fak`. Prebuilt archives and containers: [INSTALL.md](INSTALL.md). Build/test/ship: [CONTRIBUTING.md](CONTRIBUTING.md).

## The honest fence

For raw token throughput, reach for vLLM or SGLang — fak is the agent kernel around them, not a replacement. Putting a model *behind* the kernel — running the gateway, riding an engine, and scaling out KV reuse — is the other half of `fak`, with its own home: [the serving side](docs/serving/README.md). Prompt-injection classifiers help, but tool authority comes from your policy file, not the model. Keep irreversible or data-exfiltrating tools off the allow-list. Full list: [what fak is not](docs/explainers/what-fak-is-not.md).

## Docs

Going deeper starts at the [front-page overflow](docs/README-legacy.md) — why now, the per-domain use-case catalogue, model routing, vCache, and everything moved off this page.

| If you want... | Read |
|---|---|
| Guided first session (15 min) · absolute-beginner start | [docs/fak/tutorial.md](docs/fak/tutorial.md) · [START-HERE.md](START-HERE.md) |
| Which do I run — gateway, agent runtime, or client? | [docs/explainers/runtime-vs-client.md](docs/explainers/runtime-vs-client.md) |
| The serving side — run a model behind the gateway, ride vLLM/SGLang, scale out KV | [docs/serving/README.md](docs/serving/README.md) |
| Install + the four usage tiers | [GETTING-STARTED.md](GETTING-STARTED.md) · [LEARNING-PATH.md](LEARNING-PATH.md) |
| Long sessions / cache · you never manage context | [docs/explainers/long-sessions-keep-the-cache-hit.md](docs/explainers/long-sessions-keep-the-cache-hit.md) · [docs/explainers/you-never-manage-the-context-window.md](docs/explainers/you-never-manage-the-context-window.md) |
| What "managed cache" is, in plain English | [docs/explainers/what-is-managed-cache.md](docs/explainers/what-is-managed-cache.md) |
| Capability floor (policy) · security model | [POLICY.md](POLICY.md) · [docs/fak/security.md](docs/fak/security.md) |
| CLI verbs · supported models / engines / harnesses | [docs/cli-reference.md](docs/cli-reference.md) · [docs/supported/README.md](docs/supported/README.md) |
| Every feature, by subsystem, with honest status (shipped / simulated / stub) | [docs/supported/features.md](docs/supported/features.md) |
| Benchmark authority · gallery · honesty ledger | [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) · [BENCHMARK-GALLERY.md](BENCHMARK-GALLERY.md) · [CLAIMS.md](CLAIMS.md) |
| Machine-readable map | [llms.txt](llms.txt) |

---

<p align="center">
  <strong>When in doubt, <code>fak</code> it out.</strong><br>
  <sub><em>Once you go fak, you never go back · Give a fak — ship on trunk · Don't take our word for it, take a fak's</em></sub>
</p>

License: [Apache-2.0](LICENSE).
