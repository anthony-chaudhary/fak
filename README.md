<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — the Fused Agent Kernel

**fak turns a tool-using agent into a managed agent.** The agent keeps its interface and model, while a fak kernel manages its model traffic, context lifetime, cache reuse, capabilities, and recovery.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Go Reference](https://pkg.go.dev/badge/github.com/anthony-chaudhary/fak.svg)](https://pkg.go.dev/github.com/anthony-chaudhary/fak) [![Release](https://img.shields.io/github/v/release/anthony-chaudhary/fak?color=blue&label=release&sort=semver)](https://github.com/anthony-chaudhary/fak/releases/latest) [![Go 1.26+](https://img.shields.io/badge/Go-1.26%2B-00ADD8.svg)](go.mod) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/anthony-chaudhary/fak)

<!-- readme-verified: 2026-07-15 vs VERSION 0.41.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme -->

## One managed agent, two ways to run the kernel

A **managed agent** is still Claude Code, Codex, opencode, or your own client. fak does not replace it or its model. It owns the operational boundary around it: what context is retained, what work is reused, which tools may run, what results may return, and how the session recovers.

The roles stay separate: the **agent or client** owns the task loop and user experience; the **fak kernel** is the management plane; the **model provider or server** generates tokens. Change any one without replacing the others.

| Where the kernel runs | Start here | Use it when… |
|---|---|---|
| **Beside one agent** | [`fak guard`](#manage-one-local-agent-fak-guard) | You want a local launcher and supervisor for an existing agent. `guard` starts a private kernel gateway, points the child agent at it, and manages that session. |
| **As an endpoint** | [`fak serve`](#run-the-managed-agent-endpoint-fak-serve) | One or many OpenAI, Anthropic, or MCP clients need a durable kernel service. It can front an existing model server or load GGUF itself. |
| **Entirely offline** | [Run the one-minute proof](#try-the-kernel-without-a-key-model-or-gpu) | You want to evaluate the managed-agent path without an API key, model download, or GPU. |

Both commands run the **same kernel**. `guard` is the convenient client-side lifecycle wrapper; `serve` is the standalone/server-side deployment. The model remains a replaceable backend—cloud API, local model server, or in-kernel GGUF.

```mermaid
flowchart LR
    A[Agent or client] --> G{How is it managed?}
    G -->|one local session| Guard[fak guard]
    G -->|shared or durable endpoint| Serve[fak serve]
    Guard --> K[the same fak kernel]
    Serve --> K
    K --> M[context + cache + recovery]
    K --> P[capabilities + result admission]
    K --> B[cloud API, model server, or GGUF]
```

For architecture, current evidence, and development workflow, start at [START-HERE.md](START-HERE.md).

## What the managed agent gains

- **Less repeated work.** fak keeps shared prompt prefixes stable, sheds stale history before it is sent again, and reuses KV directly when it owns inference: **~4.1× less work than a tuned warm-cache stack** (up to **6.95×** on larger models).
- **Long runs keep moving.** Sessions compact their own history (up to **~107K tokens** per trim) and can resume after a crash instead of dying at the context limit.
- **Policy is on the execution path.** Every proposed tool call receives ALLOW, DENY, TRANSFORM, or REQUIRE_WITNESS against a reviewable capability floor—**362 ns** in process, with no policy model or network hop.
- **Composable or standalone.** Use either capability, combine both, proxy an existing server, or run GGUF locally.

Evidence: [tuned benchmark baselines](BENCHMARK-AUTHORITY.md) · [tagged claims ledger](CLAIMS.md).

<p align="center">
  <img src="visuals/75-token-savings-frontdoor.svg" alt="Measured token-economics summary for real fak guard sessions, separating provider prompt-cache rebates from fak-authored compaction savings and comparing fak with a tuned warm-cache baseline." width="900">
</p>

## Manage one local agent: `fak guard`

Start with the agent you already run—no rewrite, config file, API key, or second terminal:

```bash
fak guard -- claude                                  # keep a Claude Pro/Max subscription
fak guard --provider openai -- codex                 # provider is normally auto-detected
fak guard --policy examples/dev-agent-policy.json -- opencode
```

Your model wire and credentials pass through unchanged. API billing is opt-in with `--api-key-env`; a local model with `--gguf` or `--local`. On exit, fak reports savings and decisions.

Read: [Claude Code walkthrough](docs/integrations/claude.md) · [Codex](docs/integrations/openai-codex.md) · [Cursor](docs/integrations/cursor.md) · [all supported hosts](docs/supported/README.md) · [policy guide](POLICY.md)

## Run the managed-agent endpoint: `fak serve`

Use fak as a shared OpenAI, Anthropic, or MCP endpoint, with reuse, compaction, routing, observability, authentication, and policy in one place.

**Front an existing local or remote model server:**

```bash
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/dev-agent-policy.json
```

**Or stand alone with GGUF weights—no separate model server:**

```bash
fak serve --addr 127.0.0.1:8080 \
  --gguf /path/to/model.gguf \
  --model my-local-model
```

Point OpenAI clients at `http://127.0.0.1:8080/v1` and Anthropic clients at the bare host. Use `--stdio` for MCP or `fak node install` for an always-on service. Your inference backend stays in place; fak owns cross-request reuse and the agent/tool boundary.

Read: [server quickstart](docs/fak/server-quickstart.md) · [serving architecture and engines](docs/serving/README.md) · [configuration](docs/fak/server-config.md) · [API reference](docs/fak/api-reference.md) · [deployment](docs/fak/deployment-guide.md)

## Try the kernel without a key, model, or GPU

```bash
fak routebench                                    # COST / LATENCY / QUALITY vs a one-model baseline
fak preflight --tool refund_payment --args "{}"   # DENY (DEFAULT_DENY): outside the capability floor
fak agent --offline                               # deterministic end-to-end agent demo
```

The policy floor is JSON, not a second model. Copy a manifest from [`examples/`](examples/), trim it, and test it before a model runs. [How the boundary works](docs/explainers/tool-call-is-a-syscall.md).

## Install

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Go 1.26+; no external Go dependencies. Source builds, archives, and containers: [INSTALL.md](INSTALL.md).

## Where to go next

- **Use it:** [tutorial](docs/fak/tutorial.md) · [integration guides](docs/integrations/) · [examples](examples/README.md)
- **Operate it:** [serving](docs/serving/README.md) · [observability](docs/fak/observability.md) · [deployment](docs/fak/deployment-guide.md)
- **Understand it:** [managed cache](docs/explainers/what-is-managed-cache.md) · [agent integration architecture](docs/fak/agent-integration-architecture.md) · [concepts and story](docs/concepts-and-story.md)
- **Verify it:** [benchmarks](BENCHMARK-AUTHORITY.md) · [claims ledger](CLAIMS.md) · [reproduction packet](docs/repro-packet.md)
- **Build it:** [contributing](CONTRIBUTING.md) · [security](SECURITY.md) · [full documentation index](docs/index.md)

Apache-2.0. Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).