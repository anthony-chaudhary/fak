<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — the Fused Agent Kernel

**Manage the AI agent you already use.**

fak is one Go binary between an agent and its model/tools. Keep Claude Code, Codex, OpenCode, or your own client—and its UI, login, and model—while fak reuses stable work, compacts long sessions, enforces tool policy, journals activity, and recovers interrupted runs. The same native modules can also compose fleets of small, bounded agents in one process.

> **Start here:** install fak, then run `fak manage claude` (or `fak m claude`). No API key or new agent framework is required.

[![Watch the 40-second fak value walk: save tokens and turns first, then see pre-execution policy](visuals/fak-homepage-hero.gif)](visuals/fak-homepage-hero.mp4)

<p align="center"><strong><a href="visuals/fak-homepage-hero.mp4">Watch the 20-second token-savings animation (MP4)</a></strong> · <a href="visuals/fak-hero-values.mp4">separate policy story</a> · <a href="tools/videogen/projects/token-savings/">reproduce the render</a></p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Go Reference](https://pkg.go.dev/badge/github.com/anthony-chaudhary/fak.svg)](https://pkg.go.dev/github.com/anthony-chaudhary/fak) [![Release](https://img.shields.io/github/v/release/anthony-chaudhary/fak?color=blue&label=release&sort=semver)](https://github.com/anthony-chaudhary/fak/releases/latest) [![Go 1.26+](https://img.shields.io/badge/Go-1.26%2B-00ADD8.svg)](go.mod) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/anthony-chaudhary/fak)

<!-- readme-verified: 2026-08-09 vs VERSION 0.43.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme -->

**Current focus: spend fewer tokens and turns.** See the [performance-first capability map](docs/CAPABILITIES.md) for the shipped turn-tax controls, stable-prefix reuse, managed context, per-call model routing, cache-value accounting, and out-of-band session controls. The security floor remains shipped and indexed, but it supports this efficiency story rather than leading it.

## Install and run

macOS or Linux (the installer verifies the release SHA-256):

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak manage claude                 # short form: fak m claude
```

With Go 1.26+, including Windows:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
fak manage --provider openai -- codex    # provider is normally auto-detected
```

`guard` starts a private gateway, points the child agent at it, and passes its model wire and credentials through unchanged. Add a capability floor with `--policy examples/dev-agent-policy.json`; opt into API billing with `--api-key-env`; or use local inference with `--local` or `--gguf`. On exit, fak reports reuse, context, and policy decisions.

See [INSTALL.md](INSTALL.md) for manual downloads, containers, source builds, and release verification, or [first-run troubleshooting](docs/adoption/troubleshooting-first-run.md) if the command refuses to start.

## What changes at the boundary

The largest avoidable cost is often not one token string but an entire extra model round trip: the resident prefix is processed again, latency is paid again, and the result enlarges later context. fak can measure that **turn tax**, complete kernel-known work without another model call, and let an operator budget or steer a live session without prompting the model to manage itself.

| Outcome | How fak provides it |
|---|---|
| **Less model work** | Stabilizes prefixes for provider caches, sheds stale history, deduplicates tool calls, and reuses the live KV cache when it owns inference. |
| **Longer, recoverable runs** | Compacts aged turns to a resident budget, checkpoints state, and resumes interrupted work. |
| **Tool control prompts cannot override** | Returns `ALLOW`, `DENY`, `TRANSFORM`, or `REQUIRE_WITNESS` before execution; untrusted results can be quarantined before reaching the model. |
| **One observable boundary** | Journals model traffic, reuse, compaction, policy, and recovery together. |
| **Backend choice** | Preserves cloud providers, fronts OpenAI-compatible servers, or loads GGUF weights while exposing OpenAI, Anthropic, and MCP interfaces. |

These are measured features. On the current 50-turn × 5-agent Qwen2.5-1.5B Q8 workload, fak did about **4.1× less compute work** than a tuned baseline that already kept each agent's KV cache warm (about 19 versus 78 minutes on an Apple M3 Pro). Real `fak guard -- claude` sessions compacted up to about 107K aged tokens per fire while holding resident context to 48K. Local policy verdicts measured 362 ns without argument predicates and 560–605 ns with them. Scope, hardware, units, and tuned baselines are in [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) and the tagged [claims ledger](CLAIMS.md).

<p align="center">
  <img src="visuals/75-token-savings-frontdoor.svg" alt="Measured token economics for fak guard sessions versus a tuned warm-cache baseline" width="900">
</p>

## Pick an operating shape

| Goal | Command | Details |
|---|---|---|
| Manage one existing agent | `fak manage claude` (or `fak m claude`) | [Claude](docs/integrations/claude.md), [Codex](docs/integrations/openai-codex.md), [all hosts](docs/supported/README.md), [policy](POLICY.md) |
| Expose a shared endpoint | `fak serve --base-url http://localhost:11434/v1 --model qwen2.5:1.5b` | OpenAI, Anthropic, or MCP in front of an existing server; use `--gguf FILE` to stand alone. [Server quickstart](docs/fak/server-quickstart.md) |
| Prove behavior offline | `fak agent --offline` | Deterministically demonstrates repair, reuse, policy denial, and result quarantine; writes `agent-report.json`. [Walkthrough](GETTING-STARTED.md#2-tier-0--try-the-kernel-zero-downloads-2-min) |
| Run many bounded agents | `go run ./cmd/microfleetdemo -selfcheck` | Proves 24 agents in one process, four resident at once, with shared base/model state, fair scheduling, hibernation, policy, and egress control. [Concept](docs/concepts/micro-agents.md) · [measured demo](cmd/microfleetdemo/README.md) |
| Use a Mac's local model from Claude Code | `fak mac` | Targets an existing `fak serve` gateway without replacing Claude's UI. [Setup and expectations](docs/fak/mac-agent-ui.md) |

```bash
fak agent --offline
# -> task completed (booked) YES / YES · poisoned result blocked YES · destructive op prevented YES
```

```text
Claude Code / Codex / OpenCode / your client
                    │
                    ▼
        fak: cache · context · policy · journal · recovery
                    │
                    ▼
       Anthropic / OpenAI / local server / GGUF
```

Every model request and proposed tool call crosses this checkpoint, so reuse happens before another inference, compaction before another context window is consumed, and policy before a tool runs. fak manages the boundaries without becoming the agent or model.

After trying `guard`, `fak launch install --provider all --default claude` can install reversible provider shims so bare `claude`, `codex`, or `fak` uses the managed path. It never overwrites the original binaries; bypass it with `--fak-direct`, `FAK_DIRECT=1`, or `fak launch disable`. See the [zero-adoption guide](docs/zero-adoption-launch.md).

## Next steps

- **Use:** [complete first run](GETTING-STARTED.md) · [tutorial](docs/fak/tutorial.md) · [examples](examples/README.md) · [showcase](docs/showcase.html) · [integration guides](docs/integrations/)
- **Operate:** [serving](docs/serving/README.md) · [configuration](docs/fak/configuration.md) · [observability](docs/fak/observability.md) · [deployment](docs/fak/deployment-guide.md)
- **Understand:** [performance](docs/performance.md) · [architecture](docs/architecture.md) · [concepts](docs/concepts-and-story.md) · [glossary](docs/glossary.md)
- **Verify:** [benchmark methodology](docs/benchmark-methodology.md) · [reproduction packet](docs/repro-packet.md) · [claims](CLAIMS.md)
- **Contribute:** [contributing guide](CONTRIBUTING.md) · [security policy](SECURITY.md) · [documentation map](docs/index.md) · [front-page overflow](docs/README-legacy.md)

Apache-2.0. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md); participation is governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md).


