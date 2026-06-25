---
title: "fak documentation index for operators and integrators"
description: "Navigation hub for the fak serve operator and integrator docs: install, run the gateway, author policy, integrate agents, and deploy to production."
---

# fak documentation index

The `docs/fak/` directory holds the **operator and integrator** docs for `fak serve` (the
gateway) and for putting `fak` in front of a model. The conceptual docs (the two flips, the
scaling laws, the explainers) live one level up in [`docs/`](../) and at the
[repo root](https://github.com/anthony-chaudhary/fak/blob/main/README.md).

![The getting-started journey across the four tutorial parts](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/52-getting-started-journey.png)

## Start here

| If you want to… | Read |
|---|---|
| **Run `fak` for the first time** (guided, real output at every step) | [**tutorial.md**](tutorial.md) ⭐ |
| **Learn every concept in prerequisite order** (a course you can join at any level) | [`LEARNING-PATH.md`](https://github.com/anthony-chaudhary/fak/blob/main/LEARNING-PATH.md) ⭐ |
| Install the binary (Docker / prebuilt / source) | [`INSTALL.md`](https://github.com/anthony-chaudhary/fak/blob/main/INSTALL.md) · [`fak/GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md) |
| Just chat with a local model | [Simple Demo](https://github.com/anthony-chaudhary/fak/blob/main/cmd/simpledemo/README.md) |
| Quick answers — what it is, how it differs, threat model | [faq.md](faq.md) |

## Run the server

| Topic | Doc |
|---|---|
| Fast path to a running gateway | [server-quickstart.md](server-quickstart.md) |
| Every flag and env var | [server-config.md](server-config.md) |
| Every endpoint, request, and response | [api-reference.md](api-reference.md) |
| When something breaks | [server-troubleshooting.md](server-troubleshooting.md) |
| Metrics, logs, and traces | [observability.md](observability.md) |
| Performance, scaling, multi-region, and HA | [advanced-topics.md](advanced-topics.md) |
| Production deployment | [deployment-guide.md](deployment-guide.md) |
| Always-on dogfood gateway and guarded fleet | [always-on-dogfood-server.md](always-on-dogfood-server.md) |
| Activate the Tier-1 Mac dogfood node | [node-macos-a-activation.md](node-macos-a-activation.md) |
| Stand up the Tier-2 GCP control VM | [gcp-tier2-control-vm.md](gcp-tier2-control-vm.md) |
| Plan guard-hop RSI tuning | [guard-hop-rsi-loop.md](guard-hop-rsi-loop.md) |
| Guard the opencode/GLM dispatch lane | [opencode-glm-guard.md](opencode-glm-guard.md) |

## Author and harden the policy

| Topic | Doc |
|---|---|
| Build a capability floor (worked examples) | [policy-guide.md](policy-guide.md) |
| The manifest schema + refusal vocabulary | [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) |
| Hardening a deployment (auth, network, threat model) | [security.md](security.md) |

## Integrate

| Topic | Doc |
|---|---|
| Architecture of agent ↔ kernel integration | [agent-integration-architecture.md](agent-integration-architecture.md) |
| Put `fak` in front of a framework (LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI, …) | [agent-framework-integration.md](agent-framework-integration.md) |
| Client code in Python, JavaScript, Go, and Rust | [multi-language-examples.md](multi-language-examples.md) |
| Migrate an existing stack (OpenAI API, LangChain, AutoGen, llama.cpp) onto `fak` | [migration-guide.md](migration-guide.md) |
| Claude Code + Anthropic API setup | [`docs/integrations/claude.md`](../integrations/claude.md) |
| OpenAI Codex / OpenAI-compatible clients | [`docs/integrations/openai-codex.md`](../integrations/openai-codex.md) |
| Related work + prior art | [related-items.md](related-items.md) |
| Where the paid layer is heading — hosted multi-tenant policy + audit plane (RFC, not built) | [hosted-control-plane.md](hosted-control-plane.md) |

## Status

The operator/integrator documentation set above is complete — multi-language examples,
framework integration, API reference, and FAQ have all shipped. The per-page status and
any remaining polish is tracked in [documentation-roadmap.md](documentation-roadmap.md).

---

> Every command and output block in [tutorial.md](tutorial.md), [policy-guide.md](policy-guide.md),
> [observability.md](observability.md), and [security.md](security.md) was captured from a
> clean build of `fak` v0.30.0. If a command prints something different for you, that's a
> doc bug — please [open an issue](https://github.com/anthony-chaudhary/fak/issues).
