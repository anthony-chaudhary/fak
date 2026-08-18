<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — the Fused Agent Kernel

**Make the AI agent you already use faster, safer, and easier to operate.**

> TL;DR: Wrap Claude Code or Codex with `fak guard`. Or build a focused harness on the
> same context and policy boundary, with routing, journals, and recovery included.

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak preflight --tool refund_payment --args "{}"
# → DENY (DEFAULT_DENY): unknown tool, refused before execution
fak guard -- claude
# Also works with: fak guard -- codex
# Windows or any host with Go 1.26+:
go install github.com/anthony-chaudhary/fak/cmd/fak@v0.44.0
```

## Start here

fak is one Go binary that sits between an agent and its models and tools. Keep your current
assistant, login, and preferred model. fak adds a small control boundary.

It reuses shared context and routes each request. Deny-by-default rules check tool calls before
they run. A journal records what happened and helps long-running sessions recover.

Measured result: **4.1× less compute work than a tuned warm-cache baseline.** On a 50-turn ×
5-agent Qwen2.5-1.5B Q8 run, fak took about 19 minutes versus 78 minutes on an Apple M3 Pro. This result is bounded to that workload. It measures exact work elimination, not a universal
speedup or a claim that fak invented prompt caching.

Providers and gateways already cache repeated prompt beginnings. fak keeps that reuse dependable across real agent
turns, where tool results and history usually disturb the prefix. The [benchmark authority](BENCHMARK-AUTHORITY.md) records scope, hardware, units, and the tuned
baseline. The [method](docs/benchmark-methodology.md) links the reproducible evidence.

`fak guard` starts a private gateway and keeps the agent's model wire and credentials unchanged.
It previews actions, records decisions, and can undo captured file changes. No API key, model download, proxy reconfiguration, or policy file is required. It is the
lowest-friction path to safer agent runs. On Windows, use WSL or the Go install shown above.

## One boundary, five jobs

Most agent stacks assemble these concerns as separate proxies, policy engines, context scripts,
and supervisors. fak handles them at the point where every model request and tool call already
passes:

| Job | What changes for you |
|---|---|
| Reuse context | Stable instructions and history remain a cacheable prefix; repeated work can be served locally. A KV cache is simply the model's reusable memory of that prefix. |
| Route models | Choose cloud or local models per request instead of binding an entire session to one provider. |
| Control tools | Enforce a default-deny capability floor before a tool runs; model prose cannot override it. |
| Keep sessions bounded | Compact old turns while preserving the provider prefix cache, so long sessions do not grow without limit. |
| Operate and recover | Journal decisions, inspect behavior, guard unattended work, and restore captured file changes. |

The key design choice is fusion—not another generic LLM proxy. Performance and security share
the same checkpoint. The fast path cannot bypass policy, and policy does not need a second proxy. The result is a
small control plane rather than another agent UI.

## Prove it in 60 seconds

Clone the repository, then run the offline proof. It needs no key, model, network call, or GPU:

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go run ./cmd/fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool refund_payment --args "{}"
# DENY (POLICY_BLOCK): refused by structure; no model in the loop

go run ./cmd/fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool search_kb --args "{}"
# ALLOW: the policy is not a blanket block

go run ./cmd/fak agent --offline
# task completed (booked) YES / YES · poisoned result blocked YES · destructive op prevented YES
```

The [reproduction packet](docs/repro-packet.md) explains each result. For the shortest visual
tour, open the [interactive showcase](docs/showcase.html).

## Choose your path

### Keep your current assistant

Wrap it with `fak guard` first. When you are ready to move model traffic through the kernel, choose an integration guide:
[Claude Code](docs/integrations/claude.md), [Codex](docs/integrations/openai-codex.md), or
[OpenCode](docs/integrations/opencode.md). The [integration index](docs/integrations/README.md)
covers more clients and providers.

```bash
fak serve --backend http://127.0.0.1:8081/v1
# Point the client at http://127.0.0.1:8080/v1
```

`fak serve` does not replace the provider. It adds the shared boundary in front of a cloud or
local OpenAI-compatible backend.

### Build a small harness

Start with a complete, working product instead of assembling an agent loop from primitives:

```bash
go run ./cmd/fak harness init --template minimal --dir my-agent
cd my-agent
go run . -selfcheck
go run .
```

The generated harness includes instructions, tools, and reply behavior. It also includes safety
defaults and a self-check. Then edit the product profile rather than the engine. See
[`pkg/harnesskit`](pkg/harnesskit), the [harness guide](docs/harness-init.md), and the
[worked demos](examples/).

### Run bounded autonomous work

Use `fak agent` for one session and `fak nightrun` for unattended, ledgered work. Use the
separate DOS plugin when you need several workers to arbitrate shared files, dispatch a plan,
and independently witness completion. FAK is the runtime boundary; DOS is the multi-worker
control layer. See [FAK and DOS](docs/architecture.md#fak-and-dos-are-different-layers).

## How the request path works

```text
agent UI / harness
        │
        ▼
  observe → constrain → plan
        │
        ▼
  reuse context · route model · compact history
        │
        ▼
  check tool capability → execute → admit result
        │
        ▼
  journal · measure · recover
```

This is what fak means by **coordination**: context, model, policy, and tool decisions share one
typed path. It does not mean that fak silently takes over your agent or that every workload
needs a fleet. The current boundary and its Core, Enabling, Stewardship, and Peripheral work are
recorded in [project orientation](docs/project-orientation.md). Detailed proofs, operating
envelopes, and secondary capabilities stay in the [documentation index](docs/index.md) so this
front page remains a map rather than a catalog.

## What is proven—and what is not

- **Shipped:** structural tool denial and guarded execution; OpenAI-compatible serving and
  exact-prefix reuse; bounded context; local-model management; journals, recovery, and harness
  primitives.
- **Measured:** benchmark claims above are tied to committed inputs, outputs, hardware notes, and
  a reproducible methodology in [`experiments/session/`](experiments/session/).
- **Not claimed:** a new foundation model, universal speedups for every trace, perfect prompt-
  injection prevention, or a replacement for provider authentication and sandboxing.

The detailed honesty ledger is [`CLAIMS.md`](CLAIMS.md). Security assumptions and reporting are
in [`SECURITY.md`](SECURITY.md). The project is Apache-2.0 licensed.

## Find the next detail

| If you want to… | Start here |
|---|---|
| Understand the design | [Architecture](docs/architecture.md) · [capability map](docs/CAPABILITIES.md) |
| Reproduce performance | [Benchmark authority](BENCHMARK-AUTHORITY.md) · [methodology](docs/benchmark-methodology.md) · [results](experiments/session/) |
| Configure commands | [CLI reference](docs/cli-reference.md) · [serve configuration](docs/serve-config.md) |
| Connect an agent or model | [Integrations](docs/integrations/) · [model routing](docs/model-routing.md) |
| Build on the Go packages | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) |
| Explore the long-form docs | [Documentation index](docs/index.md) · [interactive showcase](docs/showcase.html) |
| Evaluate or contribute | [Start here](START-HERE.md) · [contributing](CONTRIBUTING.md) · [`llms.txt`](llms.txt) |

<!-- readme-verified: 2026-08-18 vs VERSION 0.44.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme -->
