# fak - Fused Agent Kernel

<!-- readme-verified: 2026-06-25 vs VERSION 0.33.0. Previous long-form README archived at docs/archive/README-2026-06-25-before-fresh-start.md. -->

## What It Is

`fak` is one Go binary that puts a kernel boundary under an AI agent.

The model can propose a tool call. The kernel decides whether that call exists,
whether its arguments are allowed, whether the result may enter context, and what
gets audited. Treat the model like an untrusted program; treat the tool call like
a syscall.

The most useful path today is simple: run the agent you already use through
`fak guard`, `fak serve`, or the MCP server, then give it a reviewable policy file.
You keep your model, your IDE, and your tools. `fak` adds the structural floor.
Default-deny (nothing runs unless named) is the baseline.

> TL;DR: Put `fak` under the agent you already run. It blocks unlisted tools,
> keeps unsafe tool results out of context, and records verdicts you can audit.

A verdict is plain: `ALLOW`, `DENY`, `TRANSFORM`, or `QUARANTINE`.

## Start Here

No key, no model, no GPU:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
go run ./cmd/fak agent --offline
go run ./cmd/guarddemo -selfcheck
```

Expected shape:

```text
refund_payment -> DENY (POLICY_BLOCK)
search_kb      -> ALLOW
agent --offline -> injection in context: no; destructive op: no; task still completes
guarddemo       -> WITH fak: 0 breaches
```

That is the core proof: the dangerous action is refused by structure before a
model interpretation matters.

## Use It With Your Agent

### Claude Code, OpenCode, Aider-style CLIs

```bash
fak guard -- claude
fak guard --provider openai -- opencode
```

`fak guard` starts the gateway on loopback and injects the base URL into the
child process only. It loads a built-in secure floor, forwards the real upstream
credential, and prints the kernel's decisions when the agent exits. For Claude
Code it can use your logged-in Claude subscription by default; no API key is
required.

See [docs/integrations/claude.md](docs/integrations/claude.md).

### Codex, Cursor, MCP hosts

For current Codex CLI/IDE sessions, use the MCP path first:

```bash
go build -o fak ./cmd/fak
codex mcp add fak -- ./fak serve --stdio --policy examples/dev-agent-policy.json
```

For any MCP host:

```bash
fak serve --stdio --policy examples/dev-agent-policy.json
```

The MCP surface gives an agent five kernel tools:

- `fak_adjudicate` (decide before dispatch): get a verdict for a proposed call.
- `fak_syscall`: run a checked call through the kernel.
- `fak_admit`: screen a result before it enters context.
- `fak_context_change`: notify the kernel that context changed.
- Session reset tools: start clean when the host cooperates.

Use this when your agent should keep its normal model wire but still ask the
kernel for verdicts.

See [docs/integrations/openai-codex.md](docs/integrations/openai-codex.md),
[docs/integrations/cursor.md](docs/integrations/cursor.md), and
[examples/mcp](examples/mcp).

### Any OpenAI-compatible or Anthropic-compatible client

Put `fak serve` in front of the model endpoint:

```bash
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/dev-agent-policy.json
```

Then point the client at `http://127.0.0.1:8080/v1` for OpenAI-compatible traffic,
or at `http://127.0.0.1:8080` for Anthropic Messages traffic. Harden it with
`--require-key-env FAK_TOKEN` and scrape `/metrics`.

See [GETTING-STARTED.md](GETTING-STARTED.md) and
[docs/fak/api-reference.md](docs/fak/api-reference.md).

## Why Now

The agent stack has moved from demos into operations. Coding agents now have
plugins and background agents. They also have MCP servers, prompt caches, long
sessions, and live tool permissions.

Recent public tooling points in the same direction. MCP reliability, auth, and
observability work is active. Claude Code is shipping MCP and sandbox permission
fixes. Security writing has moved toward runtime tool poisoning alongside prompt
wording.

That changes the useful first screen for `fak`. The value is:

- Put a default-deny floor under the tools your agent already has.
- Keep poisoned tool output and secret-shaped results out of model context.
- Preserve a traceable, privacy-conscious audit trail.
- Make prompt-cache and routing decisions explicit enough to test.

Relevant external signals: [Claude Code changelog](https://code.claude.com/docs/en/changelog),
[MCP stateless/auth discussion](https://dev.to/alexmercedcoder/ai-weekly-codex-goes-long-mcp-goes-stateless-584d),
and [MCP tool-poisoning/security analysis](https://www.cybedefend.com/en/blog/mcp-security-tool-poisoning).

## What The Kernel Does

| Surface | What it gives you | Status |
|---|---|---|
| `fak guard` | Drop-in guard around an existing CLI agent | shipped |
| `fak serve` | OpenAI, Anthropic, fak-native HTTP, plus MCP over HTTP/stdio | shipped |
| Policy floor | JSON allow/deny manifest with closed refusal reasons | shipped |
| Result quarantine | Secret, poison, oversize, and pollution results held out of context | shipped |
| Audit/metrics | JSON logs, optional hash-chained journal, Prometheus, `/debug/vars` | shipped |
| Session control | Budgets, reset directives, cooperative MCP reset, live session state | shipped |
| vCache proof tools | Planned and observed provider-cache savings/refutation | shipped as proof/control plane |
| Model routing | Per-aspect routing, ensembles, routebench, gateway seam | shipped spine; deploy with current flags/docs checked |
| In-kernel model | Pure-Go reference model, kernel-owned KV cache, GPU/backend witnesses | correctness/reference path |

Every claim in [CLAIMS.md](CLAIMS.md) carries exactly one tag:
`[SHIPPED]`, `[SIMULATED]`, or `[STUB]`. The lint gate enforces that honesty ledger.

## Security Model

`fak` is the lock around tool execution.

Most agent security tries to recognize bad text. Recognizers help. They are not
the floor. Prompt injection is a text game. Attackers get turns too. `fak` moves
the load-bearing decision to the capability floor: a dangerous tool outside the
allow-list cannot be called, no matter what the model was told.

Two independent gates matter:

- **Call-side gate:** tool names and selected arguments are checked before dispatch.
  A denied call never reaches the tool runner.
- **Result-side gate:** tool output is screened before it enters context. A poisoned
  or secret-bearing result is paged out or quarantined instead of being handed back
  to the model as trusted text.

The capability floor is the guarantee. The detector can miss, and the docs say
so. Irreversible effects are unwired by default. Untrusted bytes have to pass a
gate before they become model context.

Read [POLICY.md](POLICY.md), [docs/fak/security.md](docs/fak/security.md), and
[docs/integrations/agent-memory.md](docs/integrations/agent-memory.md).

## vCache: Provider Cache As A Budget Signal

Provider prompt caches are useful, but they are not memory you control. You cannot
usually ask a provider to evict span X or prove prefix Y is resident. You get
telemetry after the request.

`fak vcache` treats that correctly:

- A cache hit is a realized rebate, never a correctness dependency.
- The full prompt must always be resendable.
- Plans are proven or refuted from arithmetic and observed usage counters.
- Secrets and regulated content are refused before cache economics apply.

Useful commands:

```bash
go run ./cmd/fak vcache status
go run ./cmd/fak vcache prove
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json
```

Current evidence:

- Claude Code prefix probe: **13,141.5 input-token equivalents saved** over four
  sibling turns, **4.73%**, with the first positive request at 4.
- Codex/OpenAI session telemetry: **9,147,340.8 token equivalents saved** over
  68 token-count events, **85.98%**.

Those prove provider prompt-cache economics on those traces. Causality is
intentionally narrower: the traces show realized provider-cache rebates, while
`fak` supplies the accounting and control plane. The product focus is to make
prefix stability, cache routing, and cache accounting explicit enough that `fak`
can eventually cause and verify more of it.

Read [docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md](docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md)
and [experiments/agent-live/VCACHE-CODEX-OPENAI-PROBE-2026-06-25.md](experiments/agent-live/VCACHE-CODEX-OPENAI-PROBE-2026-06-25.md).

## Model Routing And Router Fusion

Most routers pick one model for a whole request. `fak route` routes an aspect
instead. The unit can be the request or one tool call. It can also be a
sub-query, reasoning step, or tagged stage.

An ensemble is a first-class plan. Supported reductions include `vote` and
`best_of`. They also include `first`, `concat`, and scalar `all_reduce`.

Try it offline:

```bash
go run ./cmd/fak route --aspect tool_call --tool write_file --simulate "approve,deny,deny"
go run ./cmd/fak route --aspect step --complexity high
go run ./cmd/fak routebench
```

The router is useful because it sits at the same point as the security floor. A
write-shaped call can route to a guard ensemble. An easy read can route to a
cheap model. A tenant-sensitive payload bound for a remote route is denied by
the residency floor.

Read [docs/model-routing.md](docs/model-routing.md) and
[docs/integrations/litellm.md](docs/integrations/litellm.md).

## Benchmarks, In One Page

The benchmark rule is simple: every number must trace to
[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md).

The numbers worth remembering:

- `guarddemo -selfcheck`: frozen attack traces reproduce zero breaches behind
  `fak`.
- WebVoyager geometry model: 8-worker fleet prefill is **9.7x less work than the
  naive re-prefill floor** and **1.10x less than tuned per-agent KV**. This is
  modeled prefill-token work, separate from wall-clock.
- 50-turn x 5-agent Qwen2.5-1.5B authority row: **4.1x vs tuned warm-cache**.
  Larger numbers are fenced as vs-naive.
- vCache telemetry proofs above are provider-cache accounting proofs, separate
  from serving throughput claims.

Use vLLM or SGLang for raw token serving. Put `fak` on the agent boundary. Use
it for policy and quarantine. Use it for audit, routing, and controlled reuse.

## Install

From source:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

From a clone:

```bash
git clone https://github.com/anthony-chaudhary/fak
cd fak
go build -o fak ./cmd/fak
```

Go 1.26+ is required. With `GOTOOLCHAIN=auto`, Go can fetch the toolchain on first
build. There are no external Go dependencies and no `go.sum`.

Prebuilt archives and container guidance are in [INSTALL.md](INSTALL.md) and
[GETTING-STARTED.md](GETTING-STARTED.md).

## Build And Test

Run from the repository root:

```bash
go build ./cmd/fak
make test-fast
make ci
```

On native Windows, `go build` and `go vet` work normally, but native `go test`
can be blocked by OS Application Control on freshly compiled test binaries. Use
`./test.ps1` under WSL for the full suite on that host.

## Boundaries

- Token serving: use vLLM or SGLang for raw throughput. `fak` is the agent
  kernel around them.
- Prompt injection: classifiers are useful, but policy carries the load.
- Provider prompt caches: provider hits are rebates. Treat cache state as
  telemetry until you control the memory.
- In-kernel model: the shipped path is a correctness/reference witness with real
  tests. Use a tuned serving stack for production throughput.
- Dangerous tools: keep irreversible and exfil-shaped tools off the allow-list.

## Docs Map

| If you want... | Read |
|---|---|
| First real run | [GETTING-STARTED.md](GETTING-STARTED.md) |
| Claude Code / guard path | [docs/integrations/claude.md](docs/integrations/claude.md) |
| Codex | [docs/integrations/openai-codex.md](docs/integrations/openai-codex.md) |
| MCP examples | [examples/mcp](examples/mcp) |
| Policy manifests | [POLICY.md](POLICY.md) |
| CLI verbs | [docs/cli-reference.md](docs/cli-reference.md) |
| Security model | [docs/fak/security.md](docs/fak/security.md) |
| API reference | [docs/fak/api-reference.md](docs/fak/api-reference.md) |
| vCache | [docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md](docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md) |
| Model routing | [docs/model-routing.md](docs/model-routing.md) |
| Benchmark authority | [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) |
| Honesty ledger | [CLAIMS.md](CLAIMS.md) |
| Machine-readable map | [llms.txt](llms.txt) |
| Old README snapshot | [docs/archive/README-2026-06-25-before-fresh-start.md](docs/archive/README-2026-06-25-before-fresh-start.md) |

License: [Apache-2.0](LICENSE).
