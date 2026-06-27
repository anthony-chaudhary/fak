# fak - Fused Agent Kernel

<!-- readme-verified: 2026-06-27 vs VERSION 0.34.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme. Previous long-form README archived at docs/archive/README-2026-06-25-before-fresh-start.md. -->

**One binary you put in front of the agent you already run — it keeps your key, puts a default-deny floor under every tool call, and makes long sessions cheaper.**

## What It Is

`fak` is one Go binary you put in front of the AI agent you already run: Claude
Code, Codex, Cursor, or any OpenAI / Anthropic / MCP client. You keep your model,
your IDE, and your tools. You point one base URL at `fak`, and it gives you a handle
on the parts of a real agent loop that get expensive or go wrong:

- **Cheaper long sessions.** A 100k-token Claude Code conversation re-sends its whole
  transcript every turn. `fak` sheds the old turns while keeping the provider's
  prompt-cache prefix byte-identical, so the discount survives instead of breaking.
- **The right model per call.** Send an easy read to a cheap model and a write-shaped
  call to a careful one, chosen per tool call rather than per whole request.
- **Fewer wasted turns.** A repeated read served locally, a malformed call repaired in
  place, a dead-end branch refused before the agent spends a turn on it.
- **A trail you can audit.** Every decision is a plain verdict: `ALLOW`, `DENY`,
  `TRANSFORM`, or `QUARANTINE`. It lands in JSON logs, an optional hash-chained
  journal, and Prometheus metrics.

> **fak in one line:** Put `fak` in front of the agent you already run. It makes long sessions
> cheaper, routes each call to the right model, keeps unsafe tool results out of
> context, and records every verdict. One binary, no rewrite, no key to start.

**The 30-second picture, in numbers** — every figure traces to
[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md). On a 50-turn × 5-agent run, `fak`'s
reuse does **~4.1× less work than a tuned warm-cache stack** (and ~60× less than the
naive re-send loop); reuse of the shared prompt prefix climbs to **6.95×** across the
model ladder. The guard is cheap where it counts — the kernel's allow/deny decision is
~362 ns in-process per call (measured, Apple M3 Pro), not a network hop, and on the
gated reusable-CUDA-graph path (`FAK_CUDA_GRAPH=1`) the in-kernel GPU decode holds
~120 tok/s on an RTX 4070 (SmolLM2-135M), at parity with llama.cpp Q8_0.
`fak guard` also reports live prefill vs decode tok/s on `/metrics`, so a slow first
request gets an answer instead of a shrug.

**Pick your path:** wrap the agent you already run in one command
([`fak guard`](#use-it-with-your-agent)), stand up an always-on gateway and point any CLI at
it ([`fak node`](#an-always-on-gateway-fak-node)), put `fak` in front of any OpenAI /
Anthropic / MCP endpoint ([`fak serve`](#any-openai-compatible-or-anthropic-compatible-client)),
or — if a hard security floor is why you're here — jump to
[For security teams](#for-security-teams).

It does this by sitting on the tool-call path as a kernel. The model *proposes* a
call. `fak` decides whether that call exists, whether its arguments are allowed,
whether the result may enter context, and what gets reused. The same boundary that
saves you tokens is also where a dangerous call gets refused. That is why teams who
need a hard security floor reach for it too (see
[For security teams](#for-security-teams)).

```text
agent --> proposed tool call --> fak kernel --> allowed tool / denied call
tool  --> raw result          --> fak kernel --> admitted context / quarantine
```

## Start Here

No key, no model, no GPU. Pick the line that matches how you got `fak`.

**Installed the binary** (`curl ... install.sh | sh`, see [Install](#install))? These run
from the bare binary anywhere — no clone, no Go, no `examples/` dir. They use the
built-in default floor:

```bash
fak preflight --tool refund_payment --args "{}"     # -> DENY  (DEFAULT_DENY): unknown tool, fail-closed
fak preflight --tool search_kb      --args "{}"     # -> ALLOW: a read-shaped name is not blanket-blocked
fak preflight --tool shell_rm_rf    --args "{}"     # -> DENY  (POLICY_BLOCK): refused by structure
fak preflight --tool exfiltrate     --args "{}"     # -> DENY  (SECRET_EXFIL)
fak agent --offline                                 # the injection / destructive-op A/B, fully offline
```

**Cloned the repo** (you have the Go source tree + `examples/`)? Build first, then run the same proof against a
named example floor, where the deny is by *argument value*:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline
```

Either way, the core proof is the same: the dangerous action is refused by structure
before a model interpretation matters.

## Use It With Your Agent

### Claude Code, OpenCode, Aider-style CLIs

The lowest-friction path: wrap the agent you already run in one command — no rewrite,
no key to start.

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

### Long sessions: shed history, keep the cache hit

A long session re-sends its whole transcript every turn, so a 100k-token conversation
gets expensive fast. The same wrap fixes that **on by default** — once a conversation
sprawls past ~48k resident tokens, `fak guard` sheds the old middle while a short
session is left untouched. Tighten it with one flag, or pass `0` to disable:

```bash
fak guard --compact-history-budget 8000 -- claude   # tighter than the ~48k default
```

`fak` drops the old middle turns while copying the provider's cache prefix through
byte-for-byte, so the prompt-cache discount survives instead of breaking. The obvious
fix, summarizing the old turns, rewrites the prompt and busts the cache, so it costs
*more*. On any doubt `fak` forwards the original prompt unchanged, so it never breaks a
turn. It guarantees the prefix is byte-identical, then relays the provider's own
`cache_read` number rather than claiming the hit.

How and why, with the metrics:
[Long sessions: keep the cache hit](docs/explainers/long-sessions-keep-the-cache-hit.md).
Tracking: [#745](https://github.com/anthony-chaudhary/fak/issues/745).

### An always-on gateway: `fak node`

`fak guard` is per-session. When you want one gateway running all the time — on the laptop
in front of you, or one always-on box you connect to from a phone or a second machine —
`fak node` is the durable lifecycle. It installs `fak serve` as a real system service
(macOS launchd, Linux systemd `--user`, Windows Scheduled Task), points a client at it,
and tears it down, with the same five commands whether the node is local or fleet-wide.

```bash
fak node install            # gateway as a system service on this host (loopback by default)
fak node use HOST:PORT       # on a client: record the node + print the export lines
fak node run -- claude       # launch the CLI pointed at the configured node
fak node status              # service state + /healthz for loopback and the node
fak node forget              # disconnect this client
```

The upstream credential lives on the host; clients present only the gateway's bearer key,
never the upstream secret. `--remote` binds beyond loopback and prints connection lines for
a Tailscale-routed setup.

See [docs/fak/node-setup.md](docs/fak/node-setup.md).

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

- Make prompt-cache and routing decisions explicit enough to test — and keep the
  cache discount alive across a long session instead of busting it.
- Preserve a traceable, privacy-conscious audit trail of every tool call.
- Put a default-deny floor under the tools your agent already has.
- Keep poisoned tool output and secret-shaped results out of model context.

Relevant external signals: [Claude Code changelog](https://code.claude.com/docs/en/changelog),
[MCP stateless/auth discussion](https://dev.to/alexmercedcoder/ai-weekly-codex-goes-long-mcp-goes-stateless-584d),
and [MCP tool-poisoning/security analysis](https://www.cybedefend.com/en/blog/mcp-security-tool-poisoning).

## What The Kernel Does

| Surface | What it gives you | Status |
|---|---|---|
| `fak guard` | Drop-in guard around an existing CLI agent | shipped |
| `fak node` | Install/connect an always-on `fak serve` gateway as a system service (launchd/systemd/Scheduled Task) | shipped |
| `fak console` | Native operator/client panes for issues, live sessions, guard artifacts, and guarded agent launch plans | shipped |
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

## For security teams

If a hard capability floor is *why* you're here — not just a nice-to-have — this
section is for you. The same boundary that sheds tokens above is, for your purposes,
the lock around tool execution.

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

A provider's prompt cache is not memory you control. You cannot ask it to evict a span
or prove a prefix is resident. You just get telemetry after the request. So `fak vcache`
treats a cache hit as a realized rebate, never something the answer depends on. It
proves or refutes each saving from the provider's own usage counters.

```bash
./fak vcache status
./fak vcache prove
```

Evidence from two live traces:

- Claude Code prefix probe: **13,141.5 input-token equivalents saved** over four
  sibling turns, **4.73%**.
- Codex/OpenAI session telemetry: **9,147,340.8 token equivalents saved** over
  68 token-count events, **85.98%**.

Those are provider-cache accounting proofs on those traces: `fak` supplies the
accounting and control plane. The design contract, the full command set, and the
causality fences are on the
[vCache page](docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md); the Codex/OpenAI
probe is written up in
[the probe note](experiments/agent-live/VCACHE-CODEX-OPENAI-PROBE-2026-06-25.md).

## Model Routing And Router Fusion

Most routers pick one model for a whole request. `fak route` routes an aspect
instead. The unit can be the request or one tool call. It can also be a
sub-query, reasoning step, or tagged stage.

An ensemble is a first-class plan. Supported reductions include `vote` and
`best_of`. They also include `first`, `concat`, and scalar `all_reduce`.

Try it offline:

```bash
./fak route --aspect tool_call --tool write_file --simulate "approve,deny,deny"
./fak route --aspect step --complexity high
./fak routebench
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
- WebVoyager geometry model: 8-worker fleet prefill is **1.10x less work than tuned
  per-agent KV** (and 9.7x less than the naive re-prefill floor). This is modeled
  prefill-token work, separate from wall-clock.
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
| Always-on gateway (`fak node`) | [docs/fak/node-setup.md](docs/fak/node-setup.md) |
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
