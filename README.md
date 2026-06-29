# fak — the **F**used **A**gent **K**ernel

<!-- readme-verified: 2026-06-28 vs VERSION 0.34.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme. Front-page overflow (Why Now, full use-case table, vCache/routing deep-dives, three-axes) moved to docs/README-legacy.md on 2026-06-28. Previous long-form README archived at docs/archive/README-2026-06-25-before-fresh-start.md. -->

**Make the agent you already run cheaper and faster — without changing your setup. One binary in front of Claude Code, Codex, Cursor, or any OpenAI / Anthropic / MCP client.**

A long agent session burns money on the same problem over and over: a 100k-token
Claude Code conversation re-sends its *whole* transcript every single turn. `fak`
sits in front of the agent you already run and gives you back the parts of the loop
that get expensive — while keeping your model, your IDE, and your tools exactly as
they are. You point one base URL at `fak`; nothing else changes.

**What you get, in numbers** — every figure traces to
[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md):

- **~4.1× less work than a tuned warm-cache stack** on a 50-turn × 5-agent run —
  because `fak` reuses the shared prompt prefix across agents instead of
  re-paying for it. (Reuse of that prefix climbs to **6.95×** across the model
  ladder. Against the naive re-send loop the gap is ~60×, but beating naive is
  easy — the number that matters is the one above, vs a stack that already caches.)
- **~120 tok/s in-kernel GPU decode** on an RTX 4070 (SmolLM2-135M, on the gated
  `FAK_CUDA_GRAPH=1` path) — **at parity with llama.cpp Q8_0**.
- **The cache discount survives a long session.** `fak` sheds old turns while
  keeping the provider's prompt-cache prefix byte-identical, so the rebate holds
  instead of breaking the moment the conversation sprawls.
- **The guard tax is ~362 ns per call** — the kernel's allow/deny decision is
  in-process (measured, Apple M3 Pro), not a network hop.

> **fak in one line:** Put `fak` in front of the agent you already run. It makes
> long sessions cheaper, routes each call to the right model, and — the same
> boundary — keeps unsafe tool results out of context and records every verdict.
> One binary, no rewrite, no key to start.

**Pick your path:** try the 2-minute no-key demo ([Start Here](#start-here)), wrap
the agent you already run in one command
([`fak guard`](#use-it-with-your-agent)), stand up an always-on gateway
([`fak node`](#an-always-on-gateway-fak-node)), put `fak` in front of any OpenAI /
Anthropic / MCP endpoint
([`fak serve`](#any-openai-compatible-or-anthropic-compatible-client)), or — if a
hard security floor is why you're here — jump to
[For security teams](#for-security-teams).

It does this by sitting on the tool-call path as a kernel. The model *proposes* a
call. `fak` decides whether that call exists, whether its arguments are allowed,
whether the result may enter context, and what gets reused. The same boundary that
saves you tokens is where a dangerous call gets refused.

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

This is where most of the cost goes, and where the same wrap pays for itself.
A long session re-sends its whole transcript every turn, so a 100k-token
conversation gets expensive fast. `fak guard` fixes that **on by default** — once
a conversation sprawls past ~48k resident tokens, it sheds the old middle while a
short session is left untouched. Tighten it with one flag, or pass `0` to disable:

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

## Benchmarks, In One Page

The benchmark rule is simple: every number must trace to
[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md).

The numbers worth remembering:

- 50-turn × 5-agent Qwen2.5-1.5B authority row: **4.1× vs tuned warm-cache**.
  Larger numbers are fenced as vs-naive.
- GPU decode on the gated reusable-CUDA-graph path (`FAK_CUDA_GRAPH=1`):
  **~120 tok/s on an RTX 4070** (SmolLM2-135M), at parity with llama.cpp Q8_0.
- WebVoyager geometry model: 8-worker fleet prefill is **1.10× less work than tuned
  per-agent KV** (and 9.7× less than the naive re-prefill floor). This is modeled
  prefill-token work, separate from wall-clock.
- `guarddemo -selfcheck`: frozen attack traces reproduce zero breaches behind
  `fak`.
- vCache provider-cache telemetry proofs are accounting proofs, separate from
  serving throughput — see [docs/README-legacy.md](docs/README-legacy.md#vcache-provider-cache-as-a-budget-signal).

`fak guard` also reports live prefill vs decode tok/s on `/metrics`, so a slow first
request gets an answer instead of a shrug.

Use vLLM or SGLang for raw token serving. Put `fak` on the agent boundary. Use
it for policy and quarantine. Use it for audit, routing, and controlled reuse.

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
| Cross-platform spine | One kernel across the whole deployment substrate (IoT → edge → laptop → hyperscaler) | shipped (docs/explainers/cross-platform-spine.md) |

Every claim in [CLAIMS.md](CLAIMS.md) carries exactly one tag:
`[SHIPPED]`, `[SIMULATED]`, or `[STUB]`. The lint gate enforces that honesty ledger.

## Starter Policy Floors

Each policy floor is a reviewable allow-list you copy, trim, and run `fak preflight`
against to watch the floor bite. Point your agent at one with
`fak guard --policy examples/<file>` (or `fak serve --policy …` for a gateway).

| Domain | Starter floor | The dangerous action it denies |
|---|---|---|
| Coding agent | [`presets/coding-agent-safe.json`](examples/presets/coding-agent-safe.json) | force-push, `git add -A`, out-of-tree writes, destructive shell |
| Customer support | [`customer-support-readonly-policy.json`](examples/customer-support-readonly-policy.json) | `refund_payment`, direct account or email action |
| Infra / DevOps review | [`devops-dryrun-policy.json`](examples/devops-dryrun-policy.json) | `terraform_apply`, exec, delete, production deploy |

The full catalogue — flight booking, trading, clinical/PHI, SQL analyst, and more,
with a witness command per floor — is in [examples/README.md](examples/README.md)
and the [front-page overflow page](docs/README-legacy.md#use-cases-by-domain). Every
refusal cites a closed reason code you can assert on, such as `POLICY_BLOCK`,
`OVERSIZE`, or `SECRET_EXFIL`.

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

## Going Deeper

Narrower-audience and deep-dive material that used to sit on this page now lives
on the [front-page overflow page](docs/README-legacy.md): why the agent stack
needs this now, the full per-domain use-case catalogue, the vCache provider-cache
budget signal, model routing and router fusion, and the three-axes view of the
kernel (scale → depth → deployment substrate).

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
| Front-page overflow (legacy) | [docs/README-legacy.md](docs/README-legacy.md) |
| Machine-readable map | [llms.txt](llms.txt) |
| Old README snapshot | [docs/archive/README-2026-06-25-before-fresh-start.md](docs/archive/README-2026-06-25-before-fresh-start.md) |

License: [Apache-2.0](LICENSE).
