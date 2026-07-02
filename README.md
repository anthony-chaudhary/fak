# fak — the **F**used **A**gent **K**ernel

[![ci](https://github.com/anthony-chaudhary/fak/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/anthony-chaudhary/fak/actions/workflows/ci.yml) [![release artifacts](https://github.com/anthony-chaudhary/fak/actions/workflows/release-artifacts.yml/badge.svg?branch=main)](https://github.com/anthony-chaudhary/fak/actions/workflows/release-artifacts.yml)

<!-- readme-verified: 2026-07-01 vs VERSION 0.36.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme. 2026-07-01: front page halved; overflow: docs/README-legacy.md. -->

**fak in one line:** fak is a fused agent kernel. One Go binary sits in front of an agent's
tool calls. It checks each call. It reuses the stable work in long sessions. The same agent
loop comes out safer, cheaper, and faster.

**Put one binary in front of the agent you already run** — Claude Code, Codex, Cursor, or
any OpenAI / Anthropic / MCP client. `fak guard -- claude` wraps your normal agent in one
command: your model, IDE, and keys stay exactly as they are, and `fak` points one base URL
at itself for you.

## Pick your path

[Run your agent through it now](#get-started-with-fak-guard) ·
[run the Colab quickstart](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb) ·
[run a model in the kernel](#run-the-model-in-the-kernel) ·
[the performance story](#the-performance-value-proposition) · [a hard security floor](#for-security-teams).

## What you get, in numbers

Every figure traces to [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md), and the honesty
ledger is [CLAIMS.md](CLAIMS.md):

- **~4.1× less work than a tuned warm-cache stack** on a 50-turn × 5-agent run. `fak` reuses
  the shared prompt prefix — the system prompt + tools, the *KV cache* of the work so far —
  across agents instead of re-paying for it. Reuse climbs to **6.95×** across the model
  ladder (~60× versus a naive re-send loop; the tuned figure is the honest bar).
- **In-kernel GPU decode hits ~120 tok/s** on an RTX 4070 (SmolLM2-135M, f32 weights, gated
  `FAK_CUDA_GRAPH=1`) — inside llama.cpp's Q8_0 band of 120 ± 15 tok/s.
- **The provider cache discount survives a long session.** `fak` sheds old turns while
  keeping the prompt-cache prefix byte-identical, so the rebate holds.
- **The guard tax is ~362 ns per call** — the allow/deny decision runs in-process
  (measured, Apple M3 Pro), no network hop.

## Get started with `fak guard`

The lowest-friction path: wrap the agent you already run in one command. No rewrite, no
config edit, no second terminal.

```bash
fak guard -- claude                                   # your Claude Code, on your Pro/Max subscription — no API key needed
fak guard --api-key-env ANTHROPIC_API_KEY -- claude   # use Anthropic API billing instead
fak guard --provider openai --api-key-env OPENAI_API_KEY -- opencode   # an OpenAI-compatible agent
```

`fak guard` starts a gateway in-process on loopback and injects the base URL into the child
process only. It forwards your real upstream credential
(and the `cache_control` prompt-cache breakpoints) byte-for-byte, so there is no cost
regression. On the same boundary it checks every tool call against a built-in secure
capability floor (a default-deny allow-list), and when the agent exits it prints what the
kernel decided: `fak guard: 131 kernel decision(s) — 121 allowed, 5 denied, 2 repaired,
0 quarantined, 3 deferred`.

For Claude Code, `fak guard` uses your logged-in subscription by default, so no API key is
required. The full walkthrough includes an end-to-end proof that a real `/v1/messages` turn
crossed the gateway: [docs/integrations/claude.md](docs/integrations/claude.md).

### See a real number — no key, no model, no GPU

Installed the binary (see [Install](#install))? These run from the bare binary anywhere. No
clone, no key, no model, no GPU:

```bash
fak routebench                  # -> COST / LATENCY / QUALITY delta vs a one-model baseline
fak benchmarks list --offline   # -> the zero-asset benchmark set
```

`fak routebench` replays a built-in 8-case corpus through a routing policy versus a
single-model baseline and prints `routed is ~20% cheaper, ~10% less total compute, quality
tied` — a deterministic offline lens, and the fastest way to see the kernel do something real.

## Run the model in the kernel

The kernel can *be* the model host too. `fak guard --gguf qwen2.5:7b -- claude` loads a
local GGUF model in-process — no API key, no network, your data never leaves the box — and
the kernel owns the KV cache, so the same reuse and quarantine
machinery applies to a local model as to a proxied one. On the gated reusable-CUDA-graph
path, fak's f32 in-kernel decode reaches parity with a quantized llama.cpp baseline
([head-to-head results](docs/benchmarks/LLAMACPP-HEADTOHEAD-RESULTS.md)).

The honest fence: a small local model is a quality ramp, not a frontier coder — use
`--gguf` for offline or privacy-bound work, and the proxy path for the best reasoning.
Build tags and GPU flags are in the same walkthrough linked above.

## The performance value proposition

A long agent session burns money re-solving the same setup: a 100k-token conversation
re-sends its whole transcript every turn, and a 5-agent fleet pays for the same shared
system prompt five times over. `fak` does the shared work once, two ways:

- **Reuse the shared prefix across agents.** The system prompt, tool table, and instructions
  are identical for every agent in a fleet; `fak` computes that prefix once and reuses it
  (copy-on-write) for all of them — the ~4.1× figure above.
- **Shed history without losing the cache hit.** Past ~48k resident tokens, `fak guard` (on
  by default) drops the old middle turns while copying the provider's cache prefix through
  byte-for-byte, so the prompt-cache discount holds. (Summarizing instead would rewrite the
  prompt and bust the cache.) On any doubt `fak` forwards the original prompt unchanged and
  relays the provider's own `cache_read` number rather than claiming the hit. Tune with
  `fak guard --compact-history-budget <tokens>` (`0` disables).

How and why:
[docs/explainers/long-sessions-keep-the-cache-hit.md](docs/explainers/long-sessions-keep-the-cache-hit.md);
the paying-off trend: [docs/cache-value-rollup.md](docs/cache-value-rollup.md).

## More ways to run it

`fak guard` is per-session and the right default. When you want something else:

- **Always-on gateway** — `fak node` installs `fak serve` as a real system service (macOS
  launchd, Linux systemd `--user`, a Windows Scheduled Task); credentials stay on the host.
  See [docs/fak/node-setup.md](docs/fak/node-setup.md).
- **Codex, Cursor, MCP hosts** — keep your normal model wire and let the agent ask the
  kernel for verdicts over MCP. See
  [docs/integrations/openai-codex.md](docs/integrations/openai-codex.md),
  [docs/integrations/cursor.md](docs/integrations/cursor.md), and [examples/mcp](examples/mcp).
- **Any OpenAI- or Anthropic-compatible client** — put `fak serve` in front of a model
  endpoint and point the client at it: [GETTING-STARTED.md](GETTING-STARTED.md) ·
  [docs/fak/api-reference.md](docs/fak/api-reference.md).

The kernel surface table and benchmark list moved to the
[overflow page](docs/README-legacy.md#what-the-kernel-does). Every claim in
[CLAIMS.md](CLAIMS.md) carries exactly one tag — `[SHIPPED]`, `[SIMULATED]`, or `[STUB]` —
and the lint gate enforces that honesty ledger.

## For security teams

Most agent security tries to recognize bad text. Recognizers help; they are not the floor.
So `fak` moves the load-bearing decision to the **capability floor**: a dangerous tool
outside the allow-list cannot be called, no matter what the model was told. Two independent
gates carry it — a call-side gate (a denied call never reaches the tool runner) and a
result-side gate (poisoned or secret-bearing output is quarantined before it enters
context). The floor is a deployable JSON manifest you copy, trim, and watch bite, no model
in the loop:

```bash
fak preflight --tool refund_payment --args "{}"     # -> DENY (DEFAULT_DENY): not on the allow-list, fail-closed
fak agent --offline                                 # the injection / destructive-op A/B, fully offline
```

Starter floors — coding agent, customer support, DevOps, trading, clinical/PHI, and more —
each name the dangerous action they deny and carry a witness command; point your agent at
one with `fak guard --policy examples/<file>`. The catalogue:
[examples/README.md](examples/README.md) and the
[per-domain table](docs/README-legacy.md#use-cases-by-domain). Every refusal cites a closed
reason code you can assert on (`POLICY_BLOCK`, `SECRET_EXFIL`, …). Read
[POLICY.md](POLICY.md) and [docs/integrations/agent-memory.md](docs/integrations/agent-memory.md).

## Install

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

From a clone: `go build -o fak ./cmd/fak` at the root. Go 1.26+ is required; there are no
external Go dependencies and no `go.sum`. Prebuilt archives and containers:
[INSTALL.md](INSTALL.md).

To build and test: `go build ./cmd/fak`, `make test-fast`, then `make ci` as the green bar.
(On native Windows, run the full test suite under WSL via `./test.ps1`.) The continuous,
path-scoped ship loop: [CONTRIBUTING.md](CONTRIBUTING.md) · [docs/dev-tooling.md](docs/dev-tooling.md).

## Boundaries

- Token serving: use vLLM or SGLang for raw throughput. `fak` is the agent kernel around them.
- Prompt injection: classifiers are useful, but the capability floor carries the load.
- Provider prompt caches: hits are rebates — telemetry until you control the memory.
- In-kernel model: a correctness/reference witness with real tests; use a tuned serving
  stack for production throughput.
- Dangerous tools: keep irreversible and exfil-shaped tools off the allow-list.

## Going deeper

Narrower-audience and deep-dive material lives on the
[front-page overflow page](docs/README-legacy.md): why now, the per-domain use-case
catalogue, vCache, model routing, the moved front-page detail, and the three-axes view
(scale → depth → deployment substrate).

## Docs map

| If you want... | Read |
|---|---|
| First real run | [GETTING-STARTED.md](GETTING-STARTED.md) |
| Claude Code / guard path | [docs/integrations/claude.md](docs/integrations/claude.md) |
| Always-on gateway (`fak node`) | [docs/fak/node-setup.md](docs/fak/node-setup.md) |
| Long sessions / cache | [docs/explainers/long-sessions-keep-the-cache-hit.md](docs/explainers/long-sessions-keep-the-cache-hit.md) |
| Capability floor (policy) | [POLICY.md](POLICY.md) · [examples/README.md](examples/README.md) |
| CLI verbs | [docs/cli-reference.md](docs/cli-reference.md) |
| Security model | [docs/fak/security.md](docs/fak/security.md) |
| Benchmark authority | [BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md) |
| Honesty ledger | [CLAIMS.md](CLAIMS.md) |
| Machine-readable map | [llms.txt](llms.txt) |

License: [Apache-2.0](LICENSE).
