<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — the Fused Agent Kernel

**fak turns a tool-using agent into a managed agent.**

The agent keeps its interface and model. A fak *kernel* — a management plane for one model session, not an OS kernel and not a GPU compute kernel — owns four things from outside it: model traffic and cache reuse, context lifetime, capabilities, and recovery. That pairing is what this page means by a *managed agent*.

> **TL;DR:** install one binary and run `fak agent --offline`. The managed agent still finishes its task while the kernel blocks a poisoned tool result and a destructive operation. No API key, model download, or GPU needed.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Go Reference](https://pkg.go.dev/badge/github.com/anthony-chaudhary/fak.svg)](https://pkg.go.dev/github.com/anthony-chaudhary/fak) [![Release](https://img.shields.io/github/v/release/anthony-chaudhary/fak?color=blue&label=release&sort=semver)](https://github.com/anthony-chaudhary/fak/releases/latest) [![Go 1.26+](https://img.shields.io/badge/Go-1.26%2B-00ADD8.svg)](go.mod) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/anthony-chaudhary/fak)

*This page is the front door, and the only one aimed at a reader who has not decided yet. Every other root page is narrower: [`GETTING-STARTED.md`](GETTING-STARTED.md) installs it and owns the verbatim proof output, [`START-HERE.md`](START-HERE.md) routes a job you already have, [`INDEX.md`](INDEX.md) is the by-name map, [`LEARNING-PATH.md`](LEARNING-PATH.md) teaches the concepts in order, and [`AGENTS.md`](AGENTS.md) is for automated contributors, not humans.*

<!-- readme-verified: 2026-08-03 vs VERSION 0.43.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme -->

## Try the kernel without a key, model, or GPU

For the shortest public proof, install the one binary and run one deterministic end-to-end check:

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
fak agent --offline
# -> task completed (booked) YES / YES · poisoned result blocked YES · destructive op prevented YES
```

Those three verdict rows are the proof: the managed agent still finishes its task while a poisoned tool result and a destructive operation are both stopped at the kernel boundary. Offline mode uses a deterministic mock planner, so it verifies the managed-agent path and the policy boundary without claiming live-model quality or latency.

(Go 1.26+ users can substitute `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`. Run it from any directory — no clone needed.) The run leaves one file behind: it writes `agent-report.json` into the directory you start it from and prints that full path when it finishes. Pass `--out PATH` to put it somewhere else.

That report counts four kernel events beneath those rows: malformed tool calls the kernel repaired in place instead of spending a retry turn (*in-syscall repairs*), repeated calls answered from the kernel's own cache with no engine round-trip (*vDSO dedup hits*), calls the policy refused before they ran (*adjudicator denies*), and tool results held out of the model's context (*MMU quarantines*). The [glossary](docs/glossary.md) defines these and the rest of the project's vocabulary.

The verbatim expected output is printed once among the root pages — in [Getting started § Tier 0](GETTING-STARTED.md#2-tier-0--try-the-kernel-zero-downloads-2-min) — and the unabridged capture is in the [tutorial](docs/fak/tutorial.md). If the command did not do what you expected, start at [first-run troubleshooting](docs/adoption/troubleshooting-first-run.md).

Next: for the expanded policy, routing, and benchmark sequence, use the [reproduction packet](docs/repro-packet.md). To stand up a running, server-side governed agent in under 10 minutes — offline, with a default-deny floor, an audit journal, and a visible DENY — follow the [governed-agent quickstart](docs/fak/governed-agent-quickstart.md).

## One managed agent, two ways to run the kernel

A managed agent pairs fak with Claude Code, Codex, opencode, or your own client and its chosen model. fak owns the operational boundary around that session: what context is retained, what work is reused, which tools may run, what results may return, and how the session recovers. The roles stay separate — the agent or client owns the task loop and user experience, the fak kernel is the management plane, the model provider or server generates tokens. Change any one without replacing the others.

| Where the kernel runs | Start here | Use it when… |
|---|---|---|
| Beside one agent | [`fak guard`](#manage-one-local-agent-fak-guard) | You want a local launcher and supervisor for an existing agent. `guard` starts a private kernel gateway, points the child agent at it, and manages that session. |
| As an endpoint | [`fak serve`](#run-the-managed-agent-endpoint-fak-serve) | One or many OpenAI, Anthropic, or MCP clients need a durable kernel service. It can front an existing model server or load GGUF itself. |
| Entirely offline | [the one-minute proof](#try-the-kernel-without-a-key-model-or-gpu) | You want to evaluate the managed-agent path without an API key, model download, or GPU. |

Both commands run the same kernel, and the model stays a replaceable backend — cloud API, local model server, or in-kernel GGUF. Default to `fak guard` for one existing agent; choose `fak serve` when the kernel must be a shared or durable endpoint. For architecture, current evidence, and development workflow, start at [START-HERE.md](START-HERE.md).

## What the managed agent gains

- **Less repeated work, safely.** fak's core job is to coordinate several caching layers that would otherwise fight each other into one system. It keeps shared prompt prefixes byte-stable so the provider's cache never busts, sheds stale history before it is sent again, and reuses the running scratchpad of work-so-far (the *KV cache*) directly when it owns inference. That is ~4.1× less work than a tuned warm-cache stack — compute time against a baseline that already keeps per-agent KV warm, not against a re-send-everything loop. In the measured run (Apple M3 Pro), a 50-turn × 5-agent Qwen2.5-1.5B Q8 session finishes in ~19 minutes where the tuned baseline needs ~78. It is on by default and does not change the model's output.
- **Long runs keep moving.** Sessions compact their own history — up to ~107K tokens of aged middle-turn context dropped per compaction fire, measured on a real `fak guard -- claude` session held to a 48K resident-context budget — and can resume after a crash instead of dying at the context limit.
- **Policy is on the execution path.** Every proposed tool call receives ALLOW, DENY, TRANSFORM (run it, but rewritten into a safe form), or REQUIRE_WITNESS (hold it until an outside check confirms what it claims to do — and deny if nothing does) against a reviewable *capability floor*: the policy file listing which tools the session may use at all, so anything absent from it is denied before it runs. That verdict is an in-process function call — 362 ns per decision (measured on an M3 Pro; 560–605 ns once argument predicates run), with no policy model and no network hop. Tool results carry provenance too — where the bytes came from, and whether that source is trusted — so poisoned content read from an untrusted source is stopped at *result admission*, before the model acts on it.
- **Big local models are first-class.** The kernel loads GGUF weights itself and serves OpenAI, Anthropic, and MCP clients directly. Mixture-of-experts models use only a few of the model's expert blocks per token; for those, the kernel pages just the needed expert weights in from SSD, so the whole model does not have to sit in RAM. Quantized (e.g. Q4) inference and grammar-constrained structured output — the decoder is held to your schema, so a reply cannot come back as malformed JSON — are built in.

Evidence: [tuned benchmark baselines](BENCHMARK-AUTHORITY.md) · [tagged claims ledger](CLAIMS.md). Every number above names its workload, hardware, and baseline in those two documents. Unfamiliar word: [glossary](docs/glossary.md).

<p align="center">
  <img src="visuals/75-token-savings-frontdoor.svg" alt="Measured token-economics summary for real fak guard sessions, separating provider prompt-cache rebates from fak-authored compaction savings and comparing fak with a tuned warm-cache baseline." width="900">
</p>

## What fak is not

fak is a management plane. It is not a model, not a faster model server, and not a prompt-injection detector. The boundaries, so you can rule it out quickly:

- **Not a faster model server.** vLLM, SGLang, llm-d, and llama.cpp win raw throughput; you run `fak serve` *in front of* one of them. The in-kernel GGUF path makes model state kernel-owned state — it is not a production serving engine. ([serving scope](docs/serving/README.md))
- **The security floor is structural, not detection.** An active policy limits which tools can run, and quarantine holds classified results out of model context. fak does not claim to *detect* a successful prompt injection; it removes the tools and results an injection would need. Allow-listing a destructive tool is policy authoring, not a gate bypass. ([SECURITY.md](SECURITY.md) · [POLICY.md](POLICY.md))
- **Tool-calling turns are gated whole-turn, not token-streamed.** A call cannot be allowed, denied, or repaired until its arguments have fully arrived, so `stream:true` is adjudicated in full and then re-serialized as a well-formed SSE sequence. Prose deltas still stream; tool-call bytes do not. Expect full-turn latency when wiring an interactive harness to the gateway — that is the enforcement model, not a missing feature.
- **Pre-1.0; interfaces still move.** Every capability claim carries exactly one of `[SHIPPED]`, `[SIMULATED]`, or `[STUB]` in [CLAIMS.md](CLAIMS.md), and [product status](docs/PRODUCT-STATUS.md) separates what you can run today from the subsystems that have no command yet. Check a capability's tag before you build on it.

## Manage one local agent: `fak guard`

Start with the agent you already run. No rewrite, config file, API key, or second terminal:

```bash
fak guard -- claude                                  # keep a Claude Pro/Max subscription — no API key
fak guard --provider openai -- codex                 # provider is normally auto-detected
fak guard --policy examples/dev-agent-policy.json -- opencode
```

Your model wire and credentials pass through unchanged. API billing is opt-in with `--api-key-env`; a local model with `--gguf` or `--local`. On exit, fak reports savings and decisions.

Read: [Claude Code walkthrough](docs/integrations/claude.md) · [Codex](docs/integrations/openai-codex.md) · [Cursor](docs/integrations/cursor.md) · [all supported hosts](docs/supported/README.md) · [policy guide](POLICY.md)

## Run the managed-agent endpoint: `fak serve`

Use fak as a shared OpenAI, Anthropic, or MCP endpoint. That one endpoint carries reuse and compaction, routing and observability, authentication and policy.

Front an existing local or remote model server:

```bash
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy examples/dev-agent-policy.json
```

Or stand alone with GGUF weights and no separate model server:

```bash
fak serve --addr 127.0.0.1:8080 \
  --gguf /path/to/model.gguf \
  --model my-local-model
```

Point OpenAI clients at `http://127.0.0.1:8080/v1` and Anthropic clients at the bare host. Use `--stdio` for MCP, or `fak node install` to make it an always-on service. Your inference backend stays in place; fak owns cross-request reuse and the agent/tool boundary.

Read: [server quickstart](docs/fak/server-quickstart.md) · [serving architecture and engines](docs/serving/README.md) · [configuration](docs/fak/server-config.md) · [API reference](docs/fak/api-reference.md) · [deployment](docs/fak/deployment-guide.md)

**Showcase — Claude Code on a Mac's own local model.** A premium cloud agent, open weights on your own silicon, one static binary in between.

`fak mac` is a launcher, not an installer — it needs a Mac already running a local model server behind a `fak serve` gateway. [Stand up the Mac side](docs/fak/mac-agent-ui.md#stand-up-the-mac-side) builds that from scratch, including how to pick a model that fits your Mac's memory; `fak node install` is what makes the gateway survive a reboot. Then launch Claude Code against it:

```bash
export FAK_MAC_GATEWAY="http://<your-mac>:8080"       # the gateway you started above
fak mac                                              # long form: fak claude-mac-fak
```

`fak mac` targets a gateway over the network, so it works whether you are on the Mac or driving it from another machine. Honest expectations: the first full local turn is slow (10–15 min prefill on an M3 Pro) and single-stream. The wow is that it works end to end and stays observable the whole time via the preflight panel. The [operator walkthrough](docs/fak/mac-agent-ui.md) covers the launcher's flags, the preflight panel, and the live overlay.

## Install

No Go toolchain needed — the installer downloads a prebuilt binary and verifies its SHA-256 against the release's `SHA256SUMS`:

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

If you already have Go 1.26+, or you are on Windows without a POSIX shell:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Either way, `fak version` confirms it. Manual archive downloads, containers, build-from-source, and release-provenance verification: [INSTALL.md](INSTALL.md).

## Going deeper

- Use it: [tutorial](docs/fak/tutorial.md) · [integration guides](docs/integrations/) · [examples](examples/README.md) · [first command didn't work?](docs/adoption/troubleshooting-first-run.md)
- Show it off: [the published showcase](https://anthony-chaudhary.github.io/fak/showcase.html) — this repository's configured homepage, built from [docs/showcase.html](docs/showcase.html) · [Claude Code on your own Mac's local model](docs/fak/mac-agent-ui.md) via `fak mac`
- Operate it: [serving](docs/serving/README.md) · [observability](docs/fak/observability.md) · [deployment](docs/fak/deployment-guide.md)
- Understand it: [performance outcomes and proofs](docs/performance.md) · [glossary](docs/glossary.md) · [managed cache](docs/explainers/what-is-managed-cache.md) · [external system architecture](docs/architecture.md) · [concepts and story](docs/concepts-and-story.md)
- Verify it: [benchmark route](docs/benchmark-methodology.md) · [current benchmark authority](BENCHMARK-AUTHORITY.md) · [claims ledger](CLAIMS.md) · [reproduction packet](docs/repro-packet.md)
- Build it: [contributing](CONTRIBUTING.md) (no write access? [fork and open a PR](CONTRIBUTING.md#fork-and-open-a-pull-request)) · [security](SECURITY.md) · [documentation home by audience](docs/index.md) · [front-page overflow](docs/README-legacy.md)

Apache-2.0. Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Participation is governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md).
