---
title: "The Gateway: the kernel-adjudicated wire in front of your agent"
description: "fak serve is the gateway runtime — one long-lived server that fronts the single in-process fak kernel over MCP and OpenAI- and Anthropic-compatible HTTP, so an agent in any language routes its tool calls through the syscall boundary. Every wire request gets the same verdict the kernel renders in-process (the network seam is not a bypass), and a call that fails adjudication is dropped fail-closed. Names the wires, the responsibilities, and the defaults."
slug: gateway
keywords:
  - agent gateway
  - AI gateway
  - inference gateway
  - fak serve
  - kernel-adjudicated wire
  - tool-call boundary
  - capability floor
  - result quarantine
  - verdict parity
  - fail-closed
date: 2026-07-15
---

# The gateway — the kernel-adjudicated wire in front of your agent

> **TL;DR:** the **gateway** is `fak serve` in its primary role: one long-lived
> server that puts the single in-process fak kernel on the wire. It fronts that
> kernel over **MCP** (newline-delimited JSON-RPC) and an **OpenAI- and
> Anthropic-compatible HTTP** surface, so an agent written in any language routes
> its tool calls through the syscall boundary without writing Go. It governs
> *model traffic* — routing, cost and context caps, the default-deny capability
> floor, result quarantine, and the audit journal — and it holds two guarantees:
> a verdict over the wire **equals** the verdict the same kernel renders
> in-process, and a call that fails adjudication is **dropped fail-closed**.

**Primary audience:** external builder and operator wiring an agent through fak ·
**Lifecycle:** current · **Generation:** `gen/now` — `fak serve` is the shipped,
witnessed-live path · **Owner:** documentation / gateway · **Proof:**
[`docs/proofs/gateway.md`](../proofs/gateway.md).

**Concept served:** the gateway role of the one binary — the affirmative companion
to the [runtime-vs-client naming spine](runtime-vs-client.md), which answers *which
role do I run*; this page answers *what the gateway is and what it guarantees*.

## What the gateway is

`fak serve` is a **gateway runtime**: a long-lived server that holds one
`*kernel.Kernel` and routes every request that arrives on any of its wires onto
that single kernel's decision. The gateway itself computes nothing numerical —
what it computes is a **routing of each wire request onto the one in-process
kernel's verdict** and a **fail-closed projection of that verdict back onto the
wire**. An agent in Python, JS/TS, Go, or Rust reaches the syscall boundary by
speaking a wire it already knows; it does not link the kernel or write Go.

## What it does on every request

The gateway carries the model call to the upstream provider and adjudicates each
tool call the agent proposes, on the same seam, with two load-bearing guarantees:

1. **The network seam is not a bypass.** A verdict returned over the wire equals
   the verdict the same kernel renders in-process. Untrusted wire bytes are
   re-validated into a tool call, and the gateway **mints its own tainted,
   agent-scoped handle** for that call — a client cannot smuggle a pre-trusted
   handle to skip a rung. (Witnessed: THEOREM 1 in
   [`docs/proofs/gateway.md`](../proofs/gateway.md).)
2. **A failed call is dropped fail-closed.** A tool call that does not adjudicate
   `ALLOW`/`TRANSFORM` is structurally removed from what the client receives — the
   default branch is deny, and a malformed call is dropped with a synthesized
   deny rather than passed through. (Witnessed: THEOREM 2 in the same proof.)

## Responsibilities

| The gateway... | Concretely |
|---|---|
| Fronts the one kernel on the wire | one `*kernel.Kernel`, one decision seam, three wire surfaces |
| Routes model traffic | per-aspect model routing to the configured upstreams |
| Enforces the capability floor | default-deny adjudication of every proposed tool call |
| Quarantines untrusted results | raises the IFC taint mark so a later egress call is gated |
| Caps cost and context | budget and context-window limits on the served session |
| Audits and correlates | the audit journal plus `X-Trace-Id` request correlation |
| Exposes health and metrics | Prometheus `/metrics` for operator observability |

Each row is a responsibility the gateway *holds affirmatively* — it is the seam
that carries it, not a claim about what some other tool lacks.

## The wire surfaces

| Surface | Path | Use it when |
|---|---|---|
| **OpenAI-compatible HTTP** | `/v1/chat/completions` | your client or SDK speaks the OpenAI wire — **the common default** |
| **Anthropic-compatible HTTP** | `/v1/messages` | your harness speaks the Anthropic wire (Claude Code, Anthropic SDKs) |
| **MCP** | newline-delimited JSON-RPC | you want the fak-native tool-call surface directly |
| **fak-native HTTP** | `/v1/fak/syscall`, `/v1/fak/adjudicate` | you call the kernel decision explicitly (execute vs decide-only) |

All four route through the *same* two kernel methods; none contains an independent
decision. Point your existing client at the matching base URL and its tool calls
begin flowing through the syscall boundary.

## One binary, drawn out

```text
  CLIENTS                     fak serve  =  THE GATEWAY                 UPSTREAM
  (any language, any wire)    (one kernel, fail-closed wire)           (tokens)

  ┌──────────────────┐        ┌───────────────────────────────┐
  │ Claude Code      │        │  /v1/messages   (Anthropic)   │
  │ Codex / your SDK │─base──▶ │  /v1/chat/...   (OpenAI)      │──▶ Anthropic / OpenAI
  │ an MCP client    │  URL   │  MCP JSON-RPC                 │    Gemini / xAI /
  └──────────────────┘        │  /v1/fak/{syscall,adjudicate} │    Ollama / vLLM /
                              │   → one *kernel.Kernel        │    SGLang / llama.cpp
                              │   → same verdict as in-proc   │
                              │   → failed call dropped        │
                              └───────────────────────────────┘
```

## Gateway, agent runtime, and client — the line

The gateway governs *model traffic* while your existing harness still owns the
agent loop. That is a different role from the **agent application runtime**
(`fak serve --native`), which owns the loop itself, and from a **client** (a
harness wrapped by `fak manage`, or your own SDK code). This page is about the
gateway; the full three-role distinction and the *which do I run* decision live in
[Two runtimes, one binary](runtime-vs-client.md).

## Choices and defaults

| Choice | Default | Change it with | Scope |
|---|---|---|---|
| Capability policy | the default-deny floor | `--policy <manifest>` | every proposed tool call is adjudicated against it |
| Auth | bearer / `x-api-key` accepted | `--require-key-env <VAR>` to require a key | wire authentication for the served surface |
| Model routing | the configured upstreams, per aspect | routing config | which provider serves each call |
| Loop ownership | your harness owns the loop | add `--native` to own it in fak | gateway runtime vs agent application runtime |

Limitations sit beside the option they constrain: the default-deny floor is the
load-bearing guarantee and is always on; the built-in result detector is
best-effort and evadable, which is *why* the capability floor — not the detector —
is what the gateway relies on. See
[why default-deny beats a classifier](default-deny-vs-classifier.md).

## Generation and support context

`gen/now`. The gateway runtime (`fak serve` / `fak manage`) is the mature, shipped
path, witnessed live in front of Claude Code, Codex, and OpenCode. The agent
application runtime (`fak serve --native`) is the emerging sibling surface; prefer
the gateway path unless you specifically want fak to own the loop. This page is
documentation-only and orthogonal to the runtime implementation gates.

## Next action

Route your existing agent through the gateway with no rewrite: follow the
[Claude Code / Anthropic API guide](../integrations/claude.md) (`fak manage -- claude`
starts a private kernel gateway and points the child at it), or start a shared
`fak serve` and repoint one base URL. Confirm it is live by scraping the gateway's
Prometheus `/metrics`.

## Where to go deeper

- What the kernel does on every call: [The tool call is a syscall](tool-call-is-a-syscall.md).
- Why one static binary carries the whole surface: [One binary is the whole surface](one-binary-one-surface.md).
- Which role to run (gateway vs agent runtime vs client): [Two runtimes, one binary](runtime-vs-client.md).
- How an external agent connects through the entry points, the ABI, and the verdict types: [Agent-integration architecture](../fak/agent-integration-architecture.md).
- Deploy the gateway to production: [Deployment guide](../fak/deployment-guide.md).
- The proof behind the two guarantees: [gateway wire-verdict parity + fail-closed drop](../proofs/gateway.md).
- The vocabulary, one line each: [the fak/DOS glossary](glossary.md).
