---
title: "Serving-router context-plane defaults: raw context by default, every saver opt-in"
description: "fak serve forwards the client's declared tools, tool results and history byte-identically by default — no oversize paging, no tool-result elision, no restore-tool injection, no per-lever rewriting. The savers are armed by name (flags/env) or by the harness products."
---

# Serving-router context-plane defaults: raw context by default, every saver opt-in

[← Architecture](../architecture.md) · Related: [router tool-guard defaults](router-tool-guard-defaults.md) · [when to use fak guard for harnesses](when-to-use-fak-guard-for-harnesses.md)

## The default is NOTHING (both planes)

`fak serve` is a **raw inference router** on BOTH planes:

- **The tool plane** (the earlier flip, [router tool-guard defaults](router-tool-guard-defaults.md)):
  proposed tool calls are forwarded unadjudicated and inbound tool results reach the
  model byte-identical.
- **The context plane** (this flip, 2026-11-10): the gateway does not REWRITE the
  served context by default. Concretely, with the default posture:
  - **No oversize paging** — a large but benign tool result reaches the model whole;
    no `{"_paged":true}` restore stub is created (`--ctx-paging off`, the default).
  - **No tool-result elision** — an old oversized tool result is not rewritten into a
    `...[fak: tool output elided ... recover via fak_context_restore id=sha256:...]`
    marker, on the chat or the responses wire (`--chat-elide` absent, the default).
  - **No restore-tool injection** — `fak_context_restore` is not auto-advertised into
    the request's tool list. The client's declared tools pass through byte-identical.
    (A history that already carries an elided marker — left by an earlier-armed
    posture or by the client itself — still gets the restore tool advertised, so the
    stashed bytes stay recoverable.)
  - **No per-lever rewriting** — the ctxplan planned view, Anthropic history
    compaction, oversized-result elision, stale-read elision, cache anchoring and
    cold-tool deferral all resolve OFF (`--ctx-view-budget 0`,
    `--compact-history-budget 0`, `--elide-result-bytes 0`, `--elide-stale-reads=false`,
    `--vcache-anchor=false`, `--defer-cold-tools=false`).

A served turn under the default posture is a byte-for-byte pipe: the request the
engine sees is the request the client sent (minus transport framing), and the
response the client sees is the engine's.

## Why

The savers are powerful — and every one of them rewrites what the model sees. A
router that pages, elides, re-anchors or defers by default surprises the operator,
spends hot-path time on transformations the operator never asked for, and makes the
served surface disagree with the wire the client believes it is speaking. The same
reasoning flipped the tool plane earlier the same day; this completes it: **the
router is a dumb pipe unless each plane is armed by name.** Nothing is removed —
every lever keeps its flag/env and its implementation — only WHO turns it on
changed.

## Arming the savers

| Lever | Posture | Re-enable |
| --- | --- | --- |
| oversize tool-result paging | **off** | `--ctx-paging on\|auto` / `FAK_CTX_PAGING` / per-request `X-Fak-Ctx-Paging` header |
| chat/responses tool-result elision + restore-tool injection | **off** | `--chat-elide` / `FAK_CHAT_ELIDE=on` |
| ctxplan planned view | **off** (0) | `--ctx-view-budget <tokens>` |
| Anthropic history compaction | **off** (0) | `--compact-history-budget <tokens>` |
| oversized-result elision | **off** (0) | `--elide-result-bytes <bytes>` |
| stale-read elision | **off** | `--elide-stale-reads` |
| cache anchoring (M2 star-anchor) | **off** | `--vcache-anchor` |
| cold-tool deferral | **off** | `--defer-cold-tools` |

Precedence for the two postures mirrors the tool guard: an explicitly passed flag
wins over the env, an explicit env wins over the flag default, the per-request
`X-Fak-Ctx-Paging` header wins over both (paging), and an unparsable value fails
boot loud — a context rewriter never silently resolves. The resolved posture is
stated in the serve startup messages (`context plane: raw — ...` /
`context plane: armed — ...`).

`fak guard` (the harness product) keeps every saver ARMED: its library config leaves
the postures empty, which resolve to the historical armed defaults, so a
`fak guard -- claude` run is byte-for-byte unchanged by this flip. The compaction
sub-levers (`--compact-anchor-head`, `--assume-session-turns`) also keep their
library defaults on serve: both are inert while the budget is 0, and re-arming the
budget brings the full modern compaction rather than a half-configured one.

## What stays on

- **The native in-kernel compaction shed-line** (`--native-compact-history-budget`,
  derived from the resolved window): that is a capacity lever — shrink-and-serve
  instead of refusing with `context_length_exceeded` when a transcript outgrows the
  resolved window — not a savings rewriter. It stays.
- **The vDSO dedup fast path** (`--vdso`, default on): inert on served turns while
  the tool guard is off (raw mode never enters the probe); it serves the explicit
  `fak_syscall` surface, which is opt-in per call.
- **The MCP tools/list deferral** (`--defer-tools`, default on): that paces the fak
  MCP catalog on `/mcp`, a surface a client connects to deliberately; it does not
  rewrite the chat wires this flip covers.

## Design notes

- **Why the library default stays ARMED.** `gateway.Config{}` with empty
  `CtxPaging`/`ChatToolResultElide` resolves to the historical armed savers so
  `fak guard`, the demos and every existing test keep their behavior byte-for-byte.
  The product default that flipped is the ROUTER's (`fak serve`), via its flag/env
  resolution — the narrow, honest place for a default to change (the same seam the
  tool-guard flip used).
- **The affordance gate is load-bearing.** The restore tool is auto-advertised only
  while an affordance that PRODUCES restore handles is armed (elision, or a live
  paging decision), or when the history already carries a marker. With both postures
  off, injecting a restore tool would advertise a handle-source that can never fire.
- **Raw is raw.** The client harness owns context management in raw mode (it always
  had the option to arm the savers explicitly). The router does not silently keep a
  partial rewriter under a posture named "off".

## Witness

```text
go test ./internal/gateway/ -run 'CtxPaging|ChatToolResultElide|RestoreAffordance|ServePostureConsts|UnparsableContextPlane' -count=1
go test ./cmd/fak/ -run 'ServeContextPlane|ServeSaverFlags|ResolveServeCtxPaging|ResolveServeChatToolResultElide|ElideStaleReads|DeferColdTools|CompactHistoryBudget|TokenDefaults' -count=1
```
