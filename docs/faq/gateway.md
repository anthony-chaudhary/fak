---
title: "fak FAQ — Inside fak serve (the gateway)"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# Inside fak serve (the gateway)

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

`cachemeta` is a payload-free metadata contract that names reusable objects and their validity, security, residency, and coherence metadata, plus typed lookup verdicts (Hit, Miss, Revalidate, Transform, Quarantine, Fault); it stores no payloads and owns no cache. A `KVPrefix` lowers to a position-prefix-aligned entry, radixkv nodes lower into it, and its attention-index metadata points at the K/V span whose eviction must invalidate a sparse-attention index. Its `kvtransfer` events (offload, restore, route, migrate) carry typed outcomes so a failed restore is never a silent recompute. The metadata contract itself is shipped and tested; the live external serving engine that would consume the cross-instance residency and invalidation directives is out of tree, which is why this layer is a contract rather than a running multi-node KV pool.

## Inside fak serve (the gateway)

How the gateway speaks three wire protocols on one port, fronts an upstream model, adjudicates every proposed call, and re-emits a well-formed response with a decision record attached.

## What does `fak serve` actually do?

`fak serve` fronts the kernel over HTTP, exposing three wire surfaces plus MCP on one port so an agent passes every proposed tool call through the capability floor without an agent-side code change. One `http.ServeMux` serves the OpenAI-compatible routes (`/v1/chat/completions`, `/v1/embeddings`, `/v1/moderations`, `/v1/models`), the native Anthropic Messages route (`/v1/messages`), the fak-native verbs under `/v1/fak/`, and `/mcp`. It defaults to `--addr 127.0.0.1:8080`; `--stdio` swaps HTTP for MCP-over-stdio. The gateway adjudicates a whole turn — it does not execute your tools; your own agent loop runs the calls that survive.

## What are the three wire surfaces `fak serve` exposes?

`fak serve` speaks three protocol-compatible wire surfaces on one port: the OpenAI-compatible surface, the native Anthropic Messages surface, and the fak-native `/v1/fak/` surface, with MCP available over `/mcp` or `--stdio`. The OpenAI surface covers `/v1/chat/completions`, `/v1/embeddings`, `/v1/moderations`, and `/v1/models`. The Anthropic surface covers `/v1/messages` and `/v1/messages/count_tokens` — the Claude-Code-facing wire. The fak-native surface is one POST, one verdict per endpoint: `/v1/fak/adjudicate` (verdict only), `/v1/fak/syscall` (adjudicate and execute), `/v1/fak/admit` (result-side screen), plus feeds, journal, revoke, and policy-reload routes.

## Why does pointing Claude Code at `http://127.0.0.1:8080/v1` give a 404?

Anthropic SDKs append `/v1` themselves, so an Anthropic base URL ending in `/v1` becomes `/v1/v1/messages` and 404s — point Anthropic-wire clients at the origin `http://127.0.0.1:8080` with no `/v1`. This is the single most common wiring mistake. OpenAI clients are the opposite: they do include `/v1`, so an OpenAI base URL is `http://127.0.0.1:8080/v1`. The same origin-vs-`/v1` split applies to `langchain-anthropic` and any other Anthropic-wire client. For Claude Code, set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`.

## How does the gateway decide whether to proxy an upstream, run the in-kernel model, or mock?

The gateway picks its planner backend by a fixed precedence: `--base-url` set means a live proxy in front of your upstream provider; otherwise `--gguf` (with no `--base-url`) loads the in-kernel model and decodes locally; otherwise it falls back to a deterministic scripted mock with a loud boot warning. The `--provider` flag (`openai`, `anthropic`, `gemini`, `xai`) selects the upstream wire when proxying. You can confirm which backend is live: `/healthz` reports the `planner` field as `mock`, `proxy`, `inkernel`, or `unknown`. The in-kernel path is a correctness reference, not a production serving engine — prefer fronting a real token engine for scale.

## How do I put `fak serve` in front of an existing upstream model?

Pass `--base-url URL` (and `--provider`) to make `/v1/chat/completions` and `/v1/messages` a live adjudicating proxy in front of your upstream provider, with `--api-key-env VAR` naming the environment variable that holds the upstream bearer token. The flag names the env var, never the literal key value — fak reads the secret from the environment and forwards it upstream. With `--base-url` empty, the gateway runs offline against the scripted mock instead. The request model name passes through to the upstream verbatim, so your existing prompts and tool definitions stay unchanged.

```bash
fak serve --addr 127.0.0.1:8080 --provider openai --base-url https://api.openai.com/v1 --model gpt-4o --api-key-env OPENAI_API_KEY --policy floor.json
```

## What happens if the upstream `--base-url` is down or unreachable?

If the upstream cannot be reached — dial refused, DNS failure, or a TLS error — the gateway returns a 502 with the distinct code `upstream_unreachable` and a message telling you to check that `--base-url` points at a running server. An upstream 4xx is surfaced with that same status (an unknown model becomes 404, a bad argument 400); an upstream 5xx, transport error, or unparseable body maps to a generic 502. The raw provider body never crosses the trust boundary back to your client. If the upstream announces tool calls but none parse, the gateway fails closed with a 502 rather than serving a malformed turn.

## Does `fak serve` stream responses, and is the stream adjudicated before it reaches me?

`fak serve` streams well-formed SSE, but it buffers the entire upstream turn first, adjudicates the complete proposed tool-call set, and only then synthesizes the stream — so raw upstream deltas never pass through before adjudication. The planner itself is non-streaming. On the OpenAI wire it emits an opening role chunk, the surviving tool-call chunk, content fragments split on word boundaries that reconcatenate byte-exact, a final chunk carrying `finish_reason`, `usage`, and the `fak` extension, then `data: [DONE]`. On the Anthropic wire it emits the `message_start` through `message_stop` block sequence with a real `stop_reason` and token counts, sending a keepalive ping every 15 seconds while the upstream is in flight.

## What is the `fak` response extension on a gateway reply?

The `fak` extension is a top-level object on `/v1/chat/completions` and `/v1/messages` responses that reports every adjudication the kernel made on that turn; it is omitted entirely on a turn with no tool activity. It carries `adjudications[]` — one entry per proposed call including dropped ones, with `repaired_arguments` present only on a TRANSFORM verdict — and `result_admissions[]`, one entry per inbound tool result the kernel screened. Each verdict is a `WireVerdict` with `kind`, `reason`, `by`, `disposition`, and `detail`. A result QUARANTINE overrides an otherwise-ALLOW submit, so the extension is where a fak-aware client learns a call was repaired, dropped, or held.

## Does Claude Code see the `fak` extension, or do I lose the verdicts on the Anthropic wire?

Claude Code reads content blocks but not the `fak` extension key, so on the `/v1/messages` wire any drop, repair, or quarantine is also prepended as a leading `[fak] …` text block in the response. The structured `fak` extension is still emitted for fak-aware clients; the text block is a parallel surface so a client that only parses content still sees what the kernel did. This is built specifically for Claude Code on the native Anthropic wire — point it at the origin `http://127.0.0.1:8080`, and a denied or repaired call shows up in the visible text rather than silently vanishing.

## What does the gateway return to my client when policy denies a tool call?

A policy refusal is a successful HTTP 200 carried as a verdict value, never a non-2xx error — the gateway reserves error statuses for malformed requests, auth failures, and upstream faults. On the served path the gateway keeps ALLOW and TRANSFORM calls and drops the rest; if no tool call survives, `finish_reason` becomes `stop` as a wire end-of-turn so clients do not wait for a missing tool block, and a `denySummary` is written in-band so fak-unaware clients still see what happened. That wire finish reason is not a managed session stop: per-tool refusal feedback continues by default, and a session stop only comes from a declared stop policy. The full verdict for every proposed call, including the dropped ones, lands in the response body's `fak` extension. So your client never treats "the kernel said no" as an exception.

## Is there intelligent request routing or tiered serving inside the gateway?

A tier-selection router exists in the codebase as a library, but it is not wired into the live serving path — the running gateway is single-tier, serving every request from the one engine named by its config. The router code implements size, latency, cost, and hybrid strategies with a health-aware fallback chain, and is explicitly additive: it touches no existing request path. It appears only in its own file and tests, never in a handler or the CLI. So treat tiered routing as a built-but-unwired library, not a feature of `fak serve` today.

## How do I reload the capability policy without restarting `fak serve`?

POST to `/v1/fak/policy/reload` with no body to reload the manifest in place at runtime, returning `{reloaded, source, summary}`. The reload is replace-not-merge: the floor is replaced from source, not layered on top of the old one. The loader is injected by the host CLI (wired from `--policy`), so the gateway itself stays policy-schema blind. The route returns 404 if the deployment was not configured for reload, and 400 if the reload itself fails, with the error message included. A reloaded manifest that fails to parse never silently falls back to a more permissive default — it fails loud.

## What is the difference between `/v1/fak/adjudicate` and `/v1/fak/syscall`?

`/v1/fak/adjudicate` returns a pre-execution verdict only, while `/v1/fak/syscall` adjudicates and then executes the call through the kernel. The adjudicate route runs `k.Decide` and returns `repaired_arguments` only on a TRANSFORM verdict — it is the production path for a client that wants the verdict before running the tool itself. The syscall route runs `k.Syscall`, the adjudicate-and-dispatch path. A companion route, `/v1/fak/admit`, runs the result-side floor (`k.AdmitResult`) to screen a result you already executed before it enters context. The fak-native body key is `arguments`, not `args`; unknown keys are silently dropped.

## How does the gateway screen tool results coming back from my client?

When a request carries `role:"tool"` results, the gateway runs each one through the result-side floor before it reaches the model, and reports the outcome in `result_admissions[]`. On a QUARANTINE or TRANSFORM verdict it forwards the paged-out envelope content, so poisoned bytes never reach the model; a result it cannot admit is held out fail-closed with a `{"_quarantined":true,…,"reason":"ADMIT_ERROR"}` stub and a TERMINAL verdict. A quarantine also invalidates the matching upstream KV span. The detector behind this screen is roughly 100% evadable by design — the load-bearing protection is the quarantine policy that holds bytes out of context, not the detector that flagged them.

## Does the gateway require an API key, and how does auth work once enabled?

Auth is off by default for loopback use; turn it on with `--require-key-env VAR`, after which every route except `/healthz` requires the secret held in that environment variable. The flag names the env var, not the literal key. The gateway accepts the secret as `Authorization: Bearer <tok>` or as `x-api-key: <tok>` (for Anthropic-wire clients) against one secret, compared in constant time over SHA-256 digests so it leaks neither bytes nor length. A bare `Authorization` value with no `Bearer ` prefix is rejected; an invalid or missing key returns 401. If the named env var is set but empty, the gateway refuses to start.

## Can the same gateway serve OpenAI clients and Anthropic clients at once?

Yes — one `fak serve` process serves both the OpenAI-compatible `/v1/chat/completions` and the native Anthropic `/v1/messages` on the same port, and both share the same kernel boundary. Internally both routes call the same planner via one `s.complete` path and pass each proposed tool call through the same `adjudicateProposed` boundary; only the downstream wire format differs. The catch is the base-URL convention: OpenAI clients point at `http://127.0.0.1:8080/v1`, Anthropic clients at the origin `http://127.0.0.1:8080` because their SDKs append `/v1` themselves.

## Is `fak serve` also an MCP server, and what tools does it expose?

Yes — `fak serve` is an MCP server over HTTP at `/mcp` and over stdio with `--stdio`, both serving the same JSON-RPC 2.0 dispatch. The stdio transport has no listener and no auth surface. It negotiates protocol versions `2024-11-05`, `2025-03-26`, and `2025-06-18`, falling back to the first, and reports `serverInfo.name` as `fak-gateway`. It exposes the tools `fak_adjudicate`, `fak_syscall`, `fak_admit`, `fak_changes`, `fak_revoke`, and `fak_context_change`. A DENY is a valid tool result with `isError:false`; only genuine protocol faults become JSON-RPC errors.

## When does the Anthropic wire forward my request bytes untouched to the real Anthropic API?

