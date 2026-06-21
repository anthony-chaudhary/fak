# fak Gateway API Reference

A complete reference for every HTTP endpoint exposed by `fak serve` — the kernel
gateway. Three wire surfaces share one port:

- an **OpenAI-compatible** surface (`/v1/chat/completions`, `/v1/embeddings`,
  `/v1/moderations`, `/v1/models`),
- a **native Anthropic Messages** surface (`/v1/messages`,
  `/v1/messages/count_tokens`) — the one Claude Code uses,
- a **fak-native** surface (`/v1/fak/*`) — one POST, one verdict, the simplest
  non-Go integration,

plus **MCP-over-HTTP** (`/mcp`) and the operational endpoints (`/healthz`,
`/metrics`, `/debug/vars`).

This reference is generated from the gateway package source of `fak` v0.30.0
(`fak/internal/gateway/`). Field names and types are taken directly from the wire
DTOs. For the metrics format and the `/debug/vars` snapshot in depth, see
[observability.md](observability.md); for the flags and environment variables that
configure the server, see [server-config.md](server-config.md).

---

## Conventions

### Base URL

The gateway binds the address passed to `fak serve --addr` (default
`127.0.0.1:8080`). All paths below are relative to that origin, e.g.
`http://127.0.0.1:8080/v1/chat/completions`.

> **Anthropic clients append `/v1` themselves.** Point Claude Code at the *origin*,
> not the `/v1` path: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`.

### Authentication

Auth is **off by default** (drop-in, loopback-friendly). When the operator sets a
secret via `--require-key-env <ENV_VAR>` (see
[server-config.md → Authentication](server-config.md#authentication)), **every route
except `/healthz`** requires it. The gateway accepts the secret under either header:

| Scheme | Header | Used by |
|---|---|---|
| Bearer | `Authorization: Bearer <token>` | OpenAI / fak-native / MCP clients |
| API key | `x-api-key: <token>` | Anthropic clients (Claude Code, the Anthropic SDKs) |

The comparison is constant-time over SHA-256 digests, so a reject leaks neither the
secret's bytes nor its length. A missing or invalid credential returns
**`401 Unauthorized`**. A bare `Authorization` value with no `Bearer ` prefix is
rejected (no scheme-stripping leniency).

If you bind a non-loopback address with no key set, the gateway logs a loud startup
warning — a kernel exposed without auth is a security risk.

### Request body limits

Every request body is bounded at **4 MiB** (`MaxBytesReader`). The
`ReadTimeout` (default 30 s) also caps body-delivery *time*. Both are tunable; see
[server-config.md → HTTP Server](server-config.md).

### Trace correlation

On the proxy paths (`/v1/chat/completions`, `/v1/messages`) and the fak-native
endpoints, a request may carry an `X-Trace-Id` header to correlate a session's calls
across the IFC taint ledger, plan-CFI, metrics, and the access log. If omitted, the
gateway mints a fresh non-empty id. The chosen id is echoed back in the response
`X-Trace-Id` header and, on the fak-native endpoints, in the `trace_id` response
field. The same id is what binds a result-side admission (`/v1/fak/admit`) to a later
call-side adjudication on the same session.

### Error envelope

All error responses use an OpenAI-style envelope, which both the OpenAI-compatible and
fak-native clients understand:

```json
{
  "error": {
    "message": "use POST",
    "type": "invalid_request_error",
    "code": null,
    "param": null
  }
}
```

The `type` is derived from the HTTP status class so a client can branch on it:

| Status | `error.type` |
|---|---|
| 401, 403 | `authentication_error` |
| ≥ 500 | `server_error` |
| everything else (400, 404, 405, …) | `invalid_request_error` |

> Note: `/mcp` is the exception — a protocol fault there returns a JSON-RPC 2.0 error
> object, not this envelope (see [MCP-over-HTTP](#mcp-over-http--mcp)).

### A refusal is not an error

The kernel's defining behavior: a **DENY is a successful HTTP response** (`200`),
carried as a value in the `verdict` (deny-as-value). HTTP error statuses are reserved
for malformed requests, auth failures, and upstream/gateway faults — never for a
policy refusal. This is what lets a refusal cost a non-Go agent zero extra model turns.

---

## Endpoint catalog

| Method | Path | Surface | Purpose |
|---|---|---|---|
| POST | [`/v1/chat/completions`](#post-v1chatcompletions) | OpenAI | Adjudicating chat proxy |
| POST | [`/v1/embeddings`](#post-v1embeddings) | OpenAI | Deterministic embeddings |
| POST | [`/v1/moderations`](#post-v1moderations) | OpenAI | Deterministic moderation |
| GET | [`/v1/models`](#get-v1models) | OpenAI | List the served model |
| POST | [`/v1/messages`](#post-v1messages) | Anthropic | Adjudicating Messages proxy |
| POST | [`/v1/messages/count_tokens`](#post-v1messagescount_tokens) | Anthropic | Token-count estimate |
| POST | [`/v1/fak/syscall`](#post-v1faksyscall) | fak-native | Adjudicate **and execute** one tool call |
| POST | [`/v1/fak/adjudicate`](#post-v1fakadjudicate) | fak-native | Pre-execution verdict only |
| POST | [`/v1/fak/admit`](#post-v1fakadmit) | fak-native | Admit a client-executed tool **result** |
| GET·POST | [`/v1/fak/changes`](#getpost-v1fakchanges) | fak-native | Drain the cross-agent change feed |
| POST | [`/v1/fak/revoke`](#post-v1fakrevoke) | fak-native | Refute a world-state witness |
| POST | [`/v1/fak/context/change`](#post-v1fakcontextchange) | fak-native | Tombstone a recall page |
| POST | [`/v1/fak/policy/reload`](#post-v1fakpolicyreload) | fak-native | Hot-reload the policy manifest |
| POST | [`/v1/fak/trace/reset`](#post-v1faktracereset) | fak-native | Clear a session's IFC taint mark |
| POST | [`/mcp`](#mcp-over-http--mcp) | MCP | JSON-RPC 2.0 over a single POST |
| GET | [`/healthz`](#get-healthz) | ops | Liveness (auth-exempt) |
| GET | [`/metrics`](#get-metrics) | ops | Prometheus metrics |
| GET | [`/debug/vars`](#get-debugvars) | ops | JSON diagnostics snapshot |

Unless noted, a request with the wrong method returns **`405 Method Not Allowed`**.
(`/v1/models` and `/healthz` do not enforce the method and answer any verb; `GET` is
the canonical form.)

---

## OpenAI-compatible surface

### `POST /v1/chat/completions`

The adjudication **proxy**. The gateway forwards the chat to the configured model,
then runs every **proposed** `tool_call` through the kernel **before the caller sees
it**: denied calls are dropped, grammar-repaired calls have their arguments rewritten
to the canonical form, and a fak-aware client gets the full per-call adjudication in
the `fak` extension. The gateway **never executes the client's tools** — the client
does.

**Request** (`ChatRequest`) — a drop-in OpenAI chat body. Unknown OpenAI fields
(e.g. `tool_choice`) are accepted and ignored.

| Field | Type | Notes |
|---|---|---|
| `model` | string | Echoed back; the served model is fixed at boot. |
| `messages` | array | Standard OpenAI chat messages (`role`, `content`, `tool_calls`, …). Inbound `role: "tool"` results are run through the result-side floor before the upstream model sees them. |
| `tools` | array | Standard OpenAI tool definitions. Optional. |
| `max_tokens` | int | Forwarded to the model. Omit to use the planner default. |
| `temperature` | number | Forwarded. Optional. |
| `top_p` | number | Forwarded. Optional. |
| `stop` | string \| string[] | Either shape accepted. Optional. |
| `stream` | bool | `true` ⇒ a synthetic SSE stream (see below). |

**Response** (`ChatResponse`) — a standard `chat.completion` object plus the optional
`fak` extension:

| Field | Type | Notes |
|---|---|---|
| `id` | string | `chatcmpl-fak-<nanos>`. |
| `object` | string | `"chat.completion"`. |
| `created` | int | Unix seconds. |
| `model` | string | The served model. |
| `choices` | array | One choice; `message` carries only the **surviving** (adjudicated) `tool_calls`. |
| `usage` | object | OpenAI usage counters from the upstream turn. |
| `fak` | object | Present only when there was tool activity. See [the `fak` extension](#the-fak-response-extension). |

`finish_reason` is set to `tool_calls` when at least one proposed call survives. When
**every** proposed call is refused, it is `stop` and, for a fak-unaware client, a
human-readable summary of the refusals is written into the message `content`.

**Failure modes**

| Status | When |
|---|---|
| `400` | Malformed JSON body. |
| `502` | Upstream model error, or the upstream announced tool calls but **none** parsed (fail-closed: the gateway refuses to skip adjudication on a call the model intended to make). The upstream provider's raw error body never crosses the trust boundary. |

**Streaming.** With `stream: true` the gateway buffers the whole upstream turn,
adjudicates the complete proposed tool-call set, then emits a synthetic
`text/event-stream`: an opening `role` + surviving-`tool_calls` chunk, content
fragments (word-boundary segments that reconcatenate byte-for-byte), a final chunk
carrying `finish_reason` + `usage` + the `fak` extension, and `data: [DONE]`. Raw
upstream deltas are **never** passed through before adjudication.

---

### `POST /v1/embeddings`

OpenAI-compatible embeddings with a **deterministic, self-contained backend** — an
L2-normalized feature-hash projection (the "hashing trick"). No GPU, no weights, no
network: identical text yields an identical vector, and texts sharing tokens score
higher cosine similarity. It is **not** a learned model; it is built for deterministic
tests, semantic-cache keys, and nearest-neighbour smoke checks.

**Request** (`EmbeddingsRequest`)

| Field | Type | Notes |
|---|---|---|
| `input` | string \| string[] \| int[] \| int[][] | Required, non-empty. All four OpenAI shapes accepted (bare string, batch of strings, one pre-tokenized input, batch of pre-tokenized inputs). Max **2048** items per request. |
| `encoding_format` | string | `"float"` (default) or `"base64"` (little-endian float32). |
| `dimensions` | int | Output width, clamped to `[1, 3072]`. Default **256**. |
| `model` / `user` | string | Accepted and ignored (drop-in). |

**Response** (`EmbeddingsResponse`): `object: "list"`, one `data` entry per input in
request order (`{object: "embedding", index, embedding}`), `model`, and
`usage: {prompt_tokens, total_tokens}`. The `embedding` is a JSON number array
(`float`) or a base64 string (`base64`).

`400` on a missing/empty `input`, an unsupported `encoding_format`, or a batch over the
item cap.

---

### `POST /v1/moderations`

OpenAI-compatible moderation with a **deterministic lexical backend** that scans each
input for category keywords. An honest, explainable baseline — **not** a learned safety
model — that runs on-host with no GPU or network.

**Request** (`ModerationsRequest`): `input` (string or string[], required, non-empty,
same 2048-item cap), optional `model` (echoed back).

**Response** (`ModerationsResponse`): `id`, `model`, and one `results` entry per input:

```json
{ "flagged": false, "categories": { … }, "category_scores": { … } }
```

`categories` (bool) and `category_scores` ([0,1]) always carry the **full** OpenAI
category vocabulary, so a client keying on a category never reads a missing field:
`sexual`, `hate`, `harassment`, `self-harm`, `sexual/minors`, `hate/threatening`,
`violence/graphic`, `self-harm/intent`, `self-harm/instructions`,
`harassment/threatening`, `violence`. An input is `flagged` iff any category reaches
the 0.5 threshold.

---

### `GET /v1/models`

Lists the single served model:

```json
{ "object": "list", "data": [ { "id": "<model>", "object": "model", "owned_by": "fak" } ] }
```

---

## Anthropic-compatible surface

### `POST /v1/messages`

The adjudication **proxy** on the Anthropic Messages wire — the Claude-Code-facing twin
of `/v1/chat/completions`. Same planner, same kernel boundary, different downstream
wire. Every tool call the locally-served model proposes is dropped/repaired by the
kernel before Claude Code sees it.

Decode the inbound Anthropic Messages request (`model`, `messages`, `system`, `tools`,
`max_tokens` *(required on this wire)*, `temperature`, `top_p`, `stop_sequences`,
`stream`); inbound `tool_result` blocks are run through the result-side floor first.

**Response** (`anthropicMessageResponse`) — a standard Messages object
(`id`, `type: "message"`, `role: "assistant"`, `model`, `content`, `stop_reason`,
`stop_sequence`, `usage`) plus a top-level **`fak`** extension carrying the same
per-call adjudications as the OpenAI wire. Because Claude Code reads the content blocks
but not the `fak` key, the kernel's drops/repairs/quarantines are **also** prepended as
a short in-band `[fak] …` text block so the agent actually reacts to them.

- `usage` reports `input_tokens`, `output_tokens`, and (omitted when zero)
  `cache_read_input_tokens` / `cache_creation_input_tokens`. On the
  anthropic→anthropic passthrough path the client's `cache_control` prefix is forwarded
  byte-for-byte so a real upstream cache hit reaches the client's accounting.
- `502` on an upstream model error (the raw provider body is not forwarded).

**Streaming.** With `stream: true` the gateway synthesizes a well-formed Anthropic SSE
sequence from the finished, already-adjudicated turn: `message_start`, then a
`content_block_start` / `…_delta` / `content_block_stop` triple per block, a
`message_delta` carrying the real `stop_reason` + token counts, then `message_stop`.
A `ping` event is sent every 15 s while the upstream turn is still in flight. The
`tool_use` ids the client matches results back by are byte-faithful.

### `POST /v1/messages/count_tokens`

Answers with a cheap, tokenizer-free estimate: `{ "input_tokens": <n> }`. Claude Code
treats this as optional (a 404 would be fine), but answering it keeps its
context-management heuristics from flying blind.

---

## fak-native surface

The simplest non-Go integration: one POST, one verdict. Every request body is the
small JSON DTO documented per-endpoint; every response carries a
[`verdict`](#the-verdict-object). `trace_id` is optional on every request and is minted
+ echoed when omitted.

### `POST /v1/fak/syscall`

Adjudicate **and execute** a single tool call through the kernel (the self-contained /
CI path: kernel dispatch to the registered engine + result-side admission).

**Request** (`SyscallRequest`)

| Field | Type | Notes |
|---|---|---|
| `tool` | string | The logical tool name to route through the kernel. |
| `arguments` | object \| string | The tool arguments: a JSON object, **or** a JSON-encoded string (the OpenAI `function.arguments` convention). Never a kernel CAS handle. |
| `read_only` | bool | Optional vDSO hint that the call is read-only/idempotent (enables cross-agent dedup). |
| `witness` | string | Optional external world-state token (git commit / blob hash / lease epoch) the call reads at. Keys the vDSO entry for dedup and binds it for causal revocation. |
| `trace_id` | string | Optional session id (see [Trace correlation](#trace-correlation)). |

**Response** (`SyscallResponse`): `verdict`, `result` (the executed
[`ResultEnvelope`](#the-result-envelope), present only on this execute path),
`trace_id`. `400` on a malformed body or a kernel argument error.

### `POST /v1/fak/adjudicate`

Returns the **pre-execution verdict only** (the production path for a client that runs
its own tools): no dispatch, no engine, no pending state.

Same `SyscallRequest` body. **Response** (`SyscallResponse`): `verdict`, `trace_id`,
and `repaired_arguments` (present **only** when the verdict is `TRANSFORM` — the
canonical arguments the client should run instead). `400` on a malformed body or
argument error.

### `POST /v1/fak/admit`

Runs a **client-produced tool result** through the kernel's result-side stack
(context-MMU quarantine + IFC source-stamp). The served-path complement of
`/v1/fak/adjudicate`: *adjudicate* gates the call **before** the client runs it;
*admit* contains the result **after**. A poisoned/secret-shaped result is paged out
(quarantined) and the session's IFC taint high-water mark is raised before it is
admitted — arming the exfil floor on the path where fak does **not** run the tool.

**Request** (`AdmitRequest`): `tool` (the tool that produced the result — its source
class keys the provenance taint), `result` (object or JSON-encoded string), optional
`witness`, optional `trace_id` (keys the per-trace taint ledger).

**Response** (`SyscallResponse`): `verdict` (a `QUARANTINE` kind means the bytes were
paged out), `result`, `trace_id`.

### `GET·POST /v1/fak/changes`

Drains the cross-agent **"what changed"** feed for events after the client's cursor, so
an agent can re-plan or evict its own cache when another agent changed or refuted shared
data. The only endpoint that accepts **GET or POST**.

- **GET**: `?since=<cursor>` (a non-negative integer; non-numeric ⇒ `400`).
- **POST**: `{ "since": <cursor> }` (`ChangesRequest`).
- `since = 0` (or omitted) returns everything retained.

**Response** (`ChangesResponse`): `events` and the next `cursor`. Each event
(`CoherenceEvent`):

| Field | Type | Notes |
|---|---|---|
| `kind` | string | `"mutation"` or `"revocation"`. |
| `seq` | uint | The shared coherence-bus sequence — this event's cursor. |
| `tool` | string | mutation: the write-shaped tool that completed. |
| `tags` | string[] | mutation: the invalidation scope (resource tags bumped). |
| `witness` | string | revocation: the refuted witness. |
| `evicted` | int | revocation: entries stranded. |
| `world_ver` | uint | Consistency clock at the event. |
| `trust_epoch` | uint | Integrity clock at the event. |

### `POST /v1/fak/revoke`

Triggers a fleet-wide refutation of an external world-state witness found poisoned or
stale: every pooled tier-2 entry admitted under it is causally evicted, future
re-admission under it is refused, and the eviction is broadcast on the change feed.

**Request** (`RevokeRequest`): `{ "witness": "<token>" }` — required, non-empty
(`400` otherwise).
**Response** (`RevokeResponse`): `witness`, `evicted` (local entries stranded),
`trust_epoch` (the post-bump integrity epoch).

### `POST /v1/fak/context/change`

Records a safe, requester-initiated mutation against a persisted recall core image.
Deliberately **negative-only**: today the only accepted mutation is a **tombstone** that
suppresses one persisted recall page from future model-visible context. The core
image's CAS bytes are preserved for audit.

**Request** (`ContextChangeRequest`)

| Field | Type | Notes |
|---|---|---|
| `image_dir` | string | Path to the persisted recall core image directory. |
| `step` | int | The page step to suppress. |
| `reason` | string | Why the page should be absent from future context. |
| `action` | string | Optional; omit or use `"tombstone"` / `"tombstone_page"`. |
| `digest` | string | Optional CAS digest guard; a mismatch refuses the request. |
| `requested_by` | string | Optional requesting identity. |
| `witness` | string | Optional supporting external witness. |

**Response** (`ContextChangeResponse`): the applied ledger row — `image_dir`, `id`,
`action`, `step`, `digest`, `reason`, `requested_by`, `witness`, `trust_epoch`, plus
`applied` and `tombstoned` booleans. `400` on a malformed body or a rejected mutation.

### `POST /v1/fak/policy/reload`

Hot-reloads the configured policy manifest in-place (no request body). The loader is
injected by the host CLI, so the gateway stays policy-schema blind.

**Response** (`PolicyReloadResponse`): `{ "reloaded": true, "source": "<path>",
"summary": "…" }`.

| Status | When |
|---|---|
| `404` | Policy reload is not configured for this deployment. |
| `400` | The reload failed (the error message is included). |

### `POST /v1/fak/trace/reset`

Clears the per-trace IFC taint high-water mark for a live session — e.g. to start a new
logical task on a reused session without inheriting the prior task's taint. The reset
implementation is injected by the host CLI.

**Request** (`TraceResetRequest`): `{ "trace_id": "<id>" }` — required, non-empty.
**Response** (`TraceResetResponse`): `{ "reset": true, "trace_id": "<id>" }`.

| Status | When |
|---|---|
| `404` | Trace reset is not configured for this deployment. |
| `400` | `trace_id` was empty, or the reset failed. |

---

## MCP-over-HTTP (`/mcp`)

The kernel is exposed as an **MCP server** speaking JSON-RPC 2.0, hand-rolled on the
standard library (the repo is zero-dependency by design). `POST /mcp` serves a single
JSON-RPC request/response; the same dispatch is also available over stdio
(`fak serve --stdio`, newline-delimited frames) with no listener and no auth surface.

A request body is one JSON-RPC message. A **notification** (no `id`, e.g.
`notifications/initialized`) is accepted and returns `202 Accepted` with no body.

**Methods**

| Method | Result |
|---|---|
| `initialize` | Negotiates the protocol version (one of `2024-11-05`, `2025-03-26`, `2025-06-18`; falls back to the first when the client asks for an unsupported revision) and returns `serverInfo: {name: "fak-gateway", version}` with a `tools` capability. |
| `tools/list` | The tool descriptors below, each with a JSON-Schema `inputSchema`. |
| `tools/call` | Routes `{name, arguments}` to one of the tools. |
| `ping` | `{}`. |

**Tools** (the `arguments` object mirrors the matching fak-native request DTO):

| Tool | Maps to | Notes |
|---|---|---|
| `fak_adjudicate` | `/v1/fak/adjudicate` | Pre-execution verdict only. |
| `fak_syscall` | `/v1/fak/syscall` | Adjudicate + execute. |
| `fak_admit` | `/v1/fak/admit` | Admit a client-executed result. |
| `fak_changes` | `/v1/fak/changes` | Drain the change feed (`{since}`). |
| `fak_revoke` | `/v1/fak/revoke` | Refute a witness (`{witness}`, required). |
| `fak_context_change` | `/v1/fak/context/change` | Tombstone a recall page. |

A `tools/call` result wraps the matching fak-native response JSON as a single text
content block, with `isError: false`. **A DENY is a valid result, not an error** —
JSON-RPC errors are reserved for protocol/internal faults:

| Code | Meaning |
|---|---|
| `-32700` | Parse error (unparseable frame). |
| `-32600` | Invalid request (`jsonrpc` ≠ `"2.0"`, or an oversized frame on stdio). |
| `-32601` | Method not found. |
| `-32602` | Invalid params (bad `tools/call` arguments, unknown tool, or a kernel argument error). |
| `-32603` | Internal error. |

For the MCP tool-result wire format in depth, see
[`docs/mcp-tool-result.md`](../../fak/docs/mcp-tool-result.md).

---

## Operational endpoints

### `GET /healthz`

Liveness check. **The only auth-exempt route** (so a load balancer can probe an
authenticated gateway). Returns:

```json
{ "ok": true, "engine": "<engine-id>", "model": "<model>" }
```

### `GET /metrics`

Prometheus exposition format (`text/plain; version=0.0.4`): HTTP request histograms,
kernel operation counters (submits, vDSO hits, denies, transforms, quarantines,
admits), startup-phase gauges, and more. `405` on a non-GET method. The full metric
catalog with captured output is in [observability.md](observability.md).

### `GET /debug/vars`

An expvar-style JSON snapshot for diagnostics: a `gateway` block (version, engine,
model, vDSO, `auth_required`, uptime, in-flight requests), a `runtime` block (Go
version, GOOS/GOARCH, goroutines, a full `memory` breakdown), a `kernel` block (the
same counters as `/metrics`, plus `vdso_hit_ratio`), and a `metrics` block with the
per-route HTTP and per-operation latency histograms. `405` on a non-GET method. The
annotated snapshot is in [observability.md](observability.md).

---

## Shared objects

### The verdict object

Every fak-native response (and each entry in the `fak` extension) carries a
`WireVerdict` — the stable, named projection of the kernel's decision:

| Field | Type | Notes |
|---|---|---|
| `kind` | string | `ALLOW` · `DENY` · `TRANSFORM` · `QUARANTINE` · `REQUIRE_WITNESS` · `DEFER` (an unknown registered kind renders as `KIND_<n>`, never a bare integer). |
| `reason` | string | The closed refusal vocabulary, e.g. `POLICY_BLOCK`. Omitted when there is no reason. See [`fak/POLICY.md`](../../fak/POLICY.md). |
| `by` | string | Which adjudicator decided (forensics). |
| `disposition` | string | The actionable deny-loopback class: `RETRYABLE` · `WAIT` · `ESCALATE` · `TERMINAL`. Present on a refusal; this is what lets a refusal cost a non-Go agent zero extra model turns. |
| `detail` | object | Bounded disclosure — e.g. `{"claim": "<offending claim/glob>"}`. The deny channel is **not** a policy oracle. |

`REQUIRE_WITNESS` and any non-core restrictive kind carry `disposition: "ESCALATE"`
(route to a witness / human-approval queue). A result quarantined at admit-time
overrides an otherwise-`ALLOW` submit verdict — the `kind` is reported as `QUARANTINE`.

### The result envelope

A tool result rendered for the wire (bytes resolved, never a CAS handle):

```json
{ "status": "OK", "content": "…", "meta": { "…": "…" } }
```

`status` is `OK` · `ERROR` · `PENDING`.

### The `fak` response extension

On the chat-completions and messages proxies, `fak` carries the kernel's decisions for
a turn:

```json
{
  "fak": {
    "adjudications": [
      { "tool_call_id": "…", "tool": "…", "admitted": true,
        "verdict": { "kind": "TRANSFORM", … }, "repaired_arguments": { … } }
    ],
    "result_admissions": [
      { "tool_call_id": "…", "tool": "…", "verdict": { "kind": "QUARANTINE", … } }
    ]
  }
}
```

- `adjudications` — one entry per **proposed** tool call, **including dropped ones**
  (a fak-unaware client simply never sees the dropped `tool_calls` in the message).
  `repaired_arguments` is present only on a `TRANSFORM`.
- `result_admissions` — one entry per **inbound** tool result admitted before it
  reached the upstream model.

The whole `fak` object is omitted on a turn with no tool activity.

---

## See also

- [tutorial.md](tutorial.md) — zero-to-first-call, real captured output at every step.
- [server-quickstart.md](server-quickstart.md) — the fast path to a running gateway.
- [server-config.md](server-config.md) — every flag and environment variable.
- [policy-guide.md](policy-guide.md) — authoring the capability floor the verdicts enforce.
- [observability.md](observability.md) — the `/metrics` and `/debug/vars` formats in depth.
- [security.md](security.md) — hardening a network-reachable gateway.
- [`fak/POLICY.md`](../../fak/POLICY.md) — the policy schema and the full refusal vocabulary.
