---
title: "fak client examples in Python, JS, Go, and Rust"
description: "Runnable client code for calling a fak serve gateway from Python, JavaScript, Go, and Rust across the OpenAI, Anthropic, and fak-native surfaces."
---

# Multi-Language Integration Examples

*For application developers who already have a `fak serve` gateway running and want to call
it from a non-Go codebase. Prerequisite: a reachable gateway (see the
[server quickstart](server-quickstart.md)) and basic familiarity with HTTP in your language.
You will leave able to adjudicate a tool call, read a `verdict`, and wire disposition-aware
retries from Python, JS/TS, Go, or Rust against the OpenAI, Anthropic, or fak-native surface.*

Runnable client code for talking to a `fak serve` gateway from **Python**,
**JavaScript / TypeScript**, **Go**, and **Rust**. Every snippet below targets the
real wire surfaces documented in the [API reference](api-reference.md) and verified
against the gateway source (`fak/internal/gateway/`).

`fak serve` exposes three request surfaces on one port:

- an **OpenAI-compatible** proxy — `POST /v1/chat/completions` (point any OpenAI client at it);
- a **native Anthropic Messages** proxy — `POST /v1/messages` (point Claude Code or the Anthropic SDK at it);
- a **fak-native** surface — `POST /v1/fak/adjudicate` (verdict only) and `POST /v1/fak/syscall`
  (adjudicate **and** execute): one POST, one verdict, the simplest non-Go integration.

For the Claude-Code-specific setup (env vars, the dogfood launcher, policy), see the
[Claude integration guide](../integrations/claude.md). For starting a gateway, see the
[server quickstart](server-quickstart.md).

---

## Three things every example relies on

These hold across all four languages — internalize them once and the snippets read the same.

1. **Base URL.** The gateway binds the address passed to `fak serve --addr` (default
   `http://127.0.0.1:8080`). Anthropic clients append `/v1` themselves, so point them at
   the **origin** (`http://127.0.0.1:8080`), not the `/v1` path. OpenAI clients want the
   `/v1` base (`http://127.0.0.1:8080/v1`).

2. **Auth is off by default.** When the operator sets `--require-key-env <ENV_VAR>`, every
   route except `/healthz` needs the secret, sent under **either** header:

   | Scheme | Header | Used by |
   |---|---|---|
   | Bearer | `Authorization: Bearer <token>` | OpenAI / fak-native / MCP clients |
   | API key | `x-api-key: <token>` | Anthropic clients (Claude Code, the Anthropic SDKs) |

3. **A refusal is not an error.** A `DENY` is a **successful `200` response** carrying a
   `verdict` value (deny-as-value). HTTP error statuses (`400`, `401`, `502`, …) are
   reserved for malformed requests, auth failures, and upstream faults — never for a policy
   refusal. So **always inspect `verdict.kind`**; do not branch on the HTTP status alone.

The `verdict` object every fak-native response (and each entry in a proxy's `fak`
extension) carries:

```json
{
  "kind": "DENY",
  "reason": "POLICY_BLOCK",
  "by": "monitor",
  "disposition": "TERMINAL",
  "detail": { "claim": "rm -rf /tmp/x" }
}
```

- `kind` — `ALLOW` · `DENY` · `TRANSFORM` · `QUARANTINE` · `REQUIRE_WITNESS` · `DEFER`.
- `reason` — the closed refusal vocabulary (e.g. `POLICY_BLOCK`, `DEFAULT_DENY`, `SELF_MODIFY`);
  omitted when there is none. See [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).
- `disposition` — the actionable deny-loopback class on a refusal: `RETRYABLE` · `WAIT` ·
  `ESCALATE` · `TERMINAL`. This is what lets a refusal cost a non-Go agent **zero** extra
  model turns — branch your retry logic on it (see [Retry logic](#retry-logic-disposition-aware)).

---

## Python

```bash
pip install anthropic openai httpx   # only the SDKs you actually use; urllib examples need nothing
```

### 1. Health check + verdict inspection (stdlib only)

The cleanest, most portable integration is the fak-native `/v1/fak/adjudicate` endpoint:
one POST, one verdict, no execution. This uses only the standard library.

```python
import json
import urllib.request

BASE = "http://127.0.0.1:8080"
TOKEN = None  # set when the gateway runs with --require-key-env


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# Health is the only auth-exempt route.
with urllib.request.urlopen(BASE + "/healthz", timeout=5) as r:
    print(json.loads(r.read()))   # {'ok': True, 'engine': 'inkernel', 'model': '...'}

# "Would this tool call be allowed?" — no execution, just the verdict.
resp = _post("/v1/fak/adjudicate", {
    "tool": "Bash",
    "arguments": {"command": "rm -rf /tmp/x"},
})
verdict = resp["verdict"]
print(verdict["kind"], verdict.get("reason"), verdict.get("disposition"))
# DENY POLICY_BLOCK TERMINAL

if verdict["kind"] == "ALLOW":
    run_the_tool_yourself()
```

> **Wire gotcha:** the fak-native key is `arguments`, **not** `args` — an unknown key is
> silently dropped. `arguments` accepts a JSON object *or* a JSON-encoded string (the OpenAI
> `function.arguments` convention).

When the verdict is `TRANSFORM`, the canonical arguments to run instead come back in
`repaired_arguments`:

```python
resp = _post("/v1/fak/adjudicate", {"tool": "Edit", "arguments": {...}})
if resp["verdict"]["kind"] == "TRANSFORM":
    run_the_tool_with(resp["repaired_arguments"])   # grammar-repaired args
```

### 2. Anthropic SDK pointed at fak (Claude Messages proxy)

Point the official Anthropic SDK at the gateway origin. The kernel adjudicates every tool
call the upstream model proposes before the SDK ever sees it.

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8080",   # the origin — the SDK appends /v1
    api_key="fak-local",                # sent as x-api-key; ignored on a no-auth loopback gateway
)

response = client.messages.create(
    model="qwen2.5-coder:7b",           # echoed back; the served model is fixed at boot
    max_tokens=1024,                    # required on the Anthropic wire
    messages=[{"role": "user", "content": "List the files in this directory"}],
    tools=[{
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }],
)
for block in response.content:
    print(block.type, getattr(block, "text", getattr(block, "input", None)))
```

> **Reading the kernel's decisions through an SDK.** The `/v1/messages` response carries a
> top-level `fak` extension, but typed SDK models drop unknown fields. The gateway therefore
> **also** prepends a short in-band `[fak] …` text block to the content so the agent reacts
> to drops/repairs. For programmatic verdict access, call `/v1/fak/adjudicate` directly
> (example 1) or read the raw HTTP response (example 4).

### 3. OpenAI SDK pointed at fak (chat-completions proxy)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",   # OpenAI clients want the /v1 base
    api_key="fak-local",                   # sent as Authorization: Bearer
)

completion = client.chat.completions.create(
    model="qwen2.5:1.5b",
    messages=[{"role": "user", "content": "List the files here"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
)
# Only the surviving (adjudicated) tool calls are present; dropped calls never appear.
msg = completion.choices[0].message
print(completion.choices[0].finish_reason)   # "tool_calls" if any call survived, else "stop"
for call in (msg.tool_calls or []):
    print(call.function.name, call.function.arguments)
```

### 4. Async + streaming with httpx

`httpx` gives both async and raw-response access (so you can read the `fak` extension the
typed SDKs hide). Streaming is supported by the proxy with `"stream": true` — the gateway
buffers the upstream turn, adjudicates the **complete** proposed tool-call set, then emits a
synthetic SSE stream (raw upstream deltas are never passed through before adjudication).

```python
import asyncio
import json
import httpx


async def adjudicate(client: httpx.AsyncClient, tool: str, args: dict) -> dict:
    r = await client.post("/v1/fak/adjudicate", json={"tool": tool, "arguments": args})
    r.raise_for_status()                       # 4xx/5xx are real faults, NOT a DENY
    return r.json()["verdict"]


async def stream_chat(client: httpx.AsyncClient, prompt: str):
    body = {"model": "qwen2.5:1.5b", "stream": True,
            "messages": [{"role": "user", "content": prompt}]}
    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                print(delta["content"], end="", flush=True)
            # The final chunk also carries chunk["fak"] with the adjudications.


async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", timeout=30) as client:
        # Verdicts for several candidate calls, concurrently.
        verdicts = await asyncio.gather(
            adjudicate(client, "Read", {"path": "README.md"}),
            adjudicate(client, "Bash", {"command": "sudo rm -rf /"}),
        )
        print([v["kind"] for v in verdicts])    # ['ALLOW', 'DENY']
        await stream_chat(client, "Say hello")


asyncio.run(main())
```

---

## JavaScript / TypeScript

### 1. Node.js — direct HTTP with `fetch` (Node 18+)

```ts
const BASE = "http://127.0.0.1:8080";
const TOKEN: string | null = null; // set when --require-key-env is on

async function adjudicate(tool: string, args: unknown) {
  const res = await fetch(`${BASE}/v1/fak/adjudicate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
    },
    body: JSON.stringify({ tool, arguments: args }),
  });
  if (!res.ok) throw new Error(`gateway fault ${res.status}`); // a DENY is 200, not an error
  const { verdict, repaired_arguments } = await res.json();
  return { verdict, repaired_arguments };
}

const { verdict } = await adjudicate("Bash", { command: "git push origin main" });
console.log(verdict.kind, verdict.reason, verdict.disposition);
// DENY POLICY_BLOCK TERMINAL

if (verdict.kind === "ALLOW") {
  // run the tool yourself, then optionally admit the result (see Common patterns)
}
```

### 2. Node.js — Anthropic SDK pointed at fak

```ts
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  baseURL: "http://127.0.0.1:8080", // origin; the SDK appends /v1
  apiKey: "fak-local",              // sent as x-api-key
});

const message = await client.messages.create({
  model: "qwen2.5-coder:7b",
  max_tokens: 1024,
  messages: [{ role: "user", content: "List files here" }],
  tools: [{
    name: "Bash",
    description: "Run a shell command",
    input_schema: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  }],
});
// The kernel's drops/repairs also arrive as an in-band "[fak] …" text block in content.
console.log(message.content);
```

The OpenAI SDK works the same way — `new OpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey })`
— and only the surviving tool calls reach `choices[0].message.tool_calls`.

### 3. Browser — `fetch` against the gateway

The same `fetch` call runs in a browser. Two caveats: the gateway must be reachable from the
page's origin (configure CORS / a reverse proxy in front of `fak serve`), and **never ship a
real bearer token to the browser** — front the gateway with your own authenticated backend.

```js
async function checkVerdict(tool, args) {
  const res = await fetch("http://127.0.0.1:8080/v1/fak/adjudicate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool, arguments: args }),
  });
  const { verdict } = await res.json();
  return verdict; // { kind, reason?, by?, disposition?, detail? }
}

checkVerdict("Write", { path: "notes.txt", content: "hi" })
  .then((v) => console.log(v.kind));
```

### 4. Deno

Deno ships `fetch` and the standard `Authorization` header — the Node example runs unchanged.
Run with explicit network access:

```bash
deno run --allow-net=127.0.0.1:8080 adjudicate.ts
```

```ts
const res = await fetch("http://127.0.0.1:8080/v1/fak/adjudicate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ tool: "Read", arguments: { path: "deno.json" } }),
});
console.log((await res.json()).verdict.kind); // ALLOW
```

### 5. Streaming an adjudicated chat (SSE)

The proxy emits a standard `text/event-stream`. Parse `data:` lines and stop at `[DONE]`.

```ts
async function streamChat(prompt: string) {
  const res = await fetch("http://127.0.0.1:8080/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen2.5:1.5b",
      stream: true,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      const chunk = JSON.parse(data);
      const delta = chunk.choices?.[0]?.delta;
      if (delta?.content) Deno.stdout?.writeSync?.(new TextEncoder().encode(delta.content));
      // The final chunk carries chunk.fak with the per-call adjudications.
    }
  }
}
```

---

## Go

The fak-native surface is plain JSON over `net/http` — no SDK, no dependencies (matching the
repo's zero-dependency posture). These mirror the wire DTOs in `fak/internal/gateway/wire.go`.

### 1. Standard library — adjudicate with context cancellation

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type verdict struct {
	Kind        string            `json:"kind"`
	Reason      string            `json:"reason,omitempty"`
	By          string            `json:"by,omitempty"`
	Disposition string            `json:"disposition,omitempty"`
	Detail      map[string]string `json:"detail,omitempty"`
}

type syscallResponse struct {
	Verdict           verdict         `json:"verdict"`
	RepairedArguments json.RawMessage `json:"repaired_arguments,omitempty"`
	TraceID           string          `json:"trace_id,omitempty"`
}

func adjudicate(ctx context.Context, base, token, tool string, args any) (*syscallResponse, error) {
	body, _ := json.Marshal(map[string]any{"tool": tool, "arguments": args})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		base+"/v1/fak/adjudicate", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK { // a DENY is 200 — a non-200 is a real fault
		return nil, fmt.Errorf("gateway fault: %s", resp.Status)
	}
	var out syscallResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func main() {
	// Cancel the call if it outlives 10s.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	out, err := adjudicate(ctx, "http://127.0.0.1:8080", "",
		"Bash", map[string]string{"command": "rm -rf /tmp/x"})
	if err != nil {
		panic(err)
	}
	fmt.Println(out.Verdict.Kind, out.Verdict.Reason, out.Verdict.Disposition)
	// DENY POLICY_BLOCK TERMINAL

	if out.Verdict.Kind == "ALLOW" {
		// run the tool yourself
	}
}
```

> Pointing the official OpenAI-Go or Anthropic-Go SDK at the gateway works too: set the
> client's base URL to `http://127.0.0.1:8080/v1` (OpenAI) or `http://127.0.0.1:8080`
> (Anthropic). The fak-native path above is shown because it needs no third-party module.

### 2. Adjudicate-and-execute, plus result inspection

`POST /v1/fak/syscall` runs one call through the full kernel path and returns the verdict
**and** the executed result envelope (`{status, content, meta}`):

```go
type resultEnvelope struct {
	Status  string            `json:"status"` // OK | ERROR | PENDING
	Content string            `json:"content"`
	Meta    map[string]string `json:"meta,omitempty"`
}

type syscallExecResponse struct {
	Verdict verdict         `json:"verdict"`
	Result  *resultEnvelope `json:"result,omitempty"` // present only on the execute path
	TraceID string          `json:"trace_id,omitempty"`
}
// POST the same {tool, arguments} body to /v1/fak/syscall and decode into the above.
```

### 3. Scraping metrics

```go
resp, err := http.Get("http://127.0.0.1:8080/metrics") // Prometheus exposition format
if err != nil { /* ... */ }
defer resp.Body.Close()
// resp.Body is text/plain; version=0.0.4 — kernel counters: submits, vDSO hits,
// denies, transforms, quarantines, admits. See docs/fak/observability.md.
```

---

## Rust

`reqwest` + `serde` + `tokio` — the standard async stack. Add to `Cargo.toml`:

```toml
[dependencies]
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
```

### Async adjudicate with typed verdict and error handling

```rust
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct Verdict {
    kind: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    disposition: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SyscallResponse {
    verdict: Verdict,
    #[serde(default)]
    repaired_arguments: Option<serde_json::Value>,
    #[serde(default)]
    trace_id: Option<String>,
}

async fn adjudicate(
    client: &reqwest::Client,
    base: &str,
    token: Option<&str>,
    tool: &str,
    args: serde_json::Value,
) -> Result<SyscallResponse, reqwest::Error> {
    let mut req = client
        .post(format!("{base}/v1/fak/adjudicate"))
        .json(&json!({ "tool": tool, "arguments": args }));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    // A DENY is a 200; only a non-2xx is a real fault.
    req.send().await?.error_for_status()?.json().await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;

    let resp = adjudicate(
        &client,
        "http://127.0.0.1:8080",
        None,
        "Bash",
        json!({ "command": "sudo apt-get install" }),
    )
    .await?;

    println!(
        "{} {:?} {:?}",
        resp.verdict.kind, resp.verdict.reason, resp.verdict.disposition
    );
    // DENY Some("POLICY_BLOCK") Some("TERMINAL")

    match resp.verdict.kind.as_str() {
        "ALLOW" => { /* run the tool yourself */ }
        "TRANSFORM" => { /* run with resp.repaired_arguments instead */ }
        _ => { /* refused — inspect disposition to decide whether to retry */ }
    }
    Ok(())
}
```

---

## Common patterns

### Retry logic (disposition-aware)

A refusal carries an actionable `disposition` so a non-Go agent spends **zero** model turns
deciding what to do. Branch on it instead of blindly retrying:

| `disposition` | Meaning | Client action |
|---|---|---|
| `RETRYABLE` | Transient | Retry, ideally with backoff. |
| `WAIT` | Blocked on a pending condition | Back off, then retry. |
| `ESCALATE` | Needs a witness / human approval | Route to an approval queue; don't auto-retry. |
| `TERMINAL` | Structurally refused | Stop. Retrying will never succeed. |

```python
import time

def adjudicate_with_retry(post, tool, args, max_attempts=4):
    delay = 0.5
    for attempt in range(max_attempts):
        verdict = post("/v1/fak/adjudicate", {"tool": tool, "arguments": args})["verdict"]
        if verdict["kind"] in ("ALLOW", "TRANSFORM"):
            return verdict
        if verdict.get("disposition") in ("RETRYABLE", "WAIT") and attempt < max_attempts - 1:
            time.sleep(delay)
            delay *= 2
            continue
        return verdict   # ESCALATE / TERMINAL / exhausted — surface it, don't loop
    return verdict
```

### Timeout handling

Every example sets a client-side timeout (`urllib`'s `timeout=`, `httpx`'s `timeout=`,
`context.WithTimeout` in Go, `reqwest`'s `.timeout(...)`). Match it to the gateway's own
limits: the request body is capped at **4 MiB**, the server `ReadTimeout` defaults to 30 s,
and the upstream model call is bounded by `FAK_PLANNER_TIMEOUT_S` (default 60 s). For slow
local models, raise the server side (`FAK_HTTP_WRITE_TIMEOUT_S`, `FAK_PLANNER_TIMEOUT_S`; see
[server-config.md](server-config.md)) and give your client headroom above that.

### Verdict inspection

Always read `verdict.kind` rather than the HTTP status:

```python
v = resp["verdict"]
if v["kind"] == "ALLOW":
    pass                              # run it
elif v["kind"] == "TRANSFORM":
    run(resp["repaired_arguments"])   # canonical, grammar-repaired args
elif v["kind"] == "QUARANTINE":
    pass                              # result paged out (secret/poison-shaped)
elif v["kind"] == "REQUIRE_WITNESS":
    escalate(v)                       # needs an external witness / approval
else:  # DENY / DEFER
    handle_refusal(v["reason"], v["disposition"])
```

### Tool result processing (admit)

When *your* client runs the tool (not the gateway), send the result back through the
result-side floor with `POST /v1/fak/admit`. A poisoned or secret-shaped result is paged out
(`verdict.kind == "QUARANTINE"`) and the session's IFC taint high-water mark is raised before
the bytes are admitted — arming the exfil floor on the path where fak does not run the tool.

```python
admitted = _post("/v1/fak/admit", {
    "tool": "Bash",
    "result": {"status": "OK", "content": tool_output},
    "trace_id": trace_id,                  # keys the per-trace taint ledger
})
if admitted["verdict"]["kind"] == "QUARANTINE":
    # bytes were paged out — do not feed them to the model
    ...
```

Pass a stable `trace_id` across `/v1/fak/adjudicate` → tool run → `/v1/fak/admit` to correlate
the call-side verdict with the result-side admission on one session. If you omit it, the
gateway mints one and echoes it back in the `trace_id` response field and the `X-Trace-Id`
header.

---

## See also

- [api-reference.md](api-reference.md) — every endpoint, field, and status, generated from the gateway source.
- [Claude integration guide](../integrations/claude.md) — wiring Claude Code and the Anthropic SDK end-to-end.
- [server-quickstart.md](server-quickstart.md) — start a gateway in five scenarios.
- [server-config.md](server-config.md) — every flag and tuning env var.
- [tutorial.md](tutorial.md) — zero-to-first-call with real captured output.
- [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — the policy schema and the full refusal vocabulary.
