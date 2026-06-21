# Migrating to fak

This guide shows how to put `fak` in front of an agent stack you already run —
**LangChain**, **AutoGen**, **llama.cpp**, or a **direct OpenAI / Anthropic API
client** — so that every tool call your agent proposes passes through the kernel's
capability floor *before* it executes. In almost every case the migration is **one
line**: change where your client points.

> **Why migrate at all?** `fak` treats the model as an untrusted program and a tool
> call as a syscall. Today your framework asks a model what to do and then *runs the
> tool call it asked for*. `fak serve` interposes a kernel between those two steps: a
> tool that isn't on a reviewable allow-list is refused **by structure**, a malformed
> call is grammar-repaired, and a poisoned tool result is walled off — none of which
> your framework does on its own. The conceptual background is in the
> [tutorial](tutorial.md); this page is the mechanical "how do I move my existing
> code over" reference.

---

## The one principle behind every migration

`fak serve` exposes **three wire surfaces on one port**, each byte-compatible with a
protocol your client already speaks:

| Your client speaks… | Point it at… | Surface |
|---|---|---|
| OpenAI Chat Completions | `http://127.0.0.1:8080/v1` | `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` |
| Anthropic Messages | `http://127.0.0.1:8080` *(origin — the SDK appends `/v1` itself)* | `/v1/messages` |
| fak-native / MCP | `http://127.0.0.1:8080` | `/v1/fak/*`, `/mcp` |

So **migration = redirect the base URL**. Your prompts, your tool definitions, and
your agent loop stay exactly as they are. fak adjudicates the tool calls in the
middle and returns the survivors, plus a `fak` extension describing every decision.

Two invariants that hold for **all** of the migrations below:

1. **fak never executes your tools — your client does.** The gateway returns only the
   admitted (or repaired) tool calls; your existing agent loop runs them, exactly as
   it does today. This is why LangChain, AutoGen, and a hand-rolled loop all migrate
   the same way.
2. **A refusal is a successful `200`, carried as a value** (deny-as-value). The kernel
   reserves HTTP error statuses for malformed requests, auth failures, and upstream
   faults — *never* for a policy refusal. Your client never has to treat "the kernel
   said no" as an exception. See [api-reference.md → A refusal is not an error](api-reference.md#a-refusal-is-not-an-error).

---

## Before you start: get fak running

Build or install the binary (full matrix in [`INSTALL.md`](../../INSTALL.md) and
[`fak/GETTING-STARTED.md`](../../fak/GETTING-STARTED.md)):

```bash
# Prebuilt binary (no Go required)
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# …or build from a clone (the Go module lives in the fak/ subdir)
git clone https://github.com/anthony-chaudhary/fak.git
cd fleet-public/fak && go build -o fak ./cmd/fak
```

Start a gateway in front of whatever model server you already use. The shape is
always the same — `--base-url` points at your upstream, `--model` is the id fak
advertises:

```bash
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \   # your existing model server
  --model qwen2.5:1.5b \
  --policy policy.json                      # your capability floor (optional but recommended)
```

Confirm it is up before redirecting any client:

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"mock","model":"qwen2.5:1.5b","ok":true}
```

A full flag reference is in [server-quickstart.md](server-quickstart.md) and
[server-config.md](server-config.md). The rest of this page assumes a gateway
listening on `127.0.0.1:8080`.

---

## Migrating from the OpenAI API

If you call the OpenAI API directly (the official `openai` SDK, or raw HTTP), the
migration is a `base_url` change. Your model id, messages, and `tools` array are
forwarded unchanged.

### Start fak in front of OpenAI (or a local OpenAI-compatible server)

```bash
# Proxy the real OpenAI API, adding the kernel boundary
export OPENAI_API_KEY="sk-..."
fak serve --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-4o \
  --policy policy.json
```

`--api-key-env` names the **environment variable** holding your upstream key (fak
reads it from the env, never from a flag), and fak forwards it to OpenAI for you.

### Point the SDK at fak

```python
import openai

# Before:
# client = openai.OpenAI(api_key="sk-...")

# After — the only change is base_url:
client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="fak-local",          # any value when fak auth is off; see "Authentication" below
)

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List the Go files here"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }],
)
```

Your existing tool-execution code stays the same: read `resp.choices[0].message.tool_calls`,
run each surviving call, append the result, loop. The difference is that the
`tool_calls` you receive have **already been through the kernel** — denied calls are
gone, repaired calls carry canonical arguments.

> **Embeddings & moderation also move.** fak ships deterministic, self-contained
> `/v1/embeddings` and `/v1/moderations` backends (no GPU, no network). They are built
> for tests, semantic-cache keys, and smoke checks — **not** a learned replacement for
> OpenAI's models. See [api-reference.md](api-reference.md#post-v1embeddings) before
> repointing production embedding traffic.

---

## Migrating from LangChain

LangChain already executes tools on the client side and talks to models through chat
clients that accept a base-URL override — both a perfect fit for fak. You keep your
chains, agents, and `@tool` definitions; you only change where the chat model points.

### OpenAI-backed chains (`langchain-openai`)

```python
from langchain_openai import ChatOpenAI

# Before:
# llm = ChatOpenAI(model="gpt-4o")

# After:
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="http://127.0.0.1:8080/v1",   # point at fak's OpenAI surface
    api_key="fak-local",
)
```

(On older `langchain-openai` releases the parameter is `openai_api_base` instead of
`base_url` — check your installed version.) `llm.bind_tools([...])` works unchanged:
LangChain sends your tool schemas in the standard OpenAI `tools` shape, and fak
adjudicates each proposed call before your agent executor runs it.

### Anthropic-backed chains (`langchain-anthropic`)

Run fak's Anthropic Messages surface and point `ChatAnthropic` at the **origin** (the
Anthropic SDK appends `/v1` itself, so do **not** include `/v1` here):

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    base_url="http://127.0.0.1:8080",   # origin, NOT .../v1
    api_key="fak-local",
)
```

### What you do *not* change

- Your `@tool` / `StructuredTool` definitions — they serialize to the same tool
  schemas fak reads.
- Your `AgentExecutor` / LangGraph loop — it still executes the (now adjudicated)
  tool calls client-side.
- Your prompts and output parsers.

If a tool call is denied, LangChain simply never sees it in the model's tool-call
list (a fak-unaware client gets a clean turn); the kernel's decision is still
recorded in the `fak` response extension for any code that wants to inspect it.

---

## Migrating from AutoGen

AutoGen's model clients take a `base_url` (v0.4 / AgentChat) or a `config_list` entry
with `base_url` (v0.2). Repoint either at fak and your agents, group chats, and
registered tools are unchanged — AutoGen also runs tools client-side.

### AutoGen v0.4 (`autogen-ext`)

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    base_url="http://127.0.0.1:8080/v1",   # fak's OpenAI surface
    api_key="fak-local",
)
```

> When the model id is **not** a name AutoGen recognizes (e.g. you serve a local
> `qwen2.5-coder:7b` behind fak), AutoGen v0.4 requires you to pass a `model_info`
> block describing the model's capabilities. That is an AutoGen requirement, not a
> fak one — fak advertises whatever id you set with `--model`.

### AutoGen v0.2 (`config_list`)

```python
config_list = [{
    "model": "gpt-4o",
    "base_url": "http://127.0.0.1:8080/v1",   # point the whole config at fak
    "api_key": "fak-local",
}]

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
```

Everything else — `UserProxyAgent`, `register_function`, group chats — keeps working,
because the tool execution still happens in your AutoGen process. fak only decides
*which* proposed calls reach it.

---

## Migrating from llama.cpp

There are two distinct ways `fak` relates to llama.cpp, depending on whether you keep
`llama-server` running or fold the model into the kernel.

### Option A — keep `llama-server`, put fak in front of it (recommended)

`llama-server` exposes an OpenAI-compatible API. Treat it exactly like any other
upstream: point `fak serve --base-url` at it. You gain the kernel boundary without
changing how you run or quantize your model.

```bash
# Your existing llama.cpp server (unchanged)
llama-server -m ./Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8131 --ctx-size 32768 --n-gpu-layers 99

# fak in front of it
fak serve --addr 127.0.0.1:8080 \
  --base-url http://127.0.0.1:8131/v1 \
  --model qwen2.5-7b \
  --policy policy.json
```

Clients that previously hit `http://127.0.0.1:8131/v1` now hit
`http://127.0.0.1:8080/v1`. The model, weights, and sampling are llama.cpp's; the tool
adjudication is fak's.

### Option B — let fak load the GGUF directly (in-kernel engine)

`fak serve` can load a GGUF and run the forward pass **inside the kernel address
space** — no separate `llama-server` process. Drop `--base-url` and pass `--gguf`
(a separate `--tokenizer` is optional; the GGUF's embedded tokenizer is used by
default):

```bash
fak serve --addr 127.0.0.1:8080 \
  --gguf ~/.cache/fak-models/gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --model qwen2.5-0.5b
# Large models: prefix FAK_Q4K=1 to use the direct-resident-Q4_K decode lever.
```

This serves **both** `/v1/chat/completions` and `/v1/messages` from the in-kernel
model.

> **Honest caveat — when to choose which.** fak's in-kernel model path is a
> *correctness reference* proven bit-exact against a HuggingFace oracle, not a
> production-optimized chat engine. For chat-quality serving at scale, keep
> `llama-server` and use **Option A**. Reach for **Option B** when you specifically
> want the model to be kernel-owned state (the deepest fusion). This scope is spelled
> out in [`fak/GETTING-STARTED.md` §4](../../fak/GETTING-STARTED.md) and
> [`fak/CLAIMS.md`](../../fak/CLAIMS.md).

The infrastructure-level differences between fak and per-session servers like
llama.cpp (cross-worker and cross-session KV reuse) are quantified in
[`docs/fak-vs-alternatives-comparison.md`](../../fak/docs/fak-vs-alternatives-comparison.md).

---

## What you gain: reading the `fak` extension

After migrating, the visible new thing on the wire is the top-level **`fak`** object
on `/v1/chat/completions` and `/v1/messages` responses. It is present only on a turn
with tool activity and carries the kernel's decision for every proposed call —
**including the ones that were dropped** (a fak-unaware client simply never sees the
dropped `tool_calls`):

```json
{
  "fak": {
    "adjudications": [
      { "tool_call_id": "…", "tool": "Bash", "admitted": true,
        "verdict": { "kind": "ALLOW", "by": "monitor" } },
      { "tool_call_id": "…", "tool": "rm_rf", "admitted": false,
        "verdict": { "kind": "DENY", "reason": "POLICY_BLOCK",
                     "disposition": "TERMINAL" } }
    ],
    "result_admissions": [
      { "tool_call_id": "…", "tool": "read_file",
        "verdict": { "kind": "QUARANTINE", "reason": "SECRET_EXFIL" } }
    ]
  }
}
```

- `adjudications` — one entry per **proposed** tool call. `repaired_arguments` is
  present only when `verdict.kind == "TRANSFORM"` (the canonical arguments your client
  should run instead).
- `result_admissions` — one entry per **inbound** tool result the kernel screened
  before the model saw it; a `QUARANTINE` kind means the bytes were paged out.

> **Wire note.** Some older integration pages show this as `_fak` with an `admissions`
> array. The current gateway (v0.30.0) emits the **`fak`** key with
> `adjudications` / `result_admissions` as above — verify against
> [api-reference.md → The `fak` response extension](api-reference.md#the-fak-response-extension)
> if your client parses it.

The full `verdict` object (`kind`, `reason`, `by`, `disposition`, `detail`) is
documented in [api-reference.md → The verdict object](api-reference.md#the-verdict-object).

---

## Migrating your permissions into a capability floor

The substantive part of any migration is deciding **which tools your agent may
call**. With no `--policy`, the kernel default-denies every tool, so you author a
reviewable manifest once and load it on the gateway.

```bash
fak policy --dump > policy.json    # start from the built-in default
# edit policy.json (below), then validate before it ever gates a run:
fak policy --check policy.json
```

A manifest is plain JSON (`fak-policy/v1`):

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow":        ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
  "allow_prefix": ["read_", "get_", "search_", "list_"],
  "deny":         { "git_push": "POLICY_BLOCK", "exfiltrate": "SECRET_EXFIL" },
  "self_modify_globs": [".git/", "policy.json", "internal/kernel/"],
  "redact_fields":     ["password", "secret", "api_key", "token"]
}
```

| Field | What it does in a migration |
|---|---|
| `allow` / `allow_prefix` | The tools your framework registers that the agent legitimately needs. Anything not listed here (and not explicitly denied) hits the fail-closed `DEFAULT_DENY`. |
| `deny` | Tools you want refused with a **named, provable** reason (closed vocabulary — see [`fak/POLICY.md`](../../fak/POLICY.md)). |
| `self_modify_globs` | Path fragments that prove a self-modification attempt in a write-shaped call's target argument. |
| `redact_fields` | Arg keys whose value is stripped before dispatch (secret hygiene). |

> **Honest scope — this matters when porting your permission logic.** The floor bounds
> **which tools** run, by tool *name*. It does **not** bound the *arguments* of an
> allow-listed tool (argument-level value predicates are a roadmap item, not shipped).
> So the safe pattern is: keep irreversible / exfil-shaped operations **off** the
> allow-list and let `DEFAULT_DENY` hold them, rather than allow-listing a broad tool
> and hoping to filter its arguments. The full honest-scope discussion is in
> [`fak/POLICY.md`](../../fak/POLICY.md).

Ready-made starting points ship in [`fak/examples/`](../../fak/examples/):
`dev-agent-policy.json` (coding agent), `research-agent-policy.json` (read-only),
`customer-support-readonly-policy.json`, and `devops-dryrun-policy.json`.

---

## Authentication after you migrate

fak auth is **off by default** (loopback-friendly), which is why the examples above
pass a throwaway `api_key="fak-local"`. For a network-facing gateway, require a
secret:

```bash
export FAK_TOKEN="$(openssl rand -hex 32)"
fak serve --addr 0.0.0.0:8080 --base-url … --model … \
  --require-key-env FAK_TOKEN
```

Then **every route except `/healthz`** requires the secret. fak accepts it under
either header, so each client type works unchanged:

- OpenAI / LangChain-OpenAI / AutoGen / fak-native clients → `Authorization: Bearer $FAK_TOKEN`
  (set the SDK's `api_key` to the token's value).
- Anthropic / LangChain-Anthropic clients → `x-api-key: $FAK_TOKEN`.

See [security.md](security.md) for hardening a reachable gateway and
[server-config.md → Authentication](server-config.md) for the details.

---

## Verifying the migration

Independent of any client, confirm the boundary is live:

```bash
# 1. The gateway is up and advertising your model
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/v1/models

# 2. An allow-listed call is admitted; a non-allow-listed one is refused —
#    and the refusal is a 200 carrying a DENY verdict, not an HTTP error.
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate \
  -H 'Content-Type: application/json' \
  -d '{"tool":"refund_payment","arguments":{}}'
# {"verdict":{"kind":"DENY","reason":"DEFAULT_DENY","disposition":"TERMINAL",...}}
```

Or check a single call against your policy with no server at all:

```bash
fak preflight --policy policy.json --tool git_push --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor
```

The guided, fully-captured walkthrough of these commands is in
[tutorial.md](tutorial.md).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Client gets `404` on `/v1/v1/messages` | You included `/v1` in an **Anthropic** base URL. Anthropic SDKs append `/v1` themselves — point them at the origin (`http://127.0.0.1:8080`). OpenAI clients **do** include `/v1`. |
| Every tool call is denied | No `--policy` loaded ⇒ default-deny everything. Pass `--policy policy.json` (and `fak policy --check` it first). |
| `401 Unauthorized` from fak | `--require-key-env` is set; send the secret as `Authorization: Bearer …` (OpenAI-style) or `x-api-key: …` (Anthropic-style). A bare `Authorization` value with no `Bearer ` prefix is rejected. |
| `502` from `/v1/chat/completions` | Upstream model error, or the model announced tool calls but none parsed (fail-closed). Fix the `--base-url` upstream first; its raw error body is intentionally not forwarded. |
| The model ignores tools entirely | Use a tool-calling model; base completion models don't emit `tool_calls`. |
| Streaming looks "bursty" | fak buffers the whole upstream turn, adjudicates it, then re-emits a well-formed SSE stream — the wire is identical but partial tokens are never passed through before adjudication. |
| `/v1/fak/syscall` returns an odd/empty result | The fak-native key is `arguments`, **not** `args` — unknown keys are silently dropped. |

---

## See also

- [tutorial.md](tutorial.md) — zero-to-first-call with real captured output at every step.
- [api-reference.md](api-reference.md) — every endpoint, field, and the `fak` extension in full.
- [server-quickstart.md](server-quickstart.md) · [server-config.md](server-config.md) — every flag and environment variable.
- [policy-guide.md](policy-guide.md) · [`fak/POLICY.md`](../../fak/POLICY.md) — authoring the capability floor.
- [security.md](security.md) — hardening a network-reachable gateway.
- [`docs/integrations/claude.md`](../integrations/claude.md) · [`docs/integrations/openai-codex.md`](../integrations/openai-codex.md) · [`docs/integrations/cursor.md`](../integrations/cursor.md) — per-client integration playbooks.
- [`docs/fak-vs-alternatives-comparison.md`](../../fak/docs/fak-vs-alternatives-comparison.md) — fak vs llama.cpp / vLLM / provider caching, quantified.
```
