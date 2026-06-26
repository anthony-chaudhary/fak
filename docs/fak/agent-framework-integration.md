---
title: "fak agent framework integration: LangChain to CrewAI"
description: "Per-framework cookbook for putting fak in front of LangChain, LlamaIndex, AutoGen, CrewAI, and OpenAI-compatible agents via proxy or explicit adjudication."
---

# Agent Framework Integration Guide

A per-framework cookbook for putting `fak` in front of a tool-using agent built on
**LangChain / LangGraph**, **LlamaIndex**, **AutoGen**, **CrewAI**, or any
**OpenAI-compatible** client — plus **Semantic Kernel**, **Haystack**, and
**Griptape**. For every framework the question is the same: *what is the smallest,
exact change that makes each tool call your agent proposes pass through the kernel's
capability floor before it runs?*

This page is the concrete **"do exactly this"** companion to two existing docs — read
them first if you have not:

- [agent-integration-architecture.md](agent-integration-architecture.md) — the **why /
  how it fits the kernel**: the gateway entry points, the ABI, the verdict union, the
  context-MMU. The conceptual model this page assumes.
- [migration-guide.md](migration-guide.md) — **moving existing code over by repointing a
  base URL** (the one-line migration for LangChain, AutoGen, the OpenAI SDK, and
  llama.cpp). Where that guide already covers a framework, this page links to it rather
  than repeating it, and adds the framework-specific tool-wrapper recipe and the
  frameworks it does not cover.

> **Two invariants that hold for every framework below** (from
> [migration-guide.md](migration-guide.md#the-one-principle-behind-every-migration)):
> 1. **fak never executes your tools — your framework does.** The gateway returns only
>    the admitted (or argument-repaired) calls; your existing agent loop runs them.
> 2. **A refusal is a successful `200`, carried as a value** (deny-as-value). HTTP error
>    statuses are reserved for malformed requests, auth failures, and upstream faults —
>    never for a policy refusal.

---

## Two ways to put fak in front of a framework

There are exactly two integration shapes. Most frameworks support both; pick by how
much control you need at the tool boundary.

| | **Mode A — transparent proxy** | **Mode B — explicit adjudication** |
|---|---|---|
| **What you change** | The framework's LLM client `base_url` → fak's `/v1` origin. | Wrap each tool so it calls `/v1/fak/adjudicate` before running and `/v1/fak/admit` after. |
| **Who adjudicates** | The gateway adjudicates every **proposed** tool call inside `/v1/chat/completions` (or `/v1/messages`) before your framework ever sees it: denied calls are dropped, transformed calls are argument-repaired. Inbound `role: "tool"` results that pass back through the proxy are screened by the result-side floor. | Your tool wrapper gets the kernel verdict **synchronously at the call site**, and screens the tool's **output** through quarantine + the IFC taint floor even when the result never round-trips through the chat proxy. |
| **Code change** | One line (the base URL). Tool definitions, agent loop, prompts unchanged. | A thin wrapper around each registered tool (a dozen lines, shared across tools). |
| **Best when** | You want the kernel boundary with zero changes to your tool code, and your agent already round-trips results through the model. | You need to act on the verdict at the exact tool site (block / substitute repaired args), use multiple models/providers, or quarantine tool output your agent consumes directly. |

Mode A is the [migration-guide.md](migration-guide.md) path. The rest of this page gives
you Mode A in one line per framework **and** the Mode B tool wrapper, which the migration
guide does not cover.

---

## Before you start: a gateway with a policy

Every recipe below assumes a `fak serve` gateway on `127.0.0.1:8080` with a capability
floor loaded. With **no** `--policy`, the kernel default-denies every tool — that is the
fail-closed posture, not a misconfiguration.

```bash
fak policy --dump > policy.json     # start from the built-in default, then edit
fak policy --check policy.json      # validate before it ever gates a run
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \   # your existing model server
  --model qwen2.5:1.5b \
  --policy policy.json
```

Confirm it is live before pointing any framework at it:

```bash
curl -s http://127.0.0.1:8080/healthz
# {"engine":"mock","model":"qwen2.5:1.5b","ok":true}
```

Full flag and scenario reference: [server-quickstart.md](server-quickstart.md) ·
[server-config.md](server-config.md). Cloud upstreams (OpenAI, Anthropic, Gemini, xAI),
authentication, and the in-kernel GGUF engine are covered there and in
[migration-guide.md](migration-guide.md).

---

## The shared Mode B helper

Every Mode B example reuses these two functions. They wrap the two fak-native endpoints
documented in [api-reference.md](api-reference.md#fak-native-surface): `/v1/fak/adjudicate`
(pre-execution verdict, no side effects) and `/v1/fak/admit` (screen a client-produced
result through quarantine + the IFC taint floor).

```python
import requests

FAK = "http://127.0.0.1:8080"
# When the gateway is started with --require-key-env, send the secret as a Bearer token:
#   HEADERS = {"Authorization": f"Bearer {os.environ['FAK_TOKEN']}"}
HEADERS: dict = {}


def fak_adjudicate(tool: str, arguments: dict) -> dict:
    """Pre-execution verdict for ONE proposed tool call. No dispatch, no side effects.
    Returns {"verdict": {...}, "repaired_arguments": {...}?, "trace_id": "..."}."""
    r = requests.post(f"{FAK}/v1/fak/adjudicate",
                      json={"tool": tool, "arguments": arguments}, headers=HEADERS)
    r.raise_for_status()          # a 4xx/5xx is malformed/auth/upstream — NEVER a refusal
    return r.json()


def fak_admit(tool: str, result) -> dict:
    """Screen a tool RESULT the client just produced, BEFORE the agent reads it.
    A QUARANTINE verdict means the bytes were paged out of context.
    Returns {"verdict": {...}, "result": {...}, "trace_id": "..."}."""
    r = requests.post(f"{FAK}/v1/fak/admit",
                      json={"tool": tool, "result": result}, headers=HEADERS)
    r.raise_for_status()
    return r.json()
```

The `arguments` and `result` fields accept either a JSON object (a Python `dict`) or a
JSON-encoded string (the OpenAI `function.arguments` convention) — see the
[`SyscallRequest`/`AdmitRequest` reference](api-reference.md#post-v1fakadjudicate).

A single **guarded-tool** wrapper turns any plain tool function into a kernel-governed
one. It implements the canonical verdict handling once; every framework section below
just hands its tool function to it:

```python
class ToolDenied(Exception):
    def __init__(self, verdict: dict):
        self.verdict = verdict
        super().__init__(f"{verdict.get('kind')}: {verdict.get('reason', '')}")


def guarded(tool_name: str, fn):
    """Wrap a tool fn so every call is adjudicated (pre) and admitted (post)."""
    def call(**kwargs):
        adj = fak_adjudicate(tool_name, kwargs)
        v = adj["verdict"]
        if v["kind"] == "DENY":
            # deny-as-value: surface the reason; do NOT run the tool.
            raise ToolDenied(v)
        if v["kind"] == "REQUIRE_WITNESS":
            raise ToolDenied(v)                 # route to your approval/witness queue
        if v["kind"] == "TRANSFORM":
            kwargs = adj["repaired_arguments"]  # run the kernel's canonical args, not the model's

        out = fn(**kwargs)                       # your REAL tool executes here, client-side

        adm = fak_admit(tool_name, out)
        if adm["verdict"]["kind"] == "QUARANTINE":
            return f"[fak] tool result quarantined ({adm['verdict'].get('reason', '')})"
        return out
    return call
```

The verdict `kind` is one of `ALLOW` · `DENY` · `TRANSFORM` · `QUARANTINE` ·
`REQUIRE_WITNESS` · `DEFER`; the `reason` is from the closed refusal vocabulary
(`DEFAULT_DENY`, `POLICY_BLOCK`, `SECRET_EXFIL`, …). The full object — including
`disposition` (`RETRYABLE` · `WAIT` · `ESCALATE` · `TERMINAL`), which tells your loop
whether a refusal is worth retrying — is in
[api-reference.md → The verdict object](api-reference.md#the-verdict-object).

> **The repoint parameter differs by framework — and by version.** Frameworks name the
> custom-base-URL option differently (`base_url`, `openai_api_base`, `api_base`,
> `api_base_url`, a custom client object). Each section names the one that framework
> currently uses; **verify against your installed version**. What never changes is fak's
> surface: the OpenAI `/v1` origin and the two adjudication endpoints above.

---

## Generic OpenAI-compatible clients

Anything that speaks OpenAI Chat Completions — the official `openai` SDK, raw `requests`,
or a niche client — integrates by pointing `base_url` at fak's `/v1`.

**Mode A** (the [migration-guide OpenAI section](migration-guide.md#migrating-from-the-openai-api)
has the full version, including upstream auth):

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8080/v1",   # the only change
    api_key="fak-local",                   # any value when fak auth is off
)
resp = client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])
# resp.choices[0].message.tool_calls -> ONLY the admitted/repaired calls.
# resp.fak.adjudications -> the kernel's decision for EVERY proposed call (incl. dropped).
```

Read the per-turn decisions from the top-level `fak` extension (present only on a
tool-activity turn): `adjudications` (one entry per proposed call, including dropped ones)
and `result_admissions` (one entry per screened inbound result). See
[api-reference.md → The `fak` response extension](api-reference.md#the-fak-response-extension).

**Mode B** — when you run your own loop and want the verdict at the tool site, adjudicate
each call the model returns before executing it:

```python
for tc in resp.choices[0].message.tool_calls:
    import json
    args = json.loads(tc.function.arguments)
    adj = fak_adjudicate(tc.function.name, args)
    if adj["verdict"]["kind"] == "DENY":
        tool_output = f"refused: {adj['verdict']['reason']}"
    else:
        if adj["verdict"]["kind"] == "TRANSFORM":
            args = adj["repaired_arguments"]
        tool_output = run_my_tool(tc.function.name, args)   # your dispatcher
        tool_output = fak_admit(tc.function.name, tool_output)["result"]
    # append tool_output as a role:"tool" message and continue the loop
```

The zero-framework smoke test for the same boundary is one `curl`:

```bash
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate \
  -H 'Content-Type: application/json' \
  -d '{"tool":"refund_payment","arguments":{}}'
# {"verdict":{"kind":"DENY","reason":"DEFAULT_DENY","disposition":"TERMINAL",...}}
```

---

## LangChain & LangGraph

LangChain and LangGraph execute tools client-side and talk to models through chat clients
that accept a base-URL override — both a clean fit for fak.

**Mode A** — repoint the chat model (the
[migration-guide LangChain section](migration-guide.md#migrating-from-langchain) has the
OpenAI- and Anthropic-backed variants and the `openai_api_base` legacy note):

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    base_url="http://127.0.0.1:8080/v1",   # point at fak's OpenAI surface
    api_key="fak-local",
)
# llm.bind_tools([...]) and your AgentExecutor / LangGraph graph are unchanged.
```

**Mode B** — wrap each `@tool` so it is adjudicated at the call site. This is the
"custom tool wrapper" the integration backlog asks for, and it composes with Mode A
(belt and suspenders) or stands alone:

```python
from langchain_core.tools import StructuredTool

def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

# guarded() (defined above) adjudicates, applies TRANSFORM repairs, runs the tool,
# then admits/quarantines the result.
read_file = StructuredTool.from_function(
    func=guarded("read_file", _read_file),
    name="read_file",
    description="Read a UTF-8 text file by path.",
)
# Pass read_file into bind_tools([...]) / create_react_agent([...]) as usual.
```

**LangGraph note.** A LangGraph `ToolNode` is just a node that runs your tool functions,
so wrapping the functions with `guarded(...)` governs every tool the graph can take —
no change to the graph topology. If you instead want a single choke point, put one node
*before* the `ToolNode` that calls `fak_adjudicate` on the pending tool call in
`state["messages"][-1].tool_calls` and routes to an error node on `DENY`.

If a call is denied under Mode A, LangChain never sees it in the model's tool-call list
(a fak-unaware client gets a clean turn); the decision is still recorded in the `fak`
response extension.

---

## LlamaIndex

LlamaIndex's OpenAI LLM uses **`api_base`** (not `base_url`) for a custom endpoint, and
wraps tools as `FunctionTool`.

**Mode A** — repoint the LLM:

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o", api_base="http://127.0.0.1:8080/v1", api_key="fak-local")
```

For a local, non-OpenAI model served behind fak, use `OpenAILike` (same `api_base`),
which avoids LlamaIndex's OpenAI-model-name validation:

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(model="qwen2.5-7b", api_base="http://127.0.0.1:8080/v1",
                 api_key="fak-local", is_chat_model=True)
```

**Mode B** — function calling with a governed tool (covers "tool governance" and "result
quarantine"): wrap the function before handing it to `FunctionTool`, and the helper's
`fak_admit` step quarantines a secret-shaped or poisoned tool result before the agent
reads it.

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

def _http_get(url: str) -> str:
    import requests
    return requests.get(url, timeout=10).text

http_get = FunctionTool.from_defaults(fn=guarded("http_get", _http_get), name="http_get")
agent = ReActAgent.from_tools([http_get], llm=llm)
```

---

## AutoGen

AutoGen runs tools in your process and takes a `base_url` on its model client (v0.4) or in
a `config_list` entry (v0.2). The base-URL repoint for both versions — including the
`model_info` requirement for unrecognized local model ids — is in the
[migration-guide AutoGen section](migration-guide.md#migrating-from-autogen):

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient   # AutoGen v0.4

model_client = OpenAIChatCompletionClient(
    model="gpt-4o", base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
```

**Mode B — tool-call interception in a multi-agent chat.** AutoGen tools are plain
callables registered on an agent, so `guarded(...)` is the interception point: every tool
an agent (or any agent in a group chat) invokes is adjudicated and its result admitted,
giving a uniform safety boundary across the conversation.

```python
from autogen_core.tools import FunctionTool   # v0.4 tool wrapper

def _run_sql(query: str) -> str:
    return my_db.execute(query)   # your real executor

run_sql = FunctionTool(guarded("run_sql", _run_sql), description="Run a read-only SQL query.")
# Register run_sql on the AssistantAgent's tools=[...] as usual.
```

Because tool execution stays in your AutoGen process, fak only decides *which* proposed
calls reach the tool and *whether* each result is admitted — the agents, group chats, and
hand-offs are unchanged.

---

## CrewAI

CrewAI drives models through LiteLLM, whose `LLM` wrapper accepts `base_url`, and exposes
tools as `BaseTool` subclasses or `@tool` functions.

**Mode A** — repoint the crew's LLM (prefix the model id with its provider, the LiteLLM
convention; you can also set `OPENAI_API_BASE=http://127.0.0.1:8080/v1` in the
environment instead of the kwarg):

```python
from crewai import LLM, Agent

llm = LLM(model="openai/gpt-4o", base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
analyst = Agent(role="Analyst", goal="...", backstory="...", llm=llm)
```

**Mode B — task governance with a guarded tool.** Subclass `BaseTool` and route its `_run`
through `guarded(...)` so every task that uses the tool is adjudicated, and a poisoned
result is quarantined before it enters the crew's shared context:

```python
from crewai.tools import BaseTool

class FetchTool(BaseTool):
    name: str = "fetch_url"
    description: str = "Fetch the text at a URL."

    def _run(self, url: str) -> str:
        return guarded("fetch_url", _http_get)(url=url)   # _http_get from the LlamaIndex example

crew_tools = [FetchTool()]
# Attach crew_tools to the Agent(tools=...) / Task that needs them.
```

The policy floor *is* the task-level tool policy: list the tools each crew legitimately
needs in `allow` / `allow_prefix`, and `DEFAULT_DENY` holds everything else (see
[the policy floor](#the-policy-floor-your-tool-allow-list), below).

**Manager-worker (hierarchical) pattern.** For CrewAI's *hierarchical* process — a
manager agent delegating subtasks to workers — route the `manager_llm` through fak too,
so the manager's coordination prompts hit the shared-brief KV cache instead of
re-prefilling the shared crew context on every delegation. A runnable, dependency-free
example crew (governance over every worker tool call + the manager-role
coordination-overhead model) is in
[`examples/crewai-crew/`](https://github.com/anthony-chaudhary/fak/tree/main/examples/crewai-crew).

---

## Other frameworks

The same two-mode pattern carries over. Each of these speaks OpenAI Chat Completions and
exposes a custom-endpoint option; the exact parameter name is what changes.

### Semantic Kernel (Python)

Semantic Kernel takes a custom `AsyncOpenAI` client, so point that client at fak:

```python
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

fak_client = AsyncOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
chat = OpenAIChatCompletion(ai_model_id="gpt-4o", async_client=fak_client)
```

For Mode B, wrap the function you expose as a kernel function (`@kernel_function`) with
`guarded(...)` exactly as in the LangChain example.

### Haystack (2.x)

Haystack's OpenAI generators take **`api_base_url`**:

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

gen = OpenAIChatGenerator(
    model="gpt-4o",
    api_base_url="http://127.0.0.1:8080/v1",
    api_key=Secret.from_token("fak-local"),
)
```

Wrap any tool/function you register on the generator with `guarded(...)` for Mode B.

### Griptape

Griptape's `OpenAiChatPromptDriver` takes **`base_url`**:

```python
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.structures import Agent

driver = OpenAiChatPromptDriver(
    model="gpt-4o", base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
agent = Agent(prompt_driver=driver)
```

(The driver import path and config wiring have moved across Griptape versions — confirm
against your installed release.) For Mode B, route a custom Tool's activity through
`guarded(...)`.

---

## Handling verdicts: the common pattern

Whatever framework you use, the kernel speaks one verdict vocabulary. The `guarded(...)`
helper above already implements the safe defaults; this is what each `kind` means for your
loop:

| Verdict `kind` | What your agent loop should do |
|---|---|
| `ALLOW` | Run the call as proposed. |
| `TRANSFORM` | Run the call with `repaired_arguments`, **not** the model's original args (a grammar/canonicalization repair). |
| `DENY` | Do **not** run the call. Surface `reason` (`POLICY_BLOCK`, `DEFAULT_DENY`, `SECRET_EXFIL`, …). `disposition` says whether it is `RETRYABLE` / `WAIT` / `ESCALATE` / `TERMINAL`. |
| `QUARANTINE` | (Result side.) The tool's output was paged out of context — give the model a stub, never the raw bytes. |
| `REQUIRE_WITNESS` | Gate the call pending independent verification — route to your approval/witness queue (`disposition: ESCALATE`). |
| `DEFER` | Not adjudicable at this link; in a single-gateway deployment you will not normally observe this on the wire. |

A `DENY` is **not** an exception on the wire — it arrives as a `200` with a verdict value.
The helper raises a Python exception only as a convenience so your loop can branch; the
gateway never returns an HTTP error for a refusal.

---

## The policy floor: your tool allow-list

The substantive integration decision — for every framework — is **which tools the agent
may call**. That lives in one reviewable JSON manifest (`fak-policy/v1`), loaded with
`--policy`, not in framework code:

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow":        ["read_file", "http_get", "run_sql"],
  "allow_prefix": ["read_", "get_", "search_", "list_"],
  "deny":         { "delete_account": "POLICY_BLOCK", "exfiltrate": "SECRET_EXFIL" },
  "self_modify_globs": [".git/", "policy.json"],
  "redact_fields":     ["password", "secret", "api_key", "token"]
}
```

The tool **names** here must match the names your framework registers — the `name` of a
LangChain `StructuredTool`, a LlamaIndex `FunctionTool`, an AutoGen `FunctionTool`, a
CrewAI `BaseTool`, and the `tool` string you pass to `fak_adjudicate`. Anything not in
`allow` / `allow_prefix` (and not explicitly denied) hits the fail-closed `DEFAULT_DENY`.

> **Honest scope.** The floor bounds **which tools** run, by tool *name* — it does **not**
> filter the *arguments* of an allow-listed tool (argument-value predicates are a roadmap
> item, not shipped). Keep irreversible / exfil-shaped operations **off** the allow-list
> and let `DEFAULT_DENY` hold them, rather than allow-listing a broad tool and hoping to
> constrain its arguments. Full discussion: [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).

Ready-made starting points ship in [`fak/examples/`](https://github.com/anthony-chaudhary/fak/tree/main/examples):
`dev-agent-policy.json` (coding agent), `research-agent-policy.json` (read-only),
`customer-support-readonly-policy.json`, and `devops-dryrun-policy.json`. Authoring
details are in [policy-guide.md](policy-guide.md).

---

## Verifying the integration

Independent of any framework, confirm the boundary is live — these are the same checks
whether you wired LangChain, CrewAI, or a raw client:

```bash
# 1. The gateway is up and advertising your model.
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/v1/models

# 2. An allow-listed tool is admitted; a non-allow-listed one is refused —
#    and the refusal is a 200 carrying a DENY verdict, not an HTTP error.
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate \
  -H 'Content-Type: application/json' \
  -d '{"tool":"read_file","arguments":{"path":"README.md"}}'
curl -s -X POST http://127.0.0.1:8080/v1/fak/adjudicate \
  -H 'Content-Type: application/json' \
  -d '{"tool":"refund_payment","arguments":{}}'
# -> {"verdict":{"kind":"DENY","reason":"DEFAULT_DENY",...}}
```

Or check a single call against a policy with no server running at all:

```bash
fak preflight --policy policy.json --tool delete_account --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor
```

A common-issues table for integration symptoms (`404` on `/v1/v1/messages`, every call
denied, `401`, `502`, streaming behavior) is in the
[migration-guide troubleshooting section](migration-guide.md#troubleshooting).

---

## See also

- [agent-integration-architecture.md](agent-integration-architecture.md) — the kernel,
  the ABI, and the verdict union behind these recipes.
- [migration-guide.md](migration-guide.md) — the one-line base-URL migration for
  LangChain, AutoGen, the OpenAI SDK, and llama.cpp.
- [api-reference.md](api-reference.md) — every endpoint, the fak-native DTOs, the verdict
  object, and the `fak` response extension in full.
- [policy-guide.md](policy-guide.md) · [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — authoring
  the capability floor and the closed refusal vocabulary.
- [server-quickstart.md](server-quickstart.md) · [server-config.md](server-config.md) —
  every `fak serve` flag and environment variable.
- [security.md](security.md) — hardening a network-reachable gateway (auth, bind address).
- [tutorial.md](tutorial.md) — zero-to-first-call with real captured output at every step.
