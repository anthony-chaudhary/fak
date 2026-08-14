---
title: "Embed fak in your product — the app that makes a direct API call"
description: "The front door for a product/SaaS feature that calls a model API directly (client.messages.create / chat.completions.create) and is not a coding agent: keep your API call, point its base URL at `fak serve`, and get the audit trail, trace ids, budget accounting, and a capability floor over your app's own tools. Resolves the `guard` vs `serve` category error for server-side products."
---

# Embed fak in your product (direct API call, no coding agent)

**Reader:** a product or backend developer. Your feature already makes a **direct model
API call** — a job-application screener, a support-reply drafter, a résumé
field-extractor, a content classifier — and you want fak in front of *that call*. You are
**not** running a coding agent; you own your own loop and call
`client.messages.create(...)` / `client.chat.completions.create(...)` yourself.

Every other guide in this index assumes the caller is an agent harness that proposes host
tool calls (Bash/Read/Edit/…). This one does not.

---

## The one-paragraph decision

**Keep your API call. Point its base URL at `fak serve`.** That is the whole
integration: `fak serve` is the long-lived gateway runtime — it speaks the OpenAI Chat
Completions and Anthropic Messages wires your SDK already speaks, adjudicates any tool
call the model proposes, and proxies to the upstream that actually serves your tokens.

The category error to head off: **`fak manage` is not an alternative to your API call.**
`fak manage -- <agent>` is a **child-process launcher** — it starts the gateway
in-process, injects the base URL into the *child process only*, and tears the gateway
down when that child exits. A server-side product has no child agent to launch, so there
is nothing for `guard` to wrap. The other verb you will see, `fak serve --native` (the
agent application runtime, epic #3256), is the *opposite* delegation: fak owns the agent
loop and your code drives it. You want neither — you keep your loop.

| Verb | What it is | Fits a server-side product feature? |
|---|---|---|
| `fak manage -- <agent>` | child-process launcher: in-process gateway + base-URL injection into one child, torn down on exit | No — there is no child to launch |
| `fak serve` | long-lived gateway runtime: your client repoints one base URL | **Yes — this guide** |
| `fak serve --native` | agent application runtime: fak owns the loop | No — you keep your own loop |

Naming and first-decision background:
[Two runtimes, one binary](../explainers/runtime-vs-client.md). A runtime chooser verb is tracked in
[#3257](https://github.com/anthony-chaudhary/fak/issues/3257), and a first-class
explainer for the guard/serve *enforcement* boundary (request-arrival vs turn-end) in
[#5162](https://github.com/anthony-chaudhary/fak/issues/5162).

---

## Worked example: a job-application screener

The product feature: an ATS endpoint that (a) **extracts** structured fields from a
submitted résumé — a tool-less completion — and (b) lets the model propose a
**`screen_candidate`** call — an app-defined tool your code executes. Both calls go
through one gateway.

### 1. Start the gateway in front of your provider

```bash
fak serve \
  --addr 127.0.0.1:8080 \
  --provider anthropic \
  --base-url https://api.anthropic.com/v1 \
  --api-key-env ANTHROPIC_API_KEY \
  --model claude-sonnet-4-20250514 \
  --policy screener-floor.json \
  --require-key-env FAK_TOKEN
```

`--provider openai --base-url https://api.openai.com/v1` (or any OpenAI-compatible
upstream — vLLM, Ollama, llama-server) works the same; `--provider gemini` / `xai` front
those providers. `--require-key-env FAK_TOKEN` makes the gateway require a bearer /
`x-api-key` on every request — set it for anything beyond loopback.
`screener-floor.json` is authored in step 3; the flag matters, as step 3 explains.

### 2. The tool-less call (extract / classify / draft)

Change one line in your product code — the base URL:

```python
import os
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8080",       # was: api.anthropic.com
    api_key=os.environ["FAK_TOKEN"],        # the gateway bearer (--require-key-env), not the provider key
)

resp = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    messages=[{"role": "user", "content": f"Extract name, email, and years of experience as JSON:\n{resume_text}"}],
    extra_headers={"X-Trace-Id": f"application-{application_id}"},
)
```

Or on the OpenAI wire:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key=os.environ["FAK_TOKEN"])
```

The provider key stays server-side in the gateway (`--api-key-env`); your product no
longer holds it.

**What fak adds when there are zero tools.** A plain completion still crosses the
governance band:

- **A per-call trace id.** The gateway honors a client-supplied `X-Trace-Id` header (or
  mints one per request), and that id ties together the request log line, every kernel
  decision, and the metrics — pass your application/request id and each screening
  decision becomes traceable end to end.
- **A durable audit trail.** With `FAK_AUDIT_JOURNAL=/var/log/fak-audit.jsonl` set on
  the gateway, every kernel decision is a hash-chained, tamper-evident JSONL row; JSON
  request logs stream with `--log FILE`; Prometheus counters are on `GET /metrics`.
- **Budget accounting.** `--context-budget-tokens N` seeds a session token budget the
  gateway debits from provider usage; on exhaustion the caller gets a `409` with a
  continuation id (or, with `--reset-on-budget`, a transparent re-arm). Use
  `--session-id` / `X-Trace-Id` to scope budgets to your product's sessions.
- **Result-side hygiene when tools appear later.** The policy's `redact_fields`
  (`password`, `secret`, `api_key`, `token`, `authorization` in the defaults) redact
  matching tool-call argument fields, and poisoned or secret-shaped tool results are
  quarantined before re-entering context. Note the honest scope: today this fires on
  the *tool-call* seam; broader prompt-side PII/redaction governance at the gateway is
  the parity work tracked in
  [#3280](https://github.com/anthony-chaudhary/fak/issues/3280).

### 3. The app-tool call — and why the default floor denies it

Now the screener passes its own tool:

```python
resp = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    messages=[{"role": "user", "content": f"Screen this application:\n{resume_text}"}],
    tools=[{
        "name": "screen_candidate",
        "description": "Record a screening decision for a candidate",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["advance", "hold", "reject"]},
                "rationale": {"type": "string"}
            },
            "required": ["candidate_id", "decision"]
        }
    }],
    extra_headers={"X-Trace-Id": f"application-{application_id}"},
)
```

**Out of the box this call is DENIED, and your feature stops working.** The floors fak
ships are not written for your product's tools:

- `fak serve` with **no `--policy`** loads the built-in demo floor (the tau2
  airline-demo tools plus read-only `read_`/`get_`/`search_`/`list_`/`lookup_`/`find_`
  prefixes — see `fak policy --dump`).
- `fak manage`'s embedded floor (`cmd/fak/guard-default-policy.json`) allow-lists a
  *coding harness's* tools — `Bash`, `Read`, `Edit`, `Write`, `Grep`, … — plus the same
  read-only prefix families.

Your `screen_candidate`, `send_offer`, `reject_applicant` are on neither list, and the
kernel is **fail-closed**: an unknown tool is a `DENY (DEFAULT_DENY)`, by structure, not
by model judgment. That is the feature working as designed — and it means step 1's
`--policy screener-floor.json` is not optional. Author the floor for *your* tools:

```json
{
  "version": "fak-policy/v1",
  "allow": [
    "screen_candidate"
  ],
  "deny": {
    "send_offer": "POLICY_BLOCK",
    "reject_applicant": "POLICY_BLOCK"
  },
  "redact_fields": ["password", "secret", "api_key", "token", "authorization"]
}
```

The shape to aim for: the model may *record a screening decision* (your code still
executes it), but consequential actions — sending an offer, rejecting an applicant — are
named DENYs so a prompt-injected résumé cannot drive them. Validate the manifest with
`fak policy --check screener-floor.json`; the full schema is in
[`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md). Scaffolding
this floor *from your app's existing tool schemas* (`fak policy --from-tools`) is the
sibling ticket [#5153](https://github.com/anthony-chaudhary/fak/issues/5153), so the
starting point stops being hand-written JSON.

When the model proposes the call, the kernel's verdict rides along in the response's
`fak` extension (`"fak": {"adjudications": [...]}` on the OpenAI wire, `_fak` on the
Anthropic wire), so your loop can log or act on the decision per tool call. Your code
can also ask for a verdict *without* executing anything — useful before running a tool
your own executor owns:

```bash
curl -s http://127.0.0.1:8080/v1/fak/adjudicate \
  -H 'content-type: application/json' \
  -d '{"tool":"send_offer","arguments":{"candidate_id":"c-118"}}'
# -> {"verdict":{"kind":"DENY","reason":"POLICY_BLOCK", ...}}
```

### 4. Prove the floor offline (no key, no model, no GPU)

Runnable from a clone with only Go installed — the same kernel code, no network:

```bash
go run ./cmd/fak preflight --policy screener-floor.json --tool screen_candidate --args '{"candidate_id":"c-118","decision":"hold"}'
# -> verdict=ALLOW ...
go run ./cmd/fak preflight --policy screener-floor.json --tool send_offer --args '{"candidate_id":"c-118"}'
# -> verdict=DENY reason=POLICY_BLOCK ...          (named deny)
go run ./cmd/fak preflight --policy screener-floor.json --tool delete_candidate --args '{}'
# -> verdict=DENY reason=DEFAULT_DENY ...          (unknown tool, fail-closed)
```

Put these in CI next to the floor file: the policy that governs your product is a
reviewable JSON manifest in git, and its allow/deny behavior is asserted like any other
test.

---

## The honest fence

fak is **not** the token engine. For a plain product completion the win is the
governance band — the audit trail, trace ids, budget accounting, the floor over your
app's tools, quarantine of poisoned results — not throughput; your tokens are still
served by the provider or engine behind `--base-url`. If all you want is cheaper or
faster tokens, this is not the tool. Full scope, claim by claim:
[`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).

---

## Cross-references

- [Two runtimes, one binary: gateway vs agent runtime vs client](../explainers/runtime-vs-client.md) — the naming behind the `guard` / `serve` / `serve --native` decision this guide resolves for the product persona.
- Epic [#3256](https://github.com/anthony-chaudhary/fak/issues/3256) (agent runtime) — the *opposite* shape: fak owns the agent loop server-side and your product drives it. This guide is the "keep your own call, front it" companion.
- [Adopter playbook §A — front a model](adopter-playbook.md#a--front-a-model-the-bare-serve-production-path) — the bare-`serve` production checklist (build/start, `/healthz`, auth-key env) this guide's step 1 abbreviates.
- [Customer-support playbook](customer-support-playbook.md) — a full vertical worked the same way (read-only support policy, threat model, rollout runbook).
- [Structured output through fak](structured-output.md) — if your tool-less call extracts JSON, the structured-output posture per client library.
- [Debugging a verdict](debugging.md) — why was my call denied/transformed? `fak preflight --explain`, then trace it across the live gateway.
- [Compatibility matrix](compatibility-matrix.md) — the exact base-URL key for whatever SDK or framework your product uses.
- [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — the capability-floor manifest schema; sibling [#5153](https://github.com/anthony-chaudhary/fak/issues/5153) (`fak policy --from-tools`) scaffolds it from your app's tool schemas.
