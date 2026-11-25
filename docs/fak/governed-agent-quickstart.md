---
title: "Governed Agent in 10 Minutes: Zero to a Kernel-Governed Agent, Offline"
description: "Start a local gateway, test its tool policy and inspect the audit record."
slug: governed-agent-quickstart
keywords:
  - governed agent
  - agent runtime quickstart
  - offline agent
  - default-deny floor
  - audit journal
  - fak serve
  - fak agent
date: 2026-07-23
---

# Check tool calls through a local gateway

Start a gateway with your tool policy, send a tool call, then inspect its verdict
and audit record. To run a language model on your own computer, follow the
[local model guide](../../GETTING-STARTED.md#local-model-setup).

## Step 1 — get the binary (~1–2 min)

Pick **one**. The rest of the page writes `fak`; on Windows use `.\fak.exe`.

```sh
# A. Prebuilt binary — no clone, no Go (recommended for just trying it)
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# B. Install with Go (the Go module is the repository root, so this resolves directly)
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# C. Build from a clone
git clone https://github.com/anthony-chaudhary/fak.git && cd fak
go build -o fak ./cmd/fak          # Windows: -o fak.exe
```

Confirm it runs:

```sh
fak version
```

---

## Step 2 — start the gateway runtime: floor + audit + cap, offline (~1 min)

Dump the default capability floor (a safe default-deny policy — it already denies
`shell_rm_rf` and `exfiltrate`), then start `fak serve` **with the audit journal on and a
token cap**, all offline. Leaving `--base-url` unset selects the offline mock planner, so
no key or model server is required:

```sh
# 1. The safe default floor (default-deny; allows a read-only + demo tool set)
fak policy --dump > policy.json

# 2. Start the gateway runtime, offline, with the floor + a durable audit journal + a token cap.
#    No --base-url  => offline mock planner (no key, no model, no GPU).
FAK_AUDIT_JOURNAL=audit.jsonl fak serve \
  --addr 127.0.0.1:8080 \
  --policy policy.json \
  --context-budget-tokens 200000        # a per-session token cap; exhaustion returns a reset directive
```

Confirm it is up (in a second terminal). **Real output:**

```sh
curl -s http://127.0.0.1:8080/healthz
# {"engine":"inkernel","model":"mock","ok":true,"planner":"mock"}
```

> **Cost caps, honestly.** `--context-budget-tokens` bounds a session's context/token
> spend, and `FAK_RATELIMIT_MAX_CALLS` bounds calls per window — both enforced offline. A
> hard *dollar* spend cap (the #3273 spend governor) exists in the kernel but its `fak serve`
> CLI flag is still open (#4859); attach it programmatically until then. See
> [server-config](server-config.md) for the full knob set.
>
> **Don't leave a long-lived `fak serve` running on a dev box** — it is a network server.
> Start it for this walkthrough, then stop it (Ctrl-C) when you reach the end.

For the **agent application runtime** — where fak *owns and runs the agent loop* itself
rather than adjudicating a proxied one — add `--native` and drive it with a `POST
/v1/messages`. `fak serve --native` runs the owned `agent.RunArm` loop with the same floor
and audit journal; see [Two runtimes, one binary](../explainers/runtime-vs-client.md) for
the distinction.

---

## Step 3 — run a governed session and watch a DENY (~1 min)

Send two adjudicated tool calls to the running gateway: one the floor **denies**, one it
**allows**. **Real output:**

```sh
# A dangerous call — the default floor DENIES it by structure, no model in the loop
curl -s -X POST http://127.0.0.1:8080/v1/fak/syscall \
  -H 'Content-Type: application/json' \
  -d '{"tool":"shell_rm_rf","arguments":{"path":"/"}}'
# {"verdict":{"kind":"DENY","reason":"POLICY_BLOCK","by":"monitor","disposition":"RETRYABLE"},
#  "result":{"status":"ERROR",...},"trace_id":"gw-2"}

# A safe call — allowed and executed against the offline in-kernel engine
curl -s -X POST http://127.0.0.1:8080/v1/fak/syscall \
  -H 'Content-Type: application/json' \
  -d '{"tool":"get_user_details","arguments":{"user_id":"mia"}}'
# {"verdict":{"kind":"ALLOW","by":"monitor"},"result":{"status":"OK",...},"trace_id":"gw-3"}
```

The `DENY` / `POLICY_BLOCK` verdict is the whole point: the dangerous call was refused **by
structure**, before any tool ran, with no model able to talk past it.

---

## Step 4 — read the audit tail (~30 s)

Every verdict landed in the durable, **hash-chained** audit journal you enabled in Step 3.
Read it. **Real output** (one JSON object per line; `prev_hash → hash` chains each record to
the one before it, so a tampered or dropped entry is detectable):

```sh
cat audit.jsonl
```

```jsonl
{"seq":1,"kind":"CONFIG_SWAP","tool":"floor","reason":"ok","by":"config-supervisor","config_swap":{"kind":"floor","source":"policy.json","digest":"sha256:09e398c9...","outcome":"ok"},"prev_hash":"","hash":"4cfa3075..."}
{"seq":2,"kind":"DENY","tool":"shell_rm_rf","trace_id":"gw-2","verdict":"DENY","reason":"POLICY_BLOCK","by":"monitor","args_label":"keys=path","prev_hash":"4cfa3075...","hash":"4d7f5bb3..."}
{"seq":3,"kind":"DECIDE","tool":"get_user_details","trace_id":"gw-3","verdict":"ALLOW","reason":"NONE","by":"monitor","args_label":"keys=user_id","prev_hash":"4d7f5bb3...","hash":"d8b9e294..."}
```

There it is, end to end, offline: the floor loaded (`CONFIG_SWAP`), the dangerous call
**DENIED** (`shell_rm_rf` → `POLICY_BLOCK`), the safe call **ALLOWED**, each record chained
to the last. That is a governed agent you can watch decide *and* an audit trail you can hand
to a reviewer — from a clean machine, with no key, model, or GPU.

Stop the server (Ctrl-C in the first terminal) when you are done.

---

## What you just proved

| Acceptance | Where you saw it |
|---|---|
| A governed session reaching a witnessed terminal state, offline | Step 2 (`task completed (booked) YES`) and Steps 4–5 (adjudicated verdicts) |
| A visible **DENY** of a dangerous call | Step 2 (`destructive op prevented`) and Step 4 (`POLICY_BLOCK`) |
| A durable **audit tail** | Step 5 (`audit.jsonl`, hash-chained) |
| A default-deny **floor** + a **cost cap**, offline | Step 3 (`fak policy --dump`, `--context-budget-tokens`) |

## Where to go next

- **What each runtime actually is:** [Two runtimes, one binary — gateway vs agent runtime vs
  client](../explainers/runtime-vs-client.md).
- **Front your own model / production serving:** [server quickstart](server-quickstart.md) ·
  [server config](server-config.md).
- **Wrap an existing harness (Claude Code, Codex) as a governed client:** [`fak manage`](../../README.md#manage-one-local-agent-fak-guard).
- **The guided first session (traces, policies, a real model):** [tutorial](tutorial.md).
- **The policy schema and refusal vocabulary:** [`POLICY.md`](../../POLICY.md).
