---
title: "Governed Agent in 10 Minutes: Zero to a Kernel-Governed Agent, Offline"
description: "From a clean machine to a running, kernel-governed agent — offline, no key, no GPU. One command starts the gateway runtime with a default-deny floor, an audit journal, and a token cap; a POST runs a governed session; you watch a dangerous call get DENIED and read the tamper-evident audit tail."
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

# Governed agent in 10 minutes (offline)

Getting *an* agent running is easy. Getting a **governed** one — an agent loop whose every
tool call passes through a kernel that can deny it, on a durable audit record — is the part
that usually takes a project. This page does it end to end **on a clean machine, offline,
with no API key, no model download, and no GPU**, in under 10 minutes (the fast path is
under 5).

> **TL;DR.** Get the binary, then:
>
> ```sh
> fak agent --offline                                  # a full governed session, one command
> ```
>
> You watch the kernel deny a destructive call and quarantine a prompt injection while the
> task still completes. The rest of this page does the same thing **server-side** —
> `fak serve` with a default-deny floor + audit journal + token cap — and reads the DENY out
> of the tamper-evident audit tail.

**Audience:** you have never run `fak` and want to see a governed agent decide, not just
read that it can. Every output block below is real, unedited terminal output from a clean
build.

- **Time:** ~2 min for the one-command proof; ~10 min total including the server-side path.
- **Prereqs:** [Go 1.26+](https://go.dev/dl/) *or* a
  [prebuilt binary](https://github.com/anthony-chaudhary/fak/blob/main/INSTALL.md). Nothing
  else — no key, no model, no GPU, no network after the install.
- **Which runtime is this?** The gateway runtime and the agent application runtime, offline.
  If those words are new, read [Two runtimes, one binary](../explainers/runtime-vs-client.md)
  first — it names what `fak serve`, `fak serve --native`, and `fak manage` each are.

---

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

## Step 2 — a governed agent in one command (~1 min)

The shortest path to a *complete governed session* needs no server. `fak agent --offline`
drives fak's own agent loop against a deterministic mock planner (no key, no network),
routing every tool call through the kernel:

```sh
fak agent --offline
```

**Real output** (abridged to the governance rows — the full table also prints token and
turn economics):

```
== fak agent: turn-use vs now ==
seam        : OFFLINE (deterministic mock planner)

metric                        now(base)          fak
--------------------------   ----------   ----------
in-syscall repairs                  n/a            1
adjudicator denies                  n/a            1
injection in context                YES           no
destructive op executed             YES           no
task completed (booked)             YES          YES

HEADLINE
  poisoned result blocked   : YES
  destructive op prevented  : YES

report written: agent-report.json
```

That is a governed agent reaching a witnessed terminal state (`task completed (booked)
YES`) while the kernel **denied a destructive op** (`adjudicator denies 1`) and
**quarantined a prompt injection** (`injection in context YES → no`) — offline. The
per-session detail is in `agent-report.json`.

This is enough to *see* governance. The next step is the shape you actually deploy: a
**server** with a durable audit trail you can hand to an auditor.

---

## Step 3 — start the gateway runtime: floor + audit + cap, offline (~1 min)

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

## Step 4 — run a governed session and watch a DENY (~1 min)

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

## Step 5 — read the audit tail (~30 s)

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
