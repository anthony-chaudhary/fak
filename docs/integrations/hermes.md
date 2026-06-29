---
title: "Hermes Agent + fak: governed self-hosted agent"
description: "Wire fak as a tool-governance layer for Hermes Agent, NousResearch's open-source autonomous agent. Every tool call — including execute_code — crosses a default-deny capability floor, with poisoned tool results quarantined out of context."
---

# Hermes Agent + fak Integration Guide

[Hermes Agent](https://github.com/nousresearch/hermes-agent) is NousResearch's
open-source, self-hosted autonomous agent — it executes code, searches the web, manages
files, and talks over a dozen messaging platforms, on whatever LLM backend you point it at.
It is **model-agnostic and OpenAI-compatible**: it calls `POST /v1/chat/completions` with
OpenAI `tools[]` function-calling, so it drops behind `fak` by repointing one base URL.

This guide puts `fak` between Hermes Agent and its model. Every tool call the agent
proposes — a shell command, a file write, an `execute_code` block — is adjudicated by the
kernel before it runs: dangerous calls are denied by structure, malformed calls are
repaired, and poisoned tool results are quarantined before they re-enter the agent's
context.

## Overview

```
┌────────────────┐   OpenAI Chat Completions   ┌────────────────────────┐
│  Hermes Agent  │ ──────────────────────────▶ │  fak serve (gateway)   │
│   (hermes CLI) │ ◀──────── response ───────  │  adjudicates tools     │
└────────────────┘                             └────────────────────────┘
        ▲                                                  │
        │ OPENAI_BASE_URL / OPENAI_API_KEY                 │
        │ (or ~/.hermes/config.yaml model.base_url)        ▼
        │                                          ┌───────────────┐
        │                                          │  Local Model  │
        │                                          │ or Cloud API  │
        │                                          └───────────────┘
```

**The gateway sits between Hermes Agent and the model:**

- **Hermes → fak:** Hermes Agent sends a chat request carrying its proposed tool calls.
- **fak kernel:** Adjudicates each proposed call (allow, deny, transform, quarantine).
- **fak → model:** Forwards only the admitted (or repaired) calls upstream.
- **fak → Hermes:** Returns results, with the kernel's decisions applied.

**Result:** Hermes Agent keeps its persistent memory, its self-improving skills, and its
40+ built-in tools — but the kernel blocks destructive commands, prevents self-modification,
and contains untrusted tool results.

---

## Prerequisites

### 1. Install fak

```bash
# From the repo (the Go module is the repo root)
git clone https://github.com/anthony-chaudhary/fak && cd fak
go build -o fak ./cmd/fak

# Or via the installer
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Verify:
```bash
./fak version
```

### 2. Install Hermes Agent

Follow the [Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/). Once
installed, the `hermes` CLI is on your `PATH`:

```bash
hermes --version
```

### 3. Choose your upstream model

`fak` can serve Hermes Agent in two modes:

- **Proxy mode:** `fak` forwards to an external model (OpenAI, Anthropic, Ollama, vLLM,
  SGLang, llama.cpp, or any OpenAI-compatible endpoint).
- **In-kernel mode:** `fak` serves its own fused GGUF model (`--gguf`), no second process.

For full agentic quality, proxy mode in front of a frontier model is the default; in-kernel
mode is the no-network, no-key dogfood path.

---

## Quick Start: one command

The fastest way to put the kernel in front of the Hermes Agent you already run is the
`fak guard` verb. It starts the gateway in-process on a private loopback port, injects the
base URL **into the child process only**, and proxies to your real upstream:

```bash
export OPENAI_API_KEY=sk-...                  # or point --base-url at a local model
fak guard --provider openai --api-key-env OPENAI_API_KEY -- hermes
```

`fak guard`:

1. Starts the gateway in-process on `127.0.0.1:<random-port>`.
2. Loads a secure default capability floor (print it with `fak guard --dump-policy`,
   override with `--policy FILE`).
3. Injects `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` (and `OPENAI_API_BASE`, the same
   value) into the `hermes` child only — your shell and `~/.hermes/config.yaml` are
   untouched.
4. Proxies every chat turn to your upstream, adjudicating each proposed tool call first.
5. Tears the gateway down when Hermes exits and prints what the kernel decided.

> **The provider is autodetected.** `fak guard` recognizes `hermes` as an OpenAI-wire agent
> (the same table that maps `codex`/`opencode`/`aider`), so a bare
> `fak guard -- hermes` already picks `--provider openai` and injects `OPENAI_BASE_URL` on
> its own. Name `--provider openai` explicitly if you prefer to be unambiguous, or to wrap a
> launcher whose basename is not `hermes`.

### Recorded live witness

The OpenAI-wire guard path has a live gateway-transited witness in
[`experiments/agent-live/openai-wire-seat-guard-live-witness-2026-06-29.json`](../../experiments/agent-live/openai-wire-seat-guard-live-witness-2026-06-29.json).
The run used `opencode` as the issue-approved OpenAI Chat Completions fallback because the
`hermes` CLI was not installed on that Windows host. It records a real child result, `200`
rows on `route=/v1/chat/completions` in the guard log, a direct placeholder-key `401`
against the upstream, and a hash-chained `DECIDE` row in `FAK_AUDIT_JOURNAL`.

### Local model: no key, no network, one command

Run a local GGUF in-kernel as Hermes Agent's upstream — the whole stack (model + agent +
kernel floor) in one process:

```bash
fak guard --gguf qwen2.5-coder:7b -- hermes
```

The GGUF downloads from Hugging Face on first run (cached in `~/.cache/fak-models/`), loads
in-kernel, and Hermes Agent connects over the in-process gateway. Your data never leaves the
box after the initial pull. See [`fak ls`](../../README.md) for the available aliases.

---

## Manual wiring (without `fak guard`)

If you run Hermes Agent and `fak serve` as separate long-running processes:

### Step 1: Start the fak gateway

```bash
./fak serve \
  --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5-coder:7b \
  --policy hermes-policy.json
```

Verify health:
```bash
curl http://127.0.0.1:8080/healthz
# {"ok":true,"model":"qwen2.5-coder:7b","engine":"inkernel"}
```

### Step 2: Point Hermes Agent at fak

Hermes Agent reads the standard OpenAI env vars, so the simplest wiring is:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="fak-local"
hermes
```

(The `/v1` suffix matters — OpenAI-compatible clients append `/chat/completions`, so a bare
host would 404.)

**Or persist it in the config file.** In `~/.hermes/config.yaml`:

```yaml
model:
  provider: custom
  model: "qwen2.5-coder:7b"
  base_url: "http://127.0.0.1:8080/v1"
```

Keep the secret in `~/.hermes/.env` (`OPENAI_API_KEY=fak-local`). The provider-setup wizard
`hermes model` walks you through the same custom-endpoint fields interactively.

---

## A capability floor for Hermes Agent

A capability floor is a reviewable JSON allow-list — which tools may run, in git, not a code
edit. Start from the built-in default:

```bash
./fak policy --dump > hermes-policy.json
```

Hermes Agent ships 40+ built-in tools, including **`execute_code`** (which collapses a
multi-step pipeline into one inference call — powerful, and exactly the call worth gating).
A floor that allows day-to-day work but refuses the destructive and self-modifying classes:

```json
{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow_prefix": [
    "read_",
    "get_",
    "search_",
    "list_",
    "web_",
    "file_"
  ],
  "allow": [
    "execute_code",
    "write_file",
    "edit_file"
  ],
  "deny": {
    "delete_file": "POLICY_BLOCK"
  },
  "self_modify_globs": [
    ".git/",
    ".hermes/",
    ".env",
    "id_rsa"
  ],
  "arg_rules": [
    {
      "tool": "execute_code",
      "arg": "code",
      "deny_regex": "rm\\s+-rf|sudo|os\\.system\\(|subprocess\\.|:(){:|:&};:",
      "reason": "POLICY_BLOCK"
    },
    {
      "tool": "read_file",
      "arg": "path",
      "deny_regex": ".*\\.env$",
      "reason": "SECRET_EXFIL"
    }
  ]
}
```

Two things to note for Hermes Agent specifically:

- **`execute_code` is allow-listed but arg-gated.** Allowing the tool while a `deny_regex`
  refuses `rm -rf`, `sudo`, a fork bomb, and the obvious shell-escape calls is the useful
  posture — you keep the programmatic-tool-calling speed-up without handing it a blank shell.
- **`.hermes/` is a self-modify target.** Hermes Agent's self-improving skills and config
  live there; blocking writes into it stops the agent from rewriting its own guardrails.

Validate before using:
```bash
./fak policy --check hermes-policy.json
```

Check any single call offline, without launching the agent:
```bash
./fak preflight --explain \
  --tool execute_code \
  --args '{"code":"import os; os.system(\"rm -rf /\")"}' \
  --policy hermes-policy.json
# verdict=DENY reason=POLICY_BLOCK
```

---

## Quarantine for external tool results

Hermes Agent pulls from the web and 16+ messaging platforms — exactly the untrusted inputs a
prompt-injection rides in on. The kernel contains those results on the **result side**: when a
tool result comes back poisoned or secret-shaped, the result-admit fold **quarantines** it —
pages it out before it re-enters the agent's context — so the model never reads it.

This is **automatic, not a flag you flip.** Quarantine is part of the result-admit stack the
gateway runs on every served turn (the context-MMU secret/poison check plus the IFC taint
stamp). It is in effect whenever `fak serve` / `fak guard` fronts the agent — there is no
`--quarantine` switch to set or forget. What you *do* control is **what counts as poisoned**:
the secret-shaped detector and the `SECRET_EXFIL` arg rules in your capability floor (above)
decide which results get quarantined. To watch it fire, see
[`fak_kernel_quarantines_total`](#health-and-metrics) and the `quarantined` count in the guard
exit summary.

> **`--vdso` is a different mechanism — not the quarantine toggle.** `--vdso` is the **vDSO
> dedup fast path** (content-addressed caching that speeds repeat turns); it defaults to `true`
> and drives only `fak_kernel_vdso_hits_total`. It neither enables nor disables quarantine.
> (An earlier version of this section wired `--vdso=true` into a "turn quarantine on" example —
> that conflated two unrelated features, and since `--vdso` already defaults on, the example
> changed nothing.)

### Proxy seat vs. local `--gguf`

On the **proxy** seat — `fak serve` / `fak guard` in front of an upstream model, this guide's
documented default — the quarantined result is paged out of the agent's context before the
model reads it, so the poison never enters the turn. The in-kernel **KV poison-evictor**
(dropping the local KV prefix the result would have populated) is the **`--gguf` local-model
path** only: on a proxy seat the model lives upstream, so there is no local KV prefix to evict.
Both seats stop the poisoned result from reaching the model; they differ only in whether
there is also a local cache to drop.

---

## Monitoring and debugging

### Health and metrics

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics
```

Key metrics:
- `fak_gateway_time_to_ready_seconds` — startup time
- `fak_gateway_operations_total{verdict="DENY"}` — denied calls (by reason label)
- `fak_kernel_quarantines_total` — quarantined results

### What the guard session reports

On exit, `fak guard` prints the kernel's decisions for the session:

```
fak guard: 31 kernel decision(s) — 27 allowed, 2 denied, 1 repaired, 1 quarantined, 0 deferred
  blocked: POLICY_BLOCK     x2
```

### Debugging a denied call

Reproduce any verdict offline:

```bash
./fak preflight --explain \
  --tool write_file \
  --args '{"path":".hermes/config.yaml","content":"..."}' \
  --policy hermes-policy.json
# verdict=DENY reason=SELF_MODIFY
```

---

## Troubleshooting

### Hermes Agent can't reach the gateway

1. Verify `fak` is up: `curl http://127.0.0.1:8080/healthz`.
2. Check the base URL ends in `/v1`:
   ```bash
   echo $OPENAI_BASE_URL   # should be http://127.0.0.1:8080/v1
   ```
   A bare host (no `/v1`) makes the OpenAI client POST to `<host>/chat/completions`, which
   the gateway (serving `/v1/chat/completions`) answers with a 404.
3. If `OPENAI_BASE_URL` isn't picked up, set `model.base_url` in `~/.hermes/config.yaml`
   instead (or run `hermes model` and configure a custom endpoint), then bind `fak serve` to
   a fixed `--addr` so the config URL is stable.

### Everything is denied

The default posture is `fail_closed` — tools not on `allow`/`allow_prefix` are refused.
Confirm your Hermes tool names match the floor:
```bash
./fak preflight --tool read_file --args '{"path":"README.md"}' --policy hermes-policy.json
```

### Slow first response

Expected on large local models — the agent prompt is large and the first turn has no cache.
`--vdso=true` (content-addressed caching) speeds subsequent turns.

---

## Cross-references

- **Integration index**: [README.md](README.md) — the universal recipe and which-agent routing
- **Compatibility matrix**: [compatibility-matrix.md](compatibility-matrix.md) — the full sourced field survey
- **Aider guide**: [aider.md](aider.md) — the closest sibling (another OpenAI-wire CLI agent)
- **Policy schema**: [../../POLICY.md](../../POLICY.md) — authoring capability floors
- **Hermes Agent docs**: [https://hermes-agent.nousresearch.com/docs/](https://hermes-agent.nousresearch.com/docs/)
- **fak architecture**: [../../ARCHITECTURE.md](../../ARCHITECTURE.md) — kernel internals

---

## License

Apache-2.0
