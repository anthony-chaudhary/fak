---
title: "fak FAQ — Operations, configuration, and deployment"
description: "Deep-dive FAQ theme split out of docs/FAQ.md; the essentials and the theme index live there."
---

# Operations, configuration, and deployment

Part of the [fak FAQ](../FAQ.md) — the essentials and every other theme are
indexed there.

Pair `fak` with a sandbox for blast radius, your own rate limiting for volume, and a tight allow-list scoped to safe tool names — `fak` is the governance gate, not the whole defense. `fak` structurally covers the syscall boundary (a default-deny capability floor that fails closed) and the context boundary (result quarantine that keeps poison out of attention), plus a payload-free audit trail. It does NOT contain what an admitted call does in the world, bound the arguments of a coarse allow-listed tool, or shed request-volume abuse. So run the actual tool execution inside a sandbox (for example E2B) to bound an over-broad admitted action, front the gateway with a reverse proxy or rate limiter for auth and volume, and keep exfil-shaped and destructive tools off the allow-list. `fak` makes the fail-closed decision affordable in-loop; defense-in-depth handles the effects it deliberately does not.

## Operations, configuration, and deployment

Running `fak` in production: authoring and reloading the policy floor, requiring authentication, what happens on a crash, and what to put around it.

## How do I author a capability floor for fak?

Run `fak policy --dump` to print the built-in default allow-list as a manifest, edit it to match the tools your agent should be permitted, then load it with `--policy floor.json`. The dump is the complete default floor, so you start from a working baseline and tighten rather than guess. A manifest has three core fields — `allow` (exact tool names), `allow_prefix` (read-only families like `read_`, `get_`, `search_`), and `deny` (tool name mapped to a refusal reason from the closed vocabulary). Validate any edit with `fak policy --check floor.json`, which prints the admitted floor and exits 1 on a bad file. The loaded manifest replaces the default floor wholesale; it is not merged on top of it.

```bash
fak policy --dump > floor.json
fak policy --check floor.json
fak serve --policy floor.json --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
```

## What happens if I make a typo in a policy manifest?

A typo is a hard error at load time, not a silently weakened floor — `fak` refuses to start or reload rather than run with a policy it could not parse. The manifest loader rejects unknown fields, so writing `allows` instead of `allow` fails with `invalid manifest: json: unknown field "allows"`. An unknown deny reason fails the same way, printing the offending value and the full list of the 17 valid reason codes. A bad posture, a malformed argument rule, or a different major schema version all hard-error too. Because policy load propagates a fatal error at startup, there is no fallback to a more permissive default.

## Does loading a policy add to the default allow-list or replace it?

A loaded manifest replaces the default floor entirely — it is the whole capability floor, not an overlay on the built-in default. This is why `fak policy --dump` gives you the complete default to edit: you start from the full floor and adjust it, so nothing is silently inherited that you did not put in the file. The same replace-not-merge rule applies to a runtime reload through the gateway. Round-tripping is stable, so `fak policy --dump` piped into `fak policy --check` validates exactly.

## How do I require an API key on a network-facing fak deployment?

Start the gateway with `--require-key-env VAR`, where `VAR` names an environment variable that holds the secret — the flag takes the variable *name*, never the secret value itself. Auth is off by default for loopback use, so this is the flag you add when binding somewhere reachable. Every route except `/healthz` then requires the token; clients send it as `Authorization: Bearer <token>` (OpenAI-style) or `x-api-key: <token>` (Anthropic-style), and both are compared in constant time over SHA-256 digests so neither the bytes nor the length leak. If the named variable is set but empty, the gateway refuses to start (exit 2) rather than come up unprotected.

```bash
export FAK_TOKEN=$(openssl rand -hex 32)
fak serve --addr 0.0.0.0:8080 --require-key-env FAK_TOKEN --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
```

## Why does --require-key-env take an environment variable name instead of the key itself?

`fak` reads the secret from the named environment variable so the key never appears in the command line, the flag list, or process listings where it would be visible to other users. You pass `--require-key-env FAK_TOKEN` and put the actual secret in `$FAK_TOKEN`; the gateway resolves it at startup. The same pattern applies to the upstream provider key via `--api-key-env`, which names the variable holding your real provider key that `fak` forwards upstream. A named-but-empty required key variable is treated as a misconfiguration and fails closed at startup.

## Can I update the policy floor without restarting fak?

Yes — `POST /v1/fak/policy/reload` (no body) re-reads the manifest from its source and replaces the floor in place, so you can tighten or loosen the allow-list on a running gateway. The reload is replace-not-merge, exactly like the initial load: the floor is rebuilt from the file, not patched. The endpoint returns `{reloaded, source, summary}` on success. It answers `404` if the deployment was not started with a policy to reload, and `400` (with the error message) if the new manifest fails to parse — a broken reload leaves the running floor untouched rather than weakening it.

## What happens to the policy floor and quarantine state when fak crashes and restarts?

On restart the capability floor reloads from its manifest on disk, so a crash never leaves the gate silently bypassed — there is no permissive fallback path. Policy load is fatal on error, so the process either comes up with the floor you authored or does not come up at all. The in-memory quarantine and taint ledger is a different matter: the live result-screening state (the held and cleared maps inside the context-MMU) lives in process memory with no disk backing, so it resets on restart. That is fail-safe rather than a leak, because the bytes a quarantine held were never in model context to begin with. If you need quarantine decisions to survive a process boundary, persist the session with `fak recall`, which writes a durable core image that re-screens every page on reload.

## Should I run fak under a process supervisor like systemd?

Yes — `fak serve` is a single static binary with a two-module `golang.org/x` dependency set, which makes it a clean fit for systemd, a container runtime, or any supervisor that restarts a process on exit. Because the floor reloads from its manifest on every startup and policy-load errors are fatal, a supervised restart re-establishes the same gate deterministically rather than drifting open. The binary binds its listener synchronously before marking itself ready, so a bind failure surfaces immediately instead of leaving a half-started service. Pass the secret and the policy by environment and flag (`--require-key-env`, `--policy`) so the unit file carries configuration, not secrets in the command line.

## Are the /metrics and /debug/vars endpoints exposed without authentication?

They follow the gateway's auth policy: when you run with `--require-key-env`, both `/metrics` and `/debug/vars` require the bearer token, and only `/healthz` stays open. With auth off (the loopback default) they are reachable like any other route. `/metrics` serves Prometheus exposition and `/debug/vars` serves a single JSON snapshot of the same gateway, runtime, kernel, and metrics view. If you scrape metrics over a network, gate them behind auth and treat `/healthz` as the only intentionally public probe.

## What does fak bind to by default, and is that safe to leave?

`fak serve` defaults to `127.0.0.1:8080` — loopback only — so out of the box it is reachable only from the same host and auth is off for low-friction local use. That default is safe to leave on a developer machine. If you bind to a non-loopback address without setting `--require-key-env`, the gateway prints a loud warning that it is reachable with no key, because that combination is almost always a mistake. The intended progression from laptop to fleet is adding flags (`--policy`, `--require-key-env`) rather than swapping components.

## How do I verify a policy floor before deploying it, without a model or network?

Use `fak policy --check floor.json` to validate the manifest and print the admitted floor, and `fak preflight --tool NAME --args JSON --policy floor.json` to get the exact verdict a single call would receive — both run offline with no model, key, or GPU. `--check` enforces the closed refusal vocabulary and exits 1 on a bad file, so it composes as a CI gate. `preflight` is the per-call oracle: it prints `verdict=… reason=… by=monitor`, and `--explain` traces each rung. This lets you prove that a tool you expect denied (say, `refund_payment`) returns `DENY` and a read tool returns `ALLOW` before any traffic flows.

```bash
fak policy --check floor.json
fak preflight --tool refund_payment --args '{}' --policy floor.json --explain
```

## How does fak return a policy denial over HTTP — is it an error status?

A policy denial is a successful `200` carrying the verdict as a value, never a non-2xx error status. HTTP error codes are reserved for malformed requests, auth failures, and upstream faults — a `401` for a bad key, a `502` when the upstream provider is unreachable — so your client never has to treat "the kernel said no" as an exception. On the chat and messages wires, denied tool calls are dropped from the response and the surviving calls are returned, with the full per-call verdicts in the `fak` response extension (and, for Claude Code, also prepended as a `[fak]` text note). This is the deny-as-value contract: a refusal is in-band data, not a transport failure.

## How do I turn on a durable, tamper-evident audit log?

Set the `FAK_AUDIT_JOURNAL` environment variable to a file path; the durable decision journal is opt-in and inert until you do. Once enabled, `fak` appends one hash-chained JSONL row per decision (`DECIDE`, `DENY`, `QUARANTINE`, and even `VDSO_HIT`), and the chain is tamper-evident — any after-the-fact byte mutation breaks verification at the first altered link. The journal records tool names, trace IDs, verdicts, reasons, and content *digests* only; it never materializes the argument or result bytes, so it leaks no payload. Separately, the gateway always emits a trace-correlated stdout access log that records names and verdicts but never arguments or result content. The `/v1/fak/events` route reads the journal back and returns `404` when the variable is unset.

## Can I tune fak's HTTP timeouts and request size limits for slow local inference?

