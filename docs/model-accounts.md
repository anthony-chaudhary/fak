---
title: "fak account switcher: bring your own model accounts"
description: "How fak's account switcher routes each aspect of a request to a chosen account and provider wire — OpenAI, Codex, Anthropic, Gemini, or local — so you can mix providers per tool call, step, or ensemble half."
---

# The account switcher: bring your own accounts, mix and match

`fak route` decides *which* model — or which ensemble of models — serves an aspect of
a request. The account switcher decides *whose account* runs that model, over which
provider's wire. Together they let you point fak's routing at your own OpenAI, Codex,
Anthropic, Gemini, and local accounts, and mix providers at any level: the cheap
aspect to a local model, the hard reasoning step to your OpenAI account, and a
two-model guard ensemble whose halves run on two *different* accounts.

It is the generic, in-product sibling of the fleet account switcher (`fak fleet-accounts`,
the native Go successor to the legacy `tools/fleet_accounts.py` shim): provider-neutral,
credential-safe, and composable with the routing spine. It lives in
`internal/modelroute/account.go` (pure, stdlib-only, the same package as the routing
decision and the cost lens).

## The pieces

**Account** — the switcher unit. A named credential set for one provider. Two
accounts can target the *same* provider kind (`openai-personal` and `openai-work`),
which is the switch: you choose which credential serves a model. An account names the
env var that holds its key (`cred_env`); the secret itself never lives in the file.

**Binding** — maps one routed model id (a `small`/`large`/`guard-a` from your routing
manifest, or a plan's scout) to an account plus the upstream model name to send on the
wire. The routed id is an abstract tier label; the upstream model is the
provider-specific name.

**Roster** — the declarative, version-tagged JSON manifest holding your accounts and
bindings, plus a default account for any unbound id. It loads the same way the routing
manifest does: `DisallowUnknownFields`, fail-loud validation, round-trips
`--accounts-dump` ↔ `--accounts-check`.

**Target** — the resolved destination for one model id: the account, the provider
kind, the concrete base URL, the credential env-var name, and the upstream model. The
dispatch layer turns a Target into a live planner; the resolver itself does no I/O.

```jsonc
{
  "version": "fak-accounts/v1",
  "accounts": [
    { "id": "local",           "kind": "local",            "base_url": "http://127.0.0.1:11434/v1" },
    { "id": "openai-personal", "kind": "openai",           "cred_env": "OPENAI_API_KEY" },
    { "id": "openai-work",     "kind": "openai",           "cred_env": "OPENAI_WORK_API_KEY" },
    { "id": "codex",           "kind": "openai-responses", "cred_env": "OPENAI_API_KEY" },
    { "id": "claude-sub",      "kind": "anthropic",        "cred_env": "CLAUDE_CODE_OAUTH_TOKEN" }
  ],
  "default": "openai-personal",
  "bindings": [
    { "model": "small",   "account": "local",           "upstream_model": "llama3.2" },
    { "model": "guard-a", "account": "openai-work",      "upstream_model": "gpt-5.5" },
    { "model": "guard-b", "account": "claude-sub",       "upstream_model": "claude-opus-4-6" }
  ]
}
```

A full example is `examples/model-accounts.example.json`.

## Two account layers, one vocabulary

This page is about **provider accounts**: which credential/env-var serves a routed model.
Claude Code subscription seats add a second layer: **config-home accounts** (`CLAUDE_CONFIG_DIR`
homes such as `~/.claude-gem8-seat`) that may be logged in, logged out, tombstoned, disabled, or
duplicated onto the same rate-limit bucket. That lifecycle is owned by `fak accounts`, not by the
provider roster.

Use `fak accounts status --json` for the observable config-home login report. It emits
`fak.accounts.login.v1`: one closed `status` per seat (`ready`, `needs_login`, `missing_dir`,
`disabled`, `tombstoned`), `can_serve`, roles, warnings (`duplicate_account_bucket`,
`split_setup_token`, `unverified_account`), and a next action. The human `fak accounts list` table
shows the same status in its `LOGIN` column, and `fak accounts sync` materializes
`login_status` plus `can_serve` into the generated dos/job roster rows. That keeps the account
switcher from guessing at login readiness from directory names or scattered credential booleans.
The same vocabulary is carried through `fak fleet-accounts roster/resolve`, `fak dispatch
tick/wave`, `fak accounts launch`, `fak accounts next`, and `fak guard` auth warnings, so callers
can gate on `can_serve` and surface the closed login status instead of re-deriving readiness from
raw credential files.

## Mix and match at any level

There is no per-aspect special case. The routing decision produces model ids for the
whole request, a tool call, a reasoning step, a scout probe, or each member of an
ensemble — and the roster binds *every* one of them by id. So an ensemble can span
accounts and providers (`guard-a` on your OpenAI work account, `guard-b` on your
Anthropic subscription), and the cheap scout-classify probe can switch accounts
independently of the members it gates.

```
fak route --manifest examples/model-routing.example.json \
          --aspect tool_call --tool refund_payment \
          --accounts examples/model-accounts.example.json
```

prints the routed guard ensemble *and* the account each member resolves to.

## Credentials are references, never secrets

An account names an env var (`cred_env: OPENAI_API_KEY`); the key is read with
`os.Getenv` only at dispatch time, in the layer that builds the planner. The roster, the
resolved Target, the `EngineRoute`, and every `--accounts` / `--accounts-dump` output
carry the *name*, never the value. Validation enforces this: a `cred_env` that is not a
valid env-var name — a pasted `sk-ant-…` key, a `Bearer …` string, an `X=Y` pair — is
rejected at the boundary, so a real key cannot end up committed in a roster.

Use `fak route --accounts-status roster.json` to inspect provider-account readiness in the
current shell. It emits `fak.modelroute.accounts.v1`: local accounts are `not_required`; remote
accounts are `ready` only when their named env var is present and non-empty, otherwise
`needs_credential` with the env var to set. This is intentionally an environment observation, not
a live API probe or billing claim, and it still prints only env-var names.

## Residency is declared, not guessed

fak's residency floor denies a tenant-scoped or sensitivity-tagged payload bound for a
*remote* engine. It reads the route string written to `abi.ToolCall.Engine`.
`Target.EngineRoute()` stamps that string with a structural prefix taken from the
account's kind: a local target is `local:…` (the floor reads it as on-box and exempt),
a remote target is `<kind>:…` where the kind is one of the keywords the floor
recognizes. Locality has one source of truth — `kind == local` — so it can never
disagree with a second flag the floor might trust. Validation forbids a local account
from carrying a non-loopback base URL, which would otherwise emit a `local:` route
while the bytes egress off-box. A cross-package test
(`internal/engine/account_residency_test.go`) pins that the floor and the switcher
agree for every provider kind, so a future kind that the floor could not classify is a
build-time failure, not a silent fail-open.

## A note on codex and Anthropic subscriptions

Codex's native wire is the OpenAI Responses API, so bind it to `kind: openai-responses`,
not plain `openai`. (`fak guard -- codex` autodetects the chat-completions `openai`
wire today; the roster lets you pick the Responses wire explicitly.)

An Anthropic Pro/Max subscription token (`sk-ant-oat…`) rides as
`Authorization: Bearer` plus the `oauth-2025-04-20` beta header, not as `x-api-key`.
The roster declares the account and the env var that holds the token; the Bearer-vs-key
header choice is made downstream by the Anthropic adapter at dispatch.

## What is shipped, and what is not

**Shipped:** the resolver. A roster resolves a routed plan — the scout and every
member, in member order — to per-member Targets, with fail-loud validation and the
residency-honest `EngineRoute`. The `fak route --accounts` path prints the binding.
This is pure and deterministic, witnessed by `go test ./internal/modelroute` and the
cross-package residency test.

**Deferred:** live multi-account dispatch. v1 *resolves and prints* a binding; it does
not yet send a request over the wire to your chosen account. Building the planners from
the Targets and running an ensemble's members is the additive dispatch wiring, the same
shipped-vs-deferred split the routing spine uses for the ensemble fold versus its
execution. (Per-call routing into dispatch — writing the routed id to
`abi.ToolCall.Engine` before submit — has landed for the single-model case; the
account-resolved route is the next refinement of what that field carries.)

**Out of scope for v1:** this is a declarative, static binding resolver. It has no
per-account rate-limit, capacity, health, or failover signal. It is not the fleet's
load-aware switcher — it is the portable, reviewable account map that a switcher would
consult.
