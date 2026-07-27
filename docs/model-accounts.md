---
title: "fak account switcher: bring your own model accounts"
description: "How fak's account switcher routes each aspect of a request to a chosen account and provider wire — OpenAI, Codex, Anthropic, Gemini, DeepSeek, or local — so you can mix providers per tool call, step, or ensemble half."
---

# The account switcher: bring your own accounts, mix and match

`fak route` decides *which* model — or which ensemble of models — serves an aspect of
a request. The account switcher decides *whose account* runs that model, over which
provider's wire. Together they let you point fak's routing at your own OpenAI, Codex,
Anthropic, Gemini, Groq, DeepSeek, and local accounts, and mix providers at any level: the cheap
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
OpenAI-compatible providers such as Groq are represented as `kind: "openai"` plus their
provider `base_url`, so the wire adapter stays generic while the account id names whose
credential is being used.

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
    { "id": "claude-sub",      "kind": "anthropic",        "cred_env": "CLAUDE_CODE_OAUTH_TOKEN" },
    { "id": "july6netra_groq", "kind": "openai",
      "base_url": "https://api.groq.com/openai/v1", "cred_env": "FAK_GROQ_API_KEY",
      "requests_per_minute": 30, "requests_per_day": 1000,
      "tokens_per_minute": 8000, "tokens_per_day": 200000 },
    { "id": "july6netra_groq_compound", "kind": "openai",
      "base_url": "https://api.groq.com/openai/v1", "cred_env": "FAK_GROQ_API_KEY",
      "requests_per_minute": 30, "requests_per_day": 250 },
    { "id": "deepseek",        "kind": "deepseek",         "cred_env": "DEEPSEEK_API_KEY",
      "context_tokens": 1000000, "max_output_tokens": 384000 },
    { "id": "deepseek-anthropic", "kind": "anthropic",
      "base_url": "https://api.deepseek.com/anthropic", "cred_env": "DEEPSEEK_API_KEY",
      "context_tokens": 1000000, "max_output_tokens": 384000 }
  ],
  "default": "openai-personal",
  "bindings": [
    { "model": "small",   "account": "local",           "upstream_model": "llama3.2" },
    { "model": "guard-a", "account": "openai-work",      "upstream_model": "gpt-5.5" },
    { "model": "guard-b", "account": "claude-sub",       "upstream_model": "claude-opus-4-6" },
    { "model": "qwen36-groq", "account": "july6netra_groq", "upstream_model": "qwen/qwen3.6-27b" },
    { "model": "groq-compound", "account": "july6netra_groq_compound", "upstream_model": "groq/compound" },
    { "model": "deepseek-pro", "account": "deepseek", "upstream_model": "deepseek-v4-pro" },
    { "model": "deepseek-flash", "account": "deepseek", "upstream_model": "deepseek-v4-flash" },
    { "model": "deepseek-chat-compat", "account": "deepseek", "upstream_model": "deepseek-chat",
      "compatibility_only": true, "deprecated_after_utc": "2026-07-24 15:59 UTC",
      "deprecated_alias_for": "deepseek-v4-flash non-thinking mode" }
  ]
}
```

A full example is `examples/model-accounts.example.json`.

## Two account layers, one vocabulary

This page is about **provider accounts**: which credential/env-var serves a routed model.
Coding-agent subscriptions add a second layer: **config-home accounts**. Claude uses
`CLAUDE_CONFIG_DIR` homes such as `~/.claude-gem8-seat`; Codex uses `CODEX_HOME` homes such as
`~/.codex` or `~/.codex-work`. `fak fleet-accounts` discovers both, derives their non-secret
provider-account identity, collapses duplicate homes on the same rate-limit bucket, and offers
only homes whose credential state is ready. The default Codex picker profile is
`gpt-5.6-sol` with `model_reasoning_effort=xhigh`.

The authored lifecycle registry behind `fak accounts` remains Claude-specific today; its seat
names and tombstones are therefore applied only to Claude rows. Codex rows take lifecycle/login
truth from their own home (`auth.json`/`config.toml`) and never inherit a same-named Claude seat
such as `default`.

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

## API-host probe bridge

The same model-account roster can feed the API-host no-spend probes:

```bash
fak api-host readiness  --from-model-accounts examples/model-accounts.example.json
fak api-host acceptance --from-model-accounts examples/model-accounts.example.json
```

`readiness` converts only OpenAI-compatible or local accounts into `/models` probe targets
(`openai`, `openai-responses`, `xai`, `deepseek`, and `local`). Native provider accounts are not guessed into
an OpenAI-shaped probe. `acceptance` keeps every account with a probeable base URL visible:
OpenAI-compatible/local accounts can become `READY_FOR_LIVE_BRIDGE_RUN`, while native
Anthropic/Gemini accounts are reported as `WIRE_SUPPORTED_UNPROBED`. Model hints come from the
roster bindings, so the API-host report stays tied to the same abstract model ids the route policy
uses.

Groq Qwen3.6 uses the OpenAI-compatible base `https://api.groq.com/openai/v1`, credential
env var `FAK_GROQ_API_KEY`, and upstream model slug `qwen/qwen3.6-27b`. The example records
the current account/model limits as metadata: 30 requests/minute, 1,000 requests/day,
8,000 tokens/minute, and 200,000 tokens/day. Groq Compound uses the same base and env var
under the separate account id `july6netra_groq_compound`, bound as the lower-quality
`groq-compound` target to upstream slug `groq/compound`; its metadata is request-only:
30 requests/minute and 250 requests/day, with no token-minute or token-day cap recorded.
The Claude dogfood launcher separately caps the outbound `max_tokens` field at 8192 for
`groq/compound`, because Groq rejects larger per-response output budgets. Those values
are advisory metadata for routing and launchers; the v1 resolver does not throttle
requests.

DeepSeek V4 uses `deepseek-v4-pro` / `deepseek-v4-flash` on the OpenAI-compatible base
`https://api.deepseek.com`; its Anthropic-compatible profile uses
`https://api.deepseek.com/anthropic`. The legacy `deepseek-chat` and `deepseek-reasoner`
aliases are accepted only when marked `compatibility_only`, with the documented retirement
metadata `2026-07-24 15:59 UTC`, so a new roster does not silently bind to an alias that is
about to disappear.

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
build-time failure, not a silent fail-open. The same file pins the tier-1 mirror
(`modelroute.IsRemoteRoute`) against the enforcing floor over one corpus, because
`internal/modelroute` sits below `internal/engine` and must keep its own copy of the
on-box family list. See *Placement zones* below for the third rung the binary
local/remote split does not name.

## Placement zones: your box, your company's boxes, someone else's

Locality is not binary. A company that self-hosts runs models in **three** places, and
an account declares which by its `kind`:

| Zone | `kind` | Where the weights sit | Self-hosted? | On-box? |
| --- | --- | --- | --- | --- |
| `device` | `local` | the engineer's own machine (loopback only) | yes | **yes** |
| `fleet` | `fleet` | a machine the **organization** operates | yes | no |
| `vendor` | `openai` / `anthropic` / `gemini` / `xai` / `deepseek` / `openai-responses` | a third-party lab's API | no | no |

The middle rung is the one that carries the token volume in a self-hosting shop — a
GLM- or Kimi-class open model on company GPUs, shared by every engineer. Before
`kind: fleet` existed it could not be written down: as `local` it was refused (a local
account must carry a loopback base URL), and as `openai` it was admitted but became
indistinguishable from `api.openai.com` in every downstream record, so the org got no
credit for hardware it owns.

```json
{
  "id": "corp-glm",
  "kind": "fleet",
  "base_url": "http://glm.infer.corp.internal:8000/v1",
  "label": "GLM-5.2 vLLM server the company operates"
}
```

A `fleet` account needs an explicit, non-loopback `http(s)` endpoint (there is no public
default for a host only your org can name, and a loopback address is `device` by
definition — the zones must stay disjoint). Unlike a vendor account it does **not**
require a `cred_env`: an org-operated server on a private network commonly has no API
key. Supply one when the endpoint authenticates.

**Self-hosted is not the same as on-box, deliberately.** `ZoneFleet.SelfHosted()` is
true — its tokens count toward "we ran this on our own silicon" — but
`ZoneFleet.OnBox()` is false, so the residency floor still treats a `fleet:…` route as
remote and still denies a tenant-scoped payload routed there. Naming the zone changes
what fak can *attribute*, not what it *permits*. Letting an org-operated host carry a
sensitive payload is an enforcement change that needs an operator-declared trust
boundary (authenticated transport, a named host allowlist), and it is tracked
separately — it does not arrive as a side effect of declaring a zone.

The zones are ordered as an escalation ladder — `device` (0) → `fleet` (1) →
`vendor` (2) — so "prefer the cheapest rung that can do the work" is a comparison on a
declared value rather than a string switch. An unrecognized zone ranks *above* vendor
and is neither self-hosted nor on-box: unattributable never reads as free or as safe.

## A note on codex and Anthropic subscriptions

Codex's native wire is the OpenAI Responses API, so bind it to `kind: openai-responses`,
not plain `openai`. (`fak guard -- codex` autodetects the chat-completions `openai`
wire today; the roster lets you pick the Responses wire explicitly.)

An Anthropic Pro/Max subscription token (`sk-ant-oat…`) rides as
`Authorization: Bearer` plus the `oauth-2025-04-20` beta header, not as `x-api-key`.
The roster declares the account and the env var that holds the token; the Bearer-vs-key
header choice is made downstream by the Anthropic adapter at dispatch.

## Managed-cache posture across launchers

Every fak launcher that fronts an agent with `fak guard` — `fak accounts launch`,
`fak codex`, and the dispatch worker (`fak dispatch tick`/`auto`) — can hand guard a
**managed-cache posture** that decides whether the gateway upgrades the stable-prefix
`cache_control` breakpoint to the **1h-TTL** tier on the outbound Anthropic wire.

**A subscription seat stays passive by default, on purpose.** Guard's own `--managed-cache
auto` follows a never-speculate rule: it will not manage a wire whose billing it cannot see.
A Pro/Max seat authenticates with subscription OAuth (`Authorization: Bearer`), whose cache
economics guard does not own, so `auto` resolves **passive** there and no launcher silently
flips it. Nothing changes for a subscription fleet unless you ask for it.

**An API-key-billed fleet reaches ACTIVE without hand-editing any launcher**, via two env
knobs read once per launch and threaded onto the child's guard argv (not its environment, so
a *resumed* child — whose gateway env is deliberately stripped — keeps the posture):

| Knob | Effect |
|------|--------|
| `FAK_GUARD_API_KEY_ENV=ANTHROPIC_API_KEY` | Names the env var holding the API key. On the Anthropic wire this is the explicit opt-IN to API billing that lets `auto` resolve **ACTIVE** (guard bills the key, not the seat's subscription). |
| `FAK_MANAGED_CACHE=on\|off\|auto` | Forces the posture. `on` upgrades regardless of billing; `off` disables; `auto` (default) defers to the billing signal above. |

`fak accounts launch --managed-cache <mode>` and `fak codex --managed-cache <mode>` expose
the same lever per-launch (defaulting to `$FAK_MANAGED_CACHE`); the dispatch worker reads the
env knobs only. An unknown mode fails loud at the front-ends; the headless worker warns and
falls back to `auto` so a fleet turn never dies over a cache-posture typo. Either way the
guard child prints the RESOLVED posture in its startup banner, so the worker log witnesses
whether the 1h upgrade actually engaged. An unconfigured fleet emits no `--managed-cache`
flag at all — the guard argv is byte-identical to before this knob existed.

## What is shipped, and what is not

**Shipped:** the resolver. A roster resolves a routed plan — the scout and every
member, in member order — to per-member Targets, with fail-loud validation and the
residency-honest `EngineRoute`. The `fak route --accounts` path prints the binding.
This is pure and deterministic, witnessed by `go test ./internal/modelroute` and the
cross-package residency test.

**Deferred:** live multi-account dispatch
([#2528](https://github.com/anthony-chaudhary/fak/issues/2528)). v1 *resolves and prints*
a binding; it does not yet send a request over the wire to your chosen account. Building
the planners from the Targets and running an ensemble's members is the additive dispatch
wiring, the same shipped-vs-deferred split the routing spine uses for the ensemble fold
versus its execution. (Per-call routing into dispatch — writing the routed id to
`abi.ToolCall.Engine` before submit — has landed for the single-model case; the
account-resolved route is the next refinement of what that field carries.)

**Out of scope for v1:** this is a declarative, static binding resolver. It has no
per-account rate-limit, capacity, health, or failover signal. It is not the fleet's
load-aware switcher — it is the portable, reviewable account map that a switcher would
consult.

**Placement zones — shipped and deferred, precisely.** *Shipped:* the vocabulary. A
company-operated server is expressible (`kind: fleet`), its account validates on its own
terms, its route string round-trips to `ZoneFleet`, and `PlacementZone.SelfHosted()`
gives every downstream consumer one predicate to attribute a token by. Witnessed by
`TestZonesPartitionTheKnownKinds`, `TestEngineRouteCarriesTheZoneBackOut`,
`TestShippedExampleRosterCarriesAllThreeZones`, and — at the enforcing floor —
`TestFleetZoneStaysRemoteAtTheFloor` plus `TestTierOneRouteMirrorAgreesWithTheFloor`.
*Deferred, and deliberately so:* nothing yet **chooses** a zone automatically, the serving
path does not yet **record** the zone it used, and the residency floor does not yet treat
a fleet host as inside a trust boundary. Zone-aware placement, the self-hosted
token-fraction metric, and the escalation ladder are separate, individually reviewable
changes on top of this vocabulary — not implied by it. They are tracked as
[epic #5416](https://github.com/anthony-chaudhary/fak/issues/5416).

### The first consumer: honest stratum attribution

`internal/sessionaudit` classifies audited turns into billing buckets. It used to answer
"was this self-hosted?" from the **model id**, which cannot support the claim: Qwen,
Llama, Mistral, DeepSeek, GLM and Kimi all publish open weights, so each is servable
either from hardware you operate or from a vendor's API. That produced two contradictions
in one file — `deepseek*` was bucketed self-hosted while priced from the published
DeepSeek *API* rate card, and the roster above binds `qwen/qwen3.6-27b` to **Groq**, a
vendor, while the classifier called those tokens self-hosted.

A model id now buys only a model-family answer (`open-weights (placement unknown)`), and
placement comes from a real signal:

| Call | Answers |
| --- | --- |
| `ProviderBucket(model)` | whose *model* it is — never where it ran |
| `BucketForPlacement(model, zone)` | the bucket, given a `PlacementZone` string |
| `BucketIsSelfHosted(bucket)` | `(selfHosted, known bool)` — two values on purpose |

`BucketIsSelfHosted` returns a second `known` bool so unattributable volume cannot be
folded into either side of a self-hosted fraction. `TestPricedModelsAreNeverBucketedSelfHosted`
makes the underlying rule executable and total over the rate card: a model billed per token
by a third party ran on their hardware, so it can never also be bucketed as self-hosted.
