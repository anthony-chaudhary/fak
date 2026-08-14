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
tick/wave`, `fak accounts launch`, `fak accounts next`, and `fak manage` auth warnings, so callers
can gate on `can_serve` and surface the closed login status instead of re-deriving readiness from
raw credential files.

## A seat on a third-party Anthropic-compatible endpoint

Claude is also sold through other people's gateways — a cloud vendor's serving endpoint, a
cloud marketplace, a company-internal proxy. Those speak the Anthropic wire protocol but
authenticate *their* tenant credential, serve *their* model namespace, and live at *their*
host. A seat can name all three, so a vendor endpoint becomes an ordinary member of the
switcher rather than a shell script off to the side:

```
fak accounts add --name vendor --suffix '' \
  --reserved \
  --api-key-env ANTHROPIC_AUTH_TOKEN \
  --base-url 'https://gateway.example.com/serving-endpoints/anthropic' \
  --seat-env 'ANTHROPIC_MODEL=vendor-claude-sonnet-5' \
  --seat-env 'ANTHROPIC_DEFAULT_SONNET_MODEL=vendor-claude-sonnet-5' \
  --seat-env 'CLAUDE_CODE_USE_GATEWAY=1'
```

Two new authored fields carry it. `base_url` (`--base-url`) is the endpoint; its presence is
what makes a seat third-party, and code asks `Home.ThirdParty()` rather than pattern-matching
a hostname. `extra_env` (repeatable `--seat-env KEY=VALUE`) is the non-secret environment the
launched agent needs — the model ids, the gateway toggles, any vendor header. Non-secret is
**enforced**, not requested: `ValidateExtraEnv` refuses a credential-shaped name at both write
and launch time, so a registry — a plaintext file, often in a shared tree — cannot become the
place someone parks a token. The credential itself stays a reference (`--api-key-env`), exactly
as `cred_env` does one layer up.

### Reserved: usable on request, never on rotation

`--reserved` is the answer to "register it, but never let anything pick it for me." A reserved
seat is excluded from the rotation pool (`RotationPolicy.AvoidReserved` defaults on) while
staying fully resolvable by name, so `fak accounts resolve vendor` and `fak accounts launch
--name vendor` work while `fak accounts next` will not wander onto it. `fak accounts rotation
--json` shows both halves at once — an empty `pool`, and the seat under `excluded` with reason
`reserved`, `login: ready`, `can_serve: true`. Metered or per-token billing is the usual reason
to want this: the seat is there for the one job that asked for it.

### Why a vendor seat launches with `--guard=false`

`fak manage` fronts its child with its **own** `ANTHROPIC_BASE_URL` pointing at guard's loopback
gateway, and proxies upstream with the credential guard holds. Under guard a vendor seat's
endpoint is therefore not unused — it is *replaced*, and the traffic bills a different account
than the operator named. Nothing looks wrong from the outside: the agent starts and answers.
So a guarded launch of a `base_url` seat is **refused**, with the endpoint and the escape flag
in the message. Reach the vendor directly with `--guard=false`, accepting the trade the flag
already implies: no kernel adjudication, no vCache hop.

Three further adjustments happen on that path, each because the default is a first-party
assumption that is wrong here:

- **Inherited first-party credentials are dropped.** A `fak manage` session exports its own
  `ANTHROPIC_API_KEY` into every child, so a vendor seat launched from one would carry a
  first-party key next to the vendor endpoint and could present the wrong tenant's token.
  Overriding the endpoint alone is not enough. The variable the seat itself names is kept, and
  the launch says on stderr which ones it dropped.
- **The seat's own credential is declared to the spawn broker's secret floor.** The always-on
  floor strips any variable whose name contains `TOKEN`, which is precisely the bearer variable
  a third-party gateway authenticates. Without a declaration the `--api-key-env` *reference*
  reaches the child's argv while the value does not, and the launch reports "Not logged in"
  against a configuration that is entirely correct. The declaration is narrow: only the
  variable this launch already references, exempt for this one hop, and the floor runs again at
  the child's own spawns, so nothing becomes inheritable.
- **No first-party model is pinned.** `fak accounts launch` normally pins a first-party default
  `--model` with a fallback chain behind it; against a vendor namespace those name models that
  do not exist there, turning one clean failure into a walk through several. An unset `--model`
  therefore defers to the seat's own `$ANTHROPIC_MODEL`. An explicit `--model` still wins.

Headless is then just the seat plus a prompt, and the result comes back attributed to the
vendor's model with `provider: gateway`:

```
fak accounts launch --name vendor --guard=false -- -p 'ping' --output-format json
```

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

### Walking the ladder: `Roster.Place`

`Roster.Place(class, candidates)` is what actually walks it. Given a `WorkClass` and the
models you have bound — each with the work tier you have evidence it can serve — it
returns the cheapest rung that can take the job, plus the full ladder walk that produced
the answer:

```go
p, err := roster.Place(modelroute.ClassRoutine, []modelroute.Candidate{
    {Model: "tiny",     Capability: modelroute.TierT2, Measured: true},
    {Model: "corp-mid", Capability: modelroute.TierT1, Measured: true},
    {Model: "frontier", Capability: modelroute.TierT0, Measured: true},
})
// p.Zone == ZoneDevice, p.SelfHosted() == true, p.Escalated == false
```

Two rules stop it from being wishful thinking:

**The floor belongs to the work, not to the model.** Admission runs through
`TierPolicy.Admit`, so a cheap rung cannot take work above its tier however available it
is. Security/release/destructive work has a floor that never drops to routine, so it
skips a measured local model and lands on the fleet — `p.Escalated` is true and the
device rung reports `zone-under-tier`. An unrecognized class stays at the strictest floor
rather than inferring a cheap one.

**Unmeasured capability may not descend the ladder.** `Candidate.Measured` records
whether `Capability` is a measurement or an assertion, and an assertion is skipped on
every rung below the vendor. This matters right now: `internal/ablate`'s `StubTierScorer`
measures nothing, so a placer that trusted asserted grades would route everything to the
cheapest rung on a number nobody computed, and would look like it was working. Until
capability measurement lands, an operator opts a rung in by marking a candidate
`Measured` — a claim they can be held to. `TestUnmeasuredCapabilityCannotDescendTheLadder`
pins both halves: an all-asserted ladder places on the vendor, and the same work moves to
the device the moment the cheap rung is measured.

Every rung is reported, including the ones passed over, in a closed reason vocabulary
(`placed-in-zone`, `zone-under-tier`, `zone-capability-unmeasured`, `zone-no-candidate`,
`zone-not-reached`, `escalated-past-cheaper-zone`) — because "why is this *still* going to
a vendor?" is the question an operator actually asks. A candidate the roster cannot
resolve is a loud error, never a silent skip: a typo in a placement config must surface as
a misconfiguration rather than as traffic quietly continuing to bill a vendor.

The trap this code is written around is that `WorkTier` numbers are **inverted** — T0 is
the most demanding but the lowest number. Every comparison goes through
`WorkTier.MeetsRequirement`, never a raw `<` or `>=`. Substituting the natural-looking
`capability >= required` fails four tests at once, and the way it fails is the point: a
4B laptop model takes both the ultra-hard and the security/destructive work.

### Declaring what the work is

`Place` needs a `WorkClass`; routing produces a `Subject`. `ClassOf(Subject)` is the
joint, and it is where an automatic placer is most tempted to cheat — it would be easy to
look at a tool name, decide "grep is harmless", and route it to a laptop. fak does not do
that. Judging what a call can destroy is `internal/adjudicator`'s reversibility question,
it sits a tier above this leaf, and it is deliberately not re-guessed from a name here.

So a work class is **declared, or it is conservative**. Declare it as a subject label:

```jsonc
{ "match": { "aspect": "tool_call", "labels": { "work_class": "routine" } },
  "plan":  { "model": "tiny" } }
```

| Input | Classified as | Why |
| --- | --- | --- |
| `labels.work_class` naming a known class | that class, **declared** | the operator said so |
| `aspect: "scout"`, nothing declared | `routine`, **declared** | a scout *is* the cheap classify-first probe — the aspect names the work |
| anything else | *(empty)*, **undeclared** | `PolicyFor` already puts the empty class at the strictest floor |
| `labels.work_class` misspelled | *(empty)*, **undeclared** | a typo in `routine` must not read as permission to use a laptop |

Undeclared work therefore lands exactly where it landed before this feature existed — on
a vendor — and reports `class-undeclared-conservative` so an operator can see the fix is
a label rather than more hardware. There is no second conservatism mechanism layered on
top; one gate in one place is why the behaviour stays predictable. `Complexity` can then
ratchet a floor **up** (a high-complexity `routine` subject becomes `normal-impl`) and
never down, which is what makes accepting complexity as an input safe at all.

`Roster.PlaceSubject(subject, candidates)` composes the two and returns the placement
*and* the classification, so both halves of the decision are visible together.

### Declaring what a sub-agent type does

Sub-agents are where the cheap rungs would pay for themselves — a fleet spawns far more
delegated work than top-level turns — and `Roster.PlaceSpawn(parent, class, candidates)`
already answers "which rung should this child run on". It refuses an **empty** work class
on purpose: an unclassified spawn is a missing classification, not a routine one, and
letting delegation reach a laptop without stating what the work *is* would be exactly the
shape a floor bypass takes.

That refusal is what left the call unreachable. A spawn arrives as an admitted tool call
carrying an agent **type** and a prompt. The prompt is prose, and reading a class out of
prose is the guess `ClassOf` refuses everywhere else. The type is structured — but it is
not a work class: only an operator knows whether the agent type *their* fleet calls
`explore` does bounded lookup or ships code.

So it is declared, in the roster the operator already owns:

```jsonc
"spawn_classes": [
  { "type": "explore",        "class": "routine" },
  { "type": "code-reviewer",  "class": "normal-impl" },
  { "type": "release-cutter", "class": "security-release-destructive" }
]
```

`Roster.SpawnClassFor(type)` resolves one, returning `(class, declared bool)`. The rules
are the ones the rest of this leaf runs on:

| Input | Resolves to | Why |
| --- | --- | --- |
| a declared type (case- and space-insensitive) | that class, **declared** | the operator said so |
| a type nobody declared, or no `spawn_classes` block at all | *(empty)*, **undeclared** | `PlaceSpawn` then refuses — the spawn keeps whatever placement it has today |
| a declared type whose class token this package does not know | *(empty)*, **undeclared** | same rule as a misspelled `work_class` label: a typo is not permission to use a laptop |

The match is **exact**, never a prefix and never a glob — `explore` must not answer for
`explore-and-delete`. A malformed entry is refused at load rather than skipped: an empty
type, an empty or unknown class (the error names the four options), a duplicate type, or a
type carrying a route delimiter. A silently-dropped entry is the worst outcome available,
because it looks correct from the operator's side while behaving like an undeclared one,
and nothing points at the typo.

**Shipped:** the declaration, its validation, and the resolver — plus the block in
`examples/model-accounts.example.json`, which the test suite parses and resolves.
**Not shipped:** nothing on the spawn path calls `SpawnClassFor` yet. The gateway does
observe spawns (`internal/gateway/adjudicate_proposed.go` stamps every admitted tool call
and already recognises the spawn-shaped ones — that is where `spawn_count` comes from),
and that is the seam a placement call would bind to, but binding it changes where
delegated traffic runs and is a separate, individually reviewable change under
[epic #5416](https://github.com/anthony-chaudhary/fak/issues/5416). Declaring
`spawn_classes` today moves no traffic.

One neighbouring field is *not* this signal, and it has already been mistaken for it: a
session's `ParentTrace`/`Generation` is **continuation** lineage — the same agent after a
budget-reset re-continuation — written only by `session.Table.Recontinue`. It never marks
a row as somebody's child. `SpawnCount` is the sub-agent axis, and it is parent-side: it
counts children a trace spawned, never whose child a trace is.

### Seeing the ladder before it moves traffic: `fak route --place`

```console
$ fak route --accounts accounts.json --place --labels work_class=routine
```

The oracle walks the ladder for one subject against your real roster and prints the rung
it chose, every rung it passed over, and the closed reason token for each. It exists
*before* anything on the dispatch path calls `Place`, deliberately: a placement policy
nobody can inspect is a policy nobody can be held to, and an operator has to be able to
see what the ladder would do with **their** roster before fak starts moving traffic on
the strength of it. Add `--json` for the same answer as a report.

The candidate pool is every model the roster **binds**, not just the one routing picked —
the question is which rung can serve this class of work at all. Compatibility-only and
deprecated-alias bindings are excluded, since admitting a legacy spelling would let the
same hardware appear twice on a rung.

Add `--spawn-type TYPE` to ask the *delegated* question in the same breath: where would a
sub-agent of that type run, given that this turn ran where the block above says. Against
the shipped example roster:

```console
$ fak route --accounts examples/model-accounts.example.json --place \
      --labels work_class=ultra-hard --spawn-type explore \
      --capability zone-device=t2,zone-fleet=t1,large=t0
```
```
SPAWN PLACEMENT  (--spawn-type explore)
  work class   routine  [declared by the roster's spawn_classes]
  parent       zone=vendor  model=deepseek-flash
  placed       zone=device  model=zone-device
  self-hosted  yes      escalated  no       failed-over  no
  relation     spawn-descended-from-parent-zone spawn-inherit-unmeasured
  descent      yes      self-hosted descent  yes
```

The child's class comes from the `spawn_classes` declaration above — never from the
parent, the prompt, or the spelling of the type — so an **undeclared** type is refused
with the roster's own list of declared ones rather than assumed routine. The parent's rung
is *recorded, not obeyed*: a sub-agent spawned from a frontier turn still lands on the
laptop, which is the whole point, and `self-hosted descent` is the event
[epic #5416](https://github.com/anthony-chaudhary/fak/issues/5416) counts. The report also
answers the counterfactual — what *inheriting* the parent's model would have done — and
says `UNKNOWN` when the parent was never graded, because "the status quo was fine here"
is a claim that needs a measurement behind it. `--json` carries the same answer under a
`spawn` key that is **absent** when the question was not asked.

### Saying what a model can do

Because unmeasured capability may not descend the ladder, a fresh roster places
*everything* on the vendor rung — correctly, and uselessly. There are three ways to give
the ladder something to descend on, in increasing order of what they cost to produce:

| Flag | What it is | What it costs to be wrong |
| --- | --- | --- |
| `--capability qwen3.6-4b=t2,glm-5.2=t1` | an operator **asserting** a grade | attributable to a person, but does not scale past a handful of models |
| `--evidence FILE` | a summary of observed outcomes, graded here | someone assembled the counts; you are trusting their arithmetic |
| `--outcomes FILE` | the append-only turn journal, **counted here** | nothing is trusted but the record itself |

All three feed one grader (`modelroute.GradeCapability`), and its bar is yours:

```console
$ fak route --accounts accounts.json --place --labels work_class=routine \
    --outcomes turns.jsonl --since 30d --grade-floor attempts=50,rate=0.9,witness
```

The default bar is 20 independently verified attempts at 80%. There is deliberately **no
knob that makes self-report count** — a model's own claim of 500/500 successes buys
nothing, and the refused attempts are reported *as refused* rather than as absent. An
operator who wants to assert a capability can already do so, attributably, with
`--capability`. A model claimed by both an assertion and a measurement is a named
refusal: they cannot both be the grade.

A grade is the **floor** of the work observed, never its optimal tier, and a model that
fails the bar is `UNMEASURED` **with a reason** rather than graded at the worst tier. The
three reasons call for opposite responses and are never collapsed into one:

| Reason | The fix |
| --- | --- |
| `no-trusted-evidence` | run the turns under a witness or a judge — the volume is there, the provenance is not |
| `insufficient-samples` | run more turns, or lower `attempts=` if you meant to |
| `below-success-floor` | this model cannot do this work; that is the answer |

### Grading from the turn journal

`--outcomes` reads the record itself — one JSON object per line, appended as turns
happen (`modelroute.TurnOutcome`):

```jsonl
{"id":"t-4471","model":"qwen3.6-4b","class":"routine","zone":"device","success":true,"verify":"witness","at":"2026-07-24T11:02:00Z"}
```

The fold is written to **lose** evidence rather than invent it, because each alternative
inflates a claim, and it reports every loss:

```
  journal      turns.jsonl: 1463 line(s), 1332 counted, 4 model(s) with evidence
  not counted  1 unparseable line(s), 30 replayed id(s), 60 older than 30d, 40 undated
```

- A repeated `id` is counted **once**. Replaying a file is the cheapest possible way to
  manufacture a grade: the same 20 successes appended three times must not read as 60.
- An **undated** row cannot be shown to be inside a window, so `--since` excludes it — and
  counts it apart from a stale one, because one needs a wider window and the other needs a
  producer that stamps its rows. Ask for no window and a missing date costs nothing.
- **Provenance is never merged.** 100 self-reported turns and 20 witnessed ones about the
  same model stay two rows; merging them would force a pick of provenance, and any pick
  is a lie. The grade keeps the weakest provenance behind whatever it merged.
- A **torn line** is skipped and counted, never fatal. A fleet appending to this file will
  eventually die mid-write, and discarding thousands of verified turns over one partial
  row would make the honest path the fragile one.
- Rows with no `id` are kept — a missing id is not proof of a duplicate — but reported as
  a corpus that **cannot be checked for replay**, since that is a fact about how much the
  number is worth.

`--evidence` and `--outcomes` are mutually exclusive: a run that took some of its answer
from a journal and some from a hand-written summary produces a grade whose provenance
nobody can state afterwards.

### Producing the journal

`fak dispatch tick --placement-evidence` writes it, from the witness sweep the dispatch
loop already runs. The seam is **off by default** — an unconfigured tick writes no
sidecars, adds no payload keys, and creates no journal — and switching it on turns each
finished worker slot into one graded row:

```console
$ fak dispatch tick --placement-evidence --accounts-roster tools/model-accounts.json
```

Three facts are recorded where each one exists, because a later sweep cannot reconstruct
any of them — labels get re-tagged, rosters get re-bound:

| Field | Where it comes from | Refused when |
| --- | --- | --- |
| `model` | the slot's `--model` pin | unpinned — a seat-default slot is evidence about nobody |
| `class` | the issue's `tier/*` labels (T2 → `routine`, T1 → `normal-impl`, T0/ultra → `ultra-hard`) | untagged, or a coordination label — an unknown class reads as the **T0 floor**, which is right conservatism for choosing a floor for WORK and a capability-*minting* hole read backwards to grade a MODEL |
| `zone` | the roster's binding for that model, bound entries only | the roster does not bind it — never defaulted to `device`, since over-reporting self-hosting is the error you would act on |

Provenance follows **what checked the slot**, never how it turned out:

```
the affected tests ran (GREEN or RED)  ->  verify=witness, success = GREEN
no test run                            ->  verify=judge,   success = the diff-shape claim
```

That split is the one that matters, because `--outcomes` never merges across provenance: a
producer that filed its successes as `witness` and its failures as `judge` would hand the
grader a witness row that is 100% successes — every row honest, the corpus a lie, and
nothing downstream able to tell. Both outcomes of the same check carry the same
provenance, so a model cannot bank its greens at a rung its reds never reach.

Rows land in `.dispatch-runs/turn-outcomes.jsonl`, and the tick payload reports what the
sweep could *not* grade — `unattributed`, `unclassified`, `undated`, `unidentified` — as
four separate numbers rather than one "skipped", because each is a different missing wire
with a different fix, and *"the fleet ran 400 slots and graded nobody"* has an actionable
cause where a silent zero does not.

Then grade against it, with nothing hand-written anywhere in the path:

```console
$ fak route --accounts tools/model-accounts.json --place --labels work_class=routine \
    --outcomes .dispatch-runs/turn-outcomes.jsonl --since 30d
```

The bar is still 20 witnessed attempts at 80% per (model, class), so a fleet reaches its
first MEASURED model only once it has actually done that much of that kind of work. That
is the intended shape: the ladder descends on what a model has been observed to do, not on
when the seam was switched on.

### Placing against what is actually answering

A grade says what a model *can* do. It says nothing about whether the box is powered on.
`--serving FILE` hands the ladder a liveness snapshot — what your own probes just saw —
and the ladder walks past candidates it reports are not serving:

```json
{
  "schema": "fak.modelroute.serving.v1",
  "as_of_unix": 1785312000,
  "max_age_seconds": 120,
  "covers": ["device", "fleet"],
  "models": {
    "zone-device":        {"state": "up",   "observed_unix": 1785311950},
    "zone-fleet":         {"state": "down", "observed_unix": 1785311950},
    "zone-fleet-agentic": {"state": "up",   "observed_unix": 1785311950}
  }
}
```

```console
$ fak route --accounts examples/model-accounts.example.json --place \
      --labels work_class=normal-impl --serving serving.json \
      --capability zone-device=t2,zone-fleet=t1,zone-fleet-agentic=t1,large=t0
```
```
  placed       zone=fleet  model=zone-fleet-agentic
  self-hosted  yes      escalated  yes      failed-over  yes
  ladder
    device  -                        zone-capability-unmeasured x2 zone-under-tier
    fleet   zone-fleet-agentic       zone-serving-down placed-in-zone escalated-past-cheaper-zone
    vendor  -                        zone-not-reached

SERVING SNAPSHOT  (--serving serving.json)
  as of        1785312000      max age  120s
  covers       device fleet
  observations 3, of which 3 name a model this roster binds
  SILENT       small tiny-classifier
               the snapshot claims to speak for these candidates' rungs and then says
               nothing about them, so they are passed over as unknown rather than
               assumed well. That is a gap in the probe, not an outage.
```

**One dead GPU host is not a dead rung.** A fleet is several machines, and abandoning the
rung because one box rebooted sends every token to a third-party lab to route around a
neighbour that was idle — so the failover above stayed on the company's own hardware.

| Verdict | What it means | Placed? |
| --- | --- | --- |
| `zone-serving-down` | observed unable to serve | passed over |
| `zone-serving-unknown` | no verdict, on a rung the snapshot *claims* to cover | passed over |
| `zone-serving-stale` | a verdict that cannot be shown fresh | passed over |
| `zone-serving-degraded` | serving under strain | **recorded, and placed anyway** |

Degraded is deliberately not actionable: a loaded host still takes work, and shedding at
the first sign of queueing inverts self-hosting exactly in the busy hour it is meant for.

Three rules decide the rest, and each is fail-closed in a different direction:

- **Silence only means something where you said it would.** `covers` is the operator
  declaring which rungs the probe actually watches; a candidate the snapshot is silent
  about *inside* that coverage is unknown, not healthy, because a crashed prober must not
  read as a fleet in perfect health. Outside coverage, silence gates nothing.
- **An observation is always honored**, coverage or not. Coverage governs what silence
  means, never what a verdict means — a report that says a host is down is not overruled
  by a `covers` list that forgot to mention its rung.
- **Freshness is fail-closed, and so is every way of not having it**: no observation
  stamp, no `as_of_unix`, past `max_age_seconds`, or stamped *after* the report all read
  as stale. A snapshot with a bound and no clock to measure against therefore gates
  everything it covers, which is correct and reads exactly like a total outage — so the
  summary says `STALE-ALL` and names the missing field rather than leaving you to guess.

**`failed-over` is carried apart from `escalated`** because they want opposite fixes.
Escalated says a cheaper rung could not do this work — a capability fact, stable until
someone changes the roster. Failed-over says a rung that *could* have was routed around
because something was not answering — an operations fact that may already be untrue. A
bill read without that second bit shows vendor spend attributed to work that "needed" a
vendor, and sends an operator hunting a capability problem that does not exist while the
dead host stays dead. Both are printed for every placement, including the ones that stayed
cheap: a failover *within* a rung cost nothing and still means something is down.

The snapshot and the roster are validated apart and meet only here, so the report also
names what neither half can see alone:

```
  observations 2, of which 0 name a model this roster binds
  UNBOUND      glm-5.2 kimi-k3
               these observations gate NOTHING: the ladder only ever asks about
               models the roster binds, so a probe filed under an id nothing binds
               is honored nowhere and this run is identical to one with no snapshot.
               Check the ids against the roster's bindings, not its upstream names.
               Each of these is the UPSTREAM name of a candidate: the probe filed its
               result under the name it dialled, and the ladder asks by routed id.
                 glm-5.2  ->  zone-fleet
                 kimi-k3  ->  zone-fleet-agentic
```

That is the fail-open the flag would otherwise ship: a clean-validating report, an
operator who believes they gated a dead host, and a placement identical to one with no
snapshot at all. The same typo *inside* declared coverage fails the other way instead —
the real candidate is now silent on a rung the report speaks for, so it is passed over as
unknown and a healthy host looks dead. One id appearing in `UNBOUND` and its routed name
in `SILENT` is the signature of exactly this mistake. Where an upstream name resolves to
**two** candidates, both are named: two accounts serving the same weights is how a rung
scales, and one observation cannot speak for both pools — which is why the snapshot is
keyed by routed id in the first place.

Two smaller refusals hold the same line. An unknown field is rejected rather than ignored,
because `max_age_sec` otherwise parses, validates, looks right, and silently switches
fail-closed freshness *off*. And a named file that cannot be read is an error, never an
empty report: dropping the flag means "no liveness signal", while a missing file means
your probe did not run, and placing work as though every host were healthy is precisely
the failure the gate exists to prevent. In `--json` the `serving` key is **absent** unless
a snapshot was supplied, so it can never be misread as a probe that saw nothing.

**What is not shipped:** nothing produces this file either. It is deliberately not a probe
of fak's own — placement is asserted to be pure (same roster, same candidates, same
report, same answer), and a CLI that dialled endpoints would be reading a clock and a
network inside a decision that reads neither. The snapshot is your monitoring's output,
in fak's vocabulary.

## A note on codex and Anthropic subscriptions

Codex's native wire is the OpenAI Responses API, so bind it to `kind: openai-responses`,
not plain `openai`. (`fak manage -- codex` autodetects the chat-completions `openai`
wire today; the roster lets you pick the Responses wire explicitly.)

An Anthropic Pro/Max subscription token (`sk-ant-oat…`) rides as
`Authorization: Bearer` plus the `oauth-2025-04-20` beta header, not as `x-api-key`.
The roster declares the account and the env var that holds the token; the Bearer-vs-key
header choice is made downstream by the Anthropic adapter at dispatch.

## Managed-cache posture across launchers

Every fak launcher that fronts an agent with `fak manage` — `fak accounts launch`,
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
*Also shipped:* `Roster.Place` **chooses** a zone (see *Walking the ladder* above).
*Also shipped, opt-in and off by default:* the dispatch loop now closes the loop on its
own evidence — `fak dispatch tick --placement-evidence` **records** the rung and work
class each finished slot ran under and journals its graded outcome (see *Producing the
journal* above), and `--rung-placement` grades the roster from that journal and starts an
unpinned worker on the cheapest rung the evidence supports. An unconfigured tick is
byte-identical to one from before the ladder existed.

*Deferred, and deliberately so:* the residency floor still treats a fleet host as
**remote** — `ZoneFleet.SelfHosted()` is true but `OnBox()` is false, so naming the zone
changed what fak can *attribute*, not what it *permits* (`TestFleetZoneStaysRemoteAtTheFloor`).
Sub-agent placement is still refused rather than guessed: `PlaceSpawn` will not accept an
empty work class, and mapping an agent type onto one needs an operator-owned declaration
(`spawn_classes`) rather than an inference from the call site. And nothing produces a
`--serving` liveness snapshot automatically. These are separate, individually reviewable
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
