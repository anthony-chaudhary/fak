---
title: "fak serve config: auth, policy, and timeouts reference"
description: "Configure a network-facing fak serve gateway: bearer-key auth, the default-deny policy floor with live reload, and HTTP and planner timeout tuning."
---

# serve-config.md — configuring a network-facing `fak serve`

A single reference for the environment variables and `fak serve` flags that matter
when you put the kernel gateway in front of a model and expose it beyond loopback.
Every var and default below is read directly from the code (see the **Source**
column / pointers); none is guessed.

`fak serve` is a drop-in adjudication proxy: it fronts a model (an upstream
OpenAI-compatible/Anthropic provider, or the in-kernel engine) and runs every
proposed tool call through the kernel before the client sees it. By default it
binds loopback with **no authentication** — fine for local dogfood, not for a
network-reachable deployment. The three things you almost always set for a
network-facing deploy are covered below: **auth** (`--require-key-env`), the
**policy floor** (and how to reload it live), and **timeouts** (so a slow backend
or a slow client can't pin or trip a connection).

## Env-var reference

| Variable | Default | Units | Scope | Pre-raised by `dogfood-claude.sh`? | Source |
|---|---|---|---|---|---|
| `FAK_HTTP_READ_TIMEOUT_S` | `30` | seconds | global (per-connection) | no | `internal/gateway/http.go` |
| `FAK_HTTP_WRITE_TIMEOUT_S` | `90` | seconds | global (per-connection / whole handler) | **yes** → `FAK_DOGFOOD_TIMEOUT_S` (default 300s, 900s for `openai`) | `internal/gateway/http.go` |
| `FAK_HTTP_IDLE_TIMEOUT_S` | `120` | seconds | global (per-connection keep-alive) | no | `internal/gateway/http.go` |
| `FAK_PLANNER_TIMEOUT_S` | `60` | seconds (clamped `[5, 3600]`) | global (per upstream request) | **yes** → 300s (`ollama`/`shim`), 900s (`openai`) | `internal/agent/chat.go` |
| `FAK_PROVIDER_EXTRA_BODY_JSON` | unset | JSON object | global (per upstream request) | yes, from `FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON` | `internal/agent/chat.go` |
| `FAK_MODEL_DIR` | unset (synthetic checkpoint) | filesystem path | global (process start) | no | `internal/modelengine/modelengine.go` |
| `FAK_Q4K` | unset (lean-Q8 path) | flag (set/unset) | global (process start) | no | `cmd/fak/main.go` |
| `FAK_RATELIMIT_MAX_CALLS` | `0` (unlimited / inert) | call count | global (process start) | no | `internal/ratelimit/ratelimit.go` |
| `FAK_RATELIMIT_MAX_COST` | `0` (unlimited / inert) | cost units (~arg bytes) | global (process start) | no | `internal/ratelimit/ratelimit.go` |
| `FAK_RATELIMIT_KEY` | `trace` | enum: `trace`\|`tool`\|`global` | global (process start) | no | `internal/ratelimit/ratelimit.go` |
| `FAK_AUDIT_JOURNAL` | unset (no journal) | filesystem path (`.jsonl`) | global (process start) | no | `internal/journal/journal.go` |
| `FAK_IFC` | enabled | toggle (`off` disables) | global (process start) | no | `internal/ifc/ifc.go` |

Notes on the table:

- **Scope.** None of these is a per-HTTP-request override — there is no header or
  query knob that changes them per call. The HTTP timeouts apply per *connection*;
  the planner timeout applies per *upstream model request*; the rest are read once
  at process start. "global" means "set in the environment before `fak serve`
  starts."
- **`ReadHeaderTimeout`** is fixed at 10s in code and has no env override.
- **Disabling a timeout.** For the three `FAK_HTTP_*_TIMEOUT_S` vars, `0` selects
  Go's "no timeout" semantics (an explicit opt-out for a long-running local
  backend). A negative value is rejected and the default is kept (`durEnv` in
  `internal/gateway/http.go`).
- **`FAK_PLANNER_TIMEOUT_S` clamping.** Values outside `[5, 3600]` seconds are
  ignored and the 60s default is kept (`plannerTimeout` in `internal/agent/chat.go`).

## Auth: requiring a bearer key on a network-facing gateway

With no key configured the gateway is a pass-through — every route is open. That is
the loopback default. On a non-loopback bind with no key, the gateway logs a
`WARNING: binding ... with NO --require-key set` line but still serves.

Turn on auth by naming an **environment variable that holds the secret** — the
secret value is never a command-line argument:

```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"
fak serve --addr 0.0.0.0:8080 --require-key-env FAK_GATEWAY_KEY --policy policy.json
```

- `--require-key-env VAR` reads the secret from `$VAR` at startup. If `$VAR` is
  empty, `fak serve` prints `--require-key-env <VAR> is empty — starting with NO
  authentication` and continues unauthenticated, so make sure the var is exported
  and non-empty in the serving process's environment.
- Once set, every route except `/healthz` requires the secret. The gateway accepts
  it over **either** header:
  - `Authorization: Bearer <tok>` — OpenAI-compatible and fak-native clients.
  - `x-api-key: <tok>` — what Claude Code and the Anthropic SDKs send to
    `/v1/messages`.
  Both are compared against the same secret in constant time (SHA-256 digests), so
  the reject latency leaks neither the secret's bytes nor its length. A bare
  `Authorization` value with no `Bearer ` scheme is rejected.
- The policy/lifecycle routes under `/v1/fak/*` (reload, trace reset) require the
  same bearer token when `--require-key-env` is set.

Source: `withAuth` / `gatewayCredential` in `internal/gateway/http.go`; the
`--require-key-env` flag in `cmd/fak/main.go`; the dual-header note in
`DOGFOOD-CLAUDE.md`.

## Policy: the default-deny floor and reloading it live

The kernel adjudicates every proposed tool call against a capability-floor
manifest. **Anything not affirmatively allowed and not explicitly denied resolves
to the fail-closed `DEFAULT_DENY`** — an empty manifest (`{}`) denies every call.
With no `--policy` flag the kernel uses its built-in default floor; pass `--policy
FILE` to deploy your own. (Full manifest schema and refusal vocabulary live in
`POLICY.md` — not repeated here.)

Workflow:

```bash
fak policy --dump > policy.json          # start from the built-in default
# edit policy.json: allow the tools your agent needs, deny the irreversible ones
fak policy --check policy.json           # validate before it gates a run
fak serve --addr 0.0.0.0:8080 --policy policy.json --require-key-env FAK_GATEWAY_KEY
```

You can also validate at boot without binding a listener: `fak serve --policy
policy.json --policy-check` exits after validating the manifest.

**Switching/reloading a policy on a running gateway** — no process restart, the
warm vDSO cache and IFC ledger survive:

```bash
# edit policy.json in place, then:
curl -X POST http://HOST:8080/v1/fak/policy/reload \
  -H "Authorization: Bearer $FAK_GATEWAY_KEY"
```

The reload re-reads the **same file** that was passed to `--policy` at startup, so
"switch policy" means "rewrite that file, then POST reload." If `--require-key-env`
is set, the reload route requires the bearer token like every other `/v1/fak/*`
route. A related lifecycle route clears one trace's IFC high-water mark after an
operator-approved session boundary:

```bash
curl -X POST http://HOST:8080/v1/fak/trace/reset \
  -H "Authorization: Bearer $FAK_GATEWAY_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"trace_id":"gw-123"}'
```

Source: `POLICY.md`; `--policy` / `--policy-check` flags and the reload route in
`cmd/fak/main.go`.

## Timeout tuning: remote upstream vs slow local model

The two timeouts that interact with a model round-trip are `FAK_HTTP_WRITE_TIMEOUT_S`
(bounds the **whole HTTP handler**, and a live upstream model round-trip rides it)
and `FAK_PLANNER_TIMEOUT_S` (bounds the **upstream provider request** itself). The
write timeout must be **at least** as large as the planner timeout, or the handler
will be cut off while the upstream request is still legitimately in flight.

**Remote hosted upstream (fast first token).** Keep the conservative
network-exposed defaults. A hosted API answers in seconds, so 90s write / 60s
planner is plenty and protects you from a slow-loris client pinning a connection.
Set `FAK_PROVIDER_EXTRA_BODY_JSON` if the upstream needs provider-specific request
fields (e.g. vLLM/SGLang sampling knobs). Example:

```bash
FAK_HTTP_WRITE_TIMEOUT_S=120 FAK_PLANNER_TIMEOUT_S=120 \
  fak serve --addr 0.0.0.0:8080 --provider openai \
    --base-url https://upstream/v1 --model M --api-key-env UPSTREAM_KEY \
    --require-key-env FAK_GATEWAY_KEY --policy policy.json
```

**Slow local CPU model (first token can take minutes).** A multi-thousand-token
prefill on a CPU-served model can run for minutes, so the conservative defaults
will trip mid-turn. Raise **both** timeouts together — this is exactly what
`scripts/dogfood-claude.sh` does, pre-raising `FAK_PLANNER_TIMEOUT_S` and
`FAK_HTTP_WRITE_TIMEOUT_S` to `FAK_DOGFOOD_TIMEOUT_S` (default **300s**, **900s**
for the `openai` backend):

```bash
FAK_PLANNER_TIMEOUT_S=600 FAK_HTTP_WRITE_TIMEOUT_S=600 \
  fak serve --addr 127.0.0.1:8080 --gguf model.gguf --tokenizer tok/ \
    --policy policy.json
```

If the backend genuinely streams for longer than any sane ceiling, set
`FAK_HTTP_WRITE_TIMEOUT_S=0` to disable the write deadline entirely (Go's
"no timeout"). Note `FAK_PLANNER_TIMEOUT_S` cannot be disabled — it is clamped to
at most 3600s (1h); pick a value inside `[5, 3600]`.

`FAK_HTTP_READ_TIMEOUT_S` (30s) bounds how long a client may take to *deliver its
request body*, and `FAK_HTTP_IDLE_TIMEOUT_S` (120s) bounds an idle keep-alive
connection. Neither rides the model round-trip, so leave them at the defaults
unless you have unusually slow clients or want longer-lived keep-alives.

Source: timeout semantics and the "slow local backend" rationale in
`internal/gateway/http.go`; planner timeout in `internal/agent/chat.go`; the
300s/900s pre-raises in `scripts/dogfood-claude.sh`.

## Feature wiring status

Every `fak serve` feature, traced from its operator flag through the
`gateway.Config` field to the load-bearing runtime read that produces the effect.
A green `make ci` does **not** tell you a feature is reachable: a `Config` field
can be set on the struct while the flag that feeds it is missing, so the feature
is dead on the shipped binary. This table is the answer to "is it actually wired?"

The states are: **wired** (a runtime read consumes the field on a request/turn
path and the operator reaches it by default); **off-by-default** (fully wired,
inert until a non-default flag value arms it; a deliberate guard, not a defect);
**partial** (the producer side is wired but a documented consumer step is
deferred); **dead-wired** (the gateway reads the field but `serve.go` never
feeds it; unreachable on the shipped binary).

<!-- BEGIN serve-wiring (generated by `fak serve-wiring --md`; do not hand-edit) -->
| Feature | Status | Flag | gateway.Config field | Live call site | Note |
|---|---|---|---|---|---|
| `inkernelchat` | wired | `--gguf / --tokenizer` | `InKernelModel` | `internal/gateway/gateway.go:861` | with model+tokenizer and no --base-url, /v1/chat/completions and /v1/messages serve the in-kernel model |
| `replica` | wired | `--replica-base-url` | `ReplicaBaseURLs` | `internal/gateway/gateway.go:715` | 2+ endpoints -> ReplicaRouter round-robin |
| `vdso` | wired | `--vdso / --invalidation` | `VDSO` | `internal/kernel/kernel.go:348` | dedup fast path + tier-2 invalidation granularity |
| `toolfloor` | wired | `(adjudicator.Default.NeverAdmits)` | `ToolFloorDenies` | `internal/gateway/messages.go:392` | prunes provably-unreachable tool defs from the Anthropic passthrough; default-on, fail-safe |
| `decidesession` | wired | `(host func, default-on)` | `DecideSession` | `internal/gateway/session_admit.go:57` | run-state refusal + TurnsLeft debit + budget + pace, before the model turn |
| `debitsession` | wired | `(host func, default-on)` | `DebitSession` | `internal/gateway/session_admit.go:157` | debits TokensLeft + context budget after the planner returns |
| `routemanifest` | wired | `--route-manifest` | `RouteManifest` | `internal/gateway/gateway.go:1127` | binds ToolCall.Engine before Submit; flag wired (was DEAD_WIRED before this pass) |
| `ctxview` | off-by-default (wired) | `--ctx-view-budget` | `CtxViewBudget` | `internal/gateway/gateway.go:807` | re-materializes history as an O(1) planned ctxplan view under the budget; off at 0 |
| `compacthistory` | DEFAULT-ON (wired) | `--compact-history-budget` | `CompactHistoryBudget` | `internal/gateway/messages.go:365` | compacts old turns in the Anthropic outbound body once it sprawls past the budget, cache prefix byte-identical; defaults to ~48k (`gateway.DefaultCompactHistoryBudget`), pass 0 to disable |
| `debugstats` | off-by-default (wired) | `--debug-stats` | `DebugStatsf` | `internal/gateway/debug_stats.go` | emits one compact payload-free per-turn line to stderr that leads with a verdict (ok/warming/degraded/cold) + the NET write-premium-aware token saving, then cache health + compaction; raw provider counters stay on /metrics and --log; off by default |
| `resetonbudget` | off-by-default (wired) | `--reset-on-budget` | `ResetOnBudget` | `internal/gateway/session_admit.go:108` | distills a carryover seed and continues transparently on budget exhaustion; needs --context-budget-tokens |
| `budgetwebhook` | off-by-default (wired) | `--budget-webhook` | _(observer seam)_ | `internal/session/usage.go:73` | POSTs a pre-exhaustion warning + exhaustion event; wired via WatchBudget, off when URL empty |
| `notifier` | wired | `--notify-native / --notify-webhook / --notify-slack` | _(observer seam)_ | `cmd/fak/serve.go (WatchTransitions)` | #761 stop-reason push notifier; native default-on (was DEAD_WIRED before this pass) |
| `enginecache` | off-by-default (wired) | `--engine-cache-engine` | `EngineCacheEngine` | `internal/gateway/gateway.go:1480` | resets the serving-engine cache after a quarantined proxy turn; off when engine empty |
| `backend` | off-by-default (wired) | `--backend` | `Backend` | `internal/agent/inkernel_planner.go:271` | decodes the in-kernel chat through the compute HAL device; off when name empty |
| `cpuoffloadexperts` | off-by-default (wired) | `--cpu-offload-experts` | `CPUOffloadExperts` | `internal/agent/inkernel_planner.go:282` | with --gguf --backend, keeps MoE expert GEMMs on host RAM while dense/router/attention run on the device; off by default |
| `steersession` | partial | `(host func, default-on)` | `SteerSession` | `internal/gateway/http.go:951` | POST /session/{id}/steer sends onto a2achan; the running-session TryRecv splice is deferred (#760) |
<!-- END serve-wiring -->

This table is **kept honest by the trunk**, not by hand. `fak serve-wiring`
re-derives the wiring on each run: it reads the real `serve.go` and `gateway.go`
and cross-checks that every audited row's `Config` field still exists *and* is
still set in `serve.go`'s `gateway.New(Config{...})` literal, and that no `Config`
feature is left unaudited. A field `serve.go` stops feeding flips its row to
dead-wired and reds the gate:

```bash
fak serve-wiring          # the summary (counts + per-feature call sites)
fak serve-wiring --md     # regenerate the table above
fak serve-wiring --check  # CI gate: exit 1 on wiring drift
```

The verdicts come from an audited baseline (each row traced flag -> `Config` field
-> runtime read, then adversarially verified) in `cmd/fak/servewiring.go`; the
drift cross-check is in the same file, exercised by `cmd/fak/servewiring_test.go`.
When you wire a new `serve` feature, add its row there and the gate keeps it true.
