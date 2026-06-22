# Harden `fak serve` for real use (`--require-key-env`, end to end)

Before you expose `fak serve` past localhost, exactly one flag stands between an
open gateway and an authenticated one: **`--require-key-env VAR`**. This example
is the runnable proof of the whole auth loop — an authed client succeeds, an
unauthed client is rejected, **both** header shapes work, and the operator routes
(`/metrics`, `/v1/fak/*`) inherit the same gate.

It is the deployment-surface companion to
[`../../GETTING-STARTED.md`](../../GETTING-STARTED.md) §3 ("Auth" + "Harden it for
real use"), which describes the flag; this directory *runs* it. (The doc reference
shipped in [#138](https://github.com/anthony-chaudhary/fak/issues/138).)

## Run it

```bash
examples/auth-hardening/run.sh            # build fak, serve with auth on, run 6 witnesses, teardown
```

Needs only Go (to build `fak`) and `curl` — **no model, key, or GPU**. The auth
gate is a pure function of `(token, header)`, so the witnesses hit `/v1/models`
and `/metrics`, which answer `200`-when-authed with no upstream configured — so the
result is **deterministic**, identical on every run. The six witnesses **run in a
few seconds** after the one-time `go build`. A captured run is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

The six witnesses:

| # | Request | Expect | Why |
|---|---|---|---|
| 1 | `Authorization: Bearer <correct>` | `200` | OpenAI / fak-native client shape |
| 2 | `Authorization: Bearer <wrong>`   | `401` | wrong secret rejected |
| 3 | *(no auth header)*                | `401` | missing credential rejected |
| 4 | `x-api-key: <correct>`            | `200` | Claude Code / Anthropic SDK shape |
| 5 | `GET /metrics` *(no header)*      | `401` | operator route inherits the gate |
| 6 | `GET /metrics` `Bearer <correct>` | `200` | …and opens with the same token |

## What `--require-key-env` does

- **Names an env var, never a literal.** `--require-key-env FAK_TOKEN` reads the
  secret from `$FAK_TOKEN`, so the token never lands in `argv` (where `ps` would
  leak it). If the flag is set but the var is unset/empty, `fak serve`
  **refuses to start** (exit 2) rather than silently binding an unauthenticated
  network gateway. (Conversely, binding past loopback *without* the flag prints a
  loud `WARNING: … exposed without authentication`.)
- **Constant-time compare.** The presented token and the configured secret are
  compared as SHA-256 digests via `crypto/subtle.ConstantTimeCompare`, so reject
  latency leaks neither the secret's bytes nor its length.
- **Either header is accepted.** `Authorization: Bearer <tok>` (OpenAI / fak-native
  clients) **or** `x-api-key: <tok>` (Claude Code and the Anthropic SDKs, which is
  how you front Claude Code over `ANTHROPIC_BASE_URL`). A bare `Authorization`
  value with no `Bearer ` scheme is **not** stripped leniently — it is rejected.
- **Gates every route but `/healthz`.** The single unauthenticated route is the
  `/healthz` readiness probe; everything else requires the secret. A rejected
  request gets `401` with an OpenAI-style error envelope
  (`{"error":{"message":"missing or invalid credentials","type":"authentication_error"}}`).

The whole gate is ~15 lines in
[`../../internal/gateway/http.go`](../../internal/gateway/http.go) (`withAuth` +
`gatewayCredential`), covered by tests in
[`../../internal/gateway/gateway_test.go`](../../internal/gateway/gateway_test.go)
and [`metrics_test.go`](../../internal/gateway/metrics_test.go).

## The operator-route contract

The same token gates the **operator surface**, not just the model surface — so a
secret good enough to query the gateway is required to reconfigure or observe it:

| Route | Method | What it does |
|---|---|---|
| `/v1/fak/policy/reload` | `POST` | hot-reload the capability floor ([#229](https://github.com/anthony-chaudhary/fak/issues/229)) |
| `/v1/fak/trace/reset`   | `POST` | reset a trace's IFC taint high-water mark ([#230](https://github.com/anthony-chaudhary/fak/issues/230)) |
| `/metrics`              | `GET`  | Prometheus scrape ([#237](https://github.com/anthony-chaudhary/fak/issues/237)) |
| `/debug/vars`           | `GET`  | expvar-style diagnostics |

Witnesses 5–6 demonstrate this on `/metrics`: no token → `401`, correct token →
`200`. There is no separate "admin key" — the operator routes ride the one
`--require-key-env` secret, so locking the gateway locks its control plane too.

## What it does **not** do

- **It is a bearer-token gate, not mTLS.** One shared secret authenticates the
  *caller*; it does not authenticate the *server* to the client, nor pin a client
  certificate. There is no per-client identity, rotation, or revocation built in —
  rotate by changing `$FAK_TOKEN` and restarting.
- **It is not TLS.** `fak serve` speaks plain HTTP. A bearer token sent over
  cleartext is sniffable. **Terminate TLS at a real reverse proxy** (nginx,
  Caddy, Envoy, a cloud LB) in front of `fak`, and let the proxy add rate
  limiting, IP allow-lists, and request logging. Treat `--require-key-env` as the
  *last* line of defense behind that proxy, not the only one.
- **It is not a substitute for the policy floor.** Auth decides *who* may call;
  the `--policy` capability floor decides *what* any caller may do. Harden both —
  see [`../../docs/fak/policy-guide.md`](../../docs/fak/policy-guide.md).

## Where this fits

- **The flag, in context:** [`../../GETTING-STARTED.md`](../../GETTING-STARTED.md) §3 — "Auth" + "Harden it for real use"
- **Full serve/env reference:** [`../../docs/serve-config.md`](../../docs/serve-config.md)
- **The closed doc reference that shipped first:** [#138](https://github.com/anthony-chaudhary/fak/issues/138)
- **Operator routes that inherit this gate:** [#229](https://github.com/anthony-chaudhary/fak/issues/229) (reload) · [#230](https://github.com/anthony-chaudhary/fak/issues/230) (trace reset) · [#237](https://github.com/anthony-chaudhary/fak/issues/237) (observability)
