# Captured run — `examples/auth-hardening/run.sh`

A real run on a clean checkout (Go toolchain present, no model / key / GPU). The
script builds `fak`, starts `fak serve --require-key-env FAK_TOKEN`, runs the six
witnesses, and tears everything down.

```console
$ FAK_DEMO_PORT=8137 examples/auth-hardening/run.sh
[auth] building fak -> /tmp/tmp.y3lZEhLaXL/fak
[auth] starting gateway: fak serve http://127.0.0.1:8137  --require-key-env FAK_TOKEN
[auth] gateway healthy (auth=on): {"engine":"inkernel","model":"mock","ok":true,"planner":"mock"}

[auth] FOUR auth witnesses against the gated surface (GET /v1/models):
  ✓ 1. Authorization: Bearer <correct>             -> 200
  ✓ 2. Authorization: Bearer <wrong>               -> 401
  ✓ 3. no auth header                              -> 401
  ✓ 4. x-api-key: <correct>                        -> 200

[auth] operator-route witnesses — /metrics inherits the SAME gate:
  ✓ 5. GET /metrics (no header)                    -> 401
  ✓ 6. GET /metrics (Bearer correct)               -> 200

[auth] all six witnesses passed — both header shapes authenticate; both failure modes 401; operator route gated.
```

## The bodies behind the status codes

The witnesses assert on the HTTP status (the gate's verdict). For reference, the
actual response bodies:

**401 — missing or invalid credentials** (an OpenAI-style error envelope; both
fak-native and OpenAI clients understand it):

```console
$ curl -s http://127.0.0.1:8137/v1/models
{"error":{"code":null,"message":"missing or invalid credentials","param":null,"type":"authentication_error"}}
```

**200 — `GET /v1/models` with a valid credential** (no upstream model configured,
so the gateway reports its own `mock` planner — the witness exercises the gate,
not a model):

```console
$ curl -s -H "Authorization: Bearer $FAK_TOKEN" http://127.0.0.1:8137/v1/models
{"data":[{"id":"mock","object":"model","owned_by":"fak"}],"object":"list"}
```

**200 — `GET /metrics` with a valid credential** (the operator route inherits the
exact same gate):

```console
$ curl -s -H "Authorization: Bearer $FAK_TOKEN" http://127.0.0.1:8137/metrics | head -3
# HELP fak_gateway_up Whether the fak gateway process is scrapeable.
# TYPE fak_gateway_up gauge
fak_gateway_up 1
```

## Fail-closed on a missing secret

`--require-key-env VAR` names an **env var**, not a literal. If you pass the flag
but the var is unset/empty, `fak serve` refuses to start rather than silently
binding a network-facing gateway with no auth:

```console
$ unset FAK_TOKEN; fak serve --addr 0.0.0.0:8080 --require-key-env FAK_TOKEN
fak serve: --require-key-env FAK_TOKEN is set but unset/empty — refusing to start a network-facing gateway with NO authentication (set the secret or omit the flag)
$ echo $?
2
```
