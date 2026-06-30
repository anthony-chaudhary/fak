---
title: "Run fak in front of llm-d"
description: "First-class fak support for llm-d: front the llm-d Gateway API OpenAI-compatible route, use the registered llm-d engine id for syscall/model-route dispatch, and keep metrics/KV ownership honest."
---

# Run fak in front of llm-d

llm-d is a Kubernetes-native serving stack. It exposes an OpenAI-compatible route through
Gateway API / Endpoint Picker Provider routing, then runs vLLM workers behind that control
plane. fak's support is deliberately a ride-mode integration:

- llm-d owns model serving, P/D placement, EPP scheduling, and any KV/cache policy behind the
  cluster route.
- fak sits in front as the governance gateway: capability floor, result quarantine,
  audit trail, and optional syscall/model-route dispatch.
- The registered engine id is `llm-d`, with `FAK_LLMD_*` env vars. `FAK_LLM_D_*` aliases
  are accepted for operators who prefer the upstream spelling.

## 1. Find the llm-d OpenAI route

Use the route your llm-d deployment exposes for Chat Completions. In a local smoke test,
port-forward the Gateway/Service and confirm the OpenAI-compatible surface answers:

```bash
kubectl -n <llm-d-namespace> get gateway,httproute,svc
kubectl -n <llm-d-namespace> port-forward svc/<llm-d-gateway-service> 18080:80

curl -s http://127.0.0.1:18080/v1/models
```

If your cluster terminates TLS or uses a hostname, keep that external route and use its
`/v1` root as the fak upstream.

## 2. Prove the route with fak

Before putting agents behind the route, run fak's llm-d smoke witness. It checks the
OpenAI-compatible model list, sends one streamed Chat Completions request with
`stream:true`, waits for the `[DONE]` sentinel, and optionally normalizes the metrics
endpoint as `engine="llm-d"`:

```bash
fak llmd-smoke \
  --base-url http://127.0.0.1:18080/v1 \
  --model <served-model> \
  --metrics-url http://127.0.0.1:18080/metrics
```

If the llm-d route requires a bearer token, put it in an env var and pass the env var
name, not the secret value:

```bash
export LLMD_ROUTE_TOKEN="<token>"
fak llmd-smoke \
  --base-url https://<llm-d-host>/v1 \
  --model <served-model> \
  --api-key-env LLMD_ROUTE_TOKEN \
  --json
```

`--base-url` defaults to `FAK_LLMD_BASE_URL` / `FAK_LLM_D_BASE_URL`, `--model` defaults
to `FAK_LLMD_MODEL` / `FAK_LLM_D_MODEL` or the first `/v1/models` id, and `--metrics-url`
defaults to `FAK_LLMD_METRICS_URL` / `FAK_LLM_D_METRICS_URL` when set.

## 3. Normal chat proxy mode

For agents and SDKs that speak OpenAI Chat Completions, start fak in front of llm-d:

```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"

fak serve --addr 0.0.0.0:8080 \
  --provider openai \
  --base-url http://127.0.0.1:18080/v1 \
  --model <served-model> \
  --policy floor.json \
  --require-key-env FAK_GATEWAY_KEY
```

Then point the client at fak, not llm-d:

```bash
export OPENAI_BASE_URL="http://<fak-host>:8080/v1"
export OPENAI_API_KEY="$FAK_GATEWAY_KEY"
```

This is the path most deployments want. The request body and model id go through unchanged
except for fak's adjudication/quarantine envelope, and llm-d still owns scheduling behind
its Gateway API route.

## 4. Registered engine mode

Use the `llm-d` engine id when a fak route manifest or `fak_syscall` path should dispatch
an admitted call through llm-d instead of the in-kernel engine:

```bash
export FAK_LLMD_BASE_URL="http://127.0.0.1:18080/v1"
export FAK_LLMD_MODEL="<served-model>"
# Optional:
export FAK_LLMD_API_KEY="<llm-d-upstream-bearer-if-needed>"
export FAK_LLMD_METRICS_URL="http://127.0.0.1:18080/metrics"

fak serve --engine llm-d --model "<served-model>"
```

The adapter streams via `/v1/chat/completions` or `/v1/completions`, forces
`stream:true`, and reports result metadata with `engine="llm-d"`. That makes route
manifests and audit records name the real serving control plane instead of collapsing it
into a generic vLLM label.

For a concrete route-manifest starting point, validate the shipped preset and run serve
with it:

```bash
fak route --check examples/routing-presets/llm-d.json
fak serve --engine llm-d \
  --route-manifest examples/routing-presets/llm-d.json \
  --model "<served-model>"
```

The preset defaults governed tool-call dispatch to `llm-d` and keeps common
sensitivity labels (`tenant`, `pii`, `secret`) on `inkernel`. The residency gate still
fails closed for any sensitive/tenant-scoped call that reaches the remote `llm-d` route.

## 5. Metrics and KV boundary

The `llm-d` adapter normalizes vLLM-style Prometheus worker signals under
`engine="llm-d"` when `FAK_LLMD_METRICS_URL` is set or the metrics endpoint is reachable
next to the `/v1` route. It does not import llm-d internals.

Remote KV eviction stays conservative. llm-d may route across vLLM workers and cache tiers,
but fak only claims whole-prefix remote invalidation where a public control endpoint proves
it. Do not use `--engine-cache-require-exact-span` through llm-d unless you have installed
a separate exact-span adapter that can witness the span. Without that witness, fak fails
closed rather than pretending it deleted a middle KV span inside the llm-d fleet.

## References

- [llm-d upstream](https://github.com/llm-d/llm-d)
- [llm-d architecture](https://llm-d.ai/docs/architecture)
- [llm-d EPP router docs](https://llm-d.ai/docs/architecture/core/router/epp)
- [Supported serving engines](../supported/engines.md)
- [Compatibility matrix](compatibility-matrix.md)
