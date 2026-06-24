---
title: "fak on Kubernetes — copy-paste manifest"
description: "A committed, apply-ready Kubernetes manifest for fak serve in proxy mode: Secret, policy ConfigMap, hardened Deployment, and Service, with the auth rule and a smoke test."
---

# fak on Kubernetes

A committed, apply-ready manifest for running `fak serve` — the kernel gateway that
adjudicates every proposed tool call — on Kubernetes in **proxy mode** (fak
adjudicates; an upstream OpenAI/Anthropic/local model generates).

This is the artifact the deployment guide's
[Kubernetes section](../../docs/fak/deployment-guide.md#3-kubernetes) describes,
materialized as files you can `kubectl apply` directly instead of copying out of a
fenced block. **The [deployment guide](../../docs/fak/deployment-guide.md) is the
source of truth** for every flag, env var, route, and default — read it for the
production-readiness checklist before exposing the gateway beyond loopback.

## What's here

| File | What it is |
|---|---|
| [`fak.yaml`](fak.yaml) | The whole stack: a `Secret` (gateway + provider keys), a policy `ConfigMap`, a hardened `Deployment`, and a `Service`. |
| [`kustomization.yaml`](kustomization.yaml) | Wraps `fak.yaml` so `kubectl apply -k deploy/k8s` works and you can patch image / namespace / replicas without editing the manifest. |

The committed `Deployment` adds two hardenings on top of the guide's example —
`seccompProfile: RuntimeDefault` and `automountServiceAccountToken: false` (fak never
calls the Kubernetes API) — and otherwise matches it field-for-field.

## The one rule that bites first

A network-facing gateway **must** set `--require-key-env` and pin a policy floor.
`/healthz` is the only unauthenticated route (it drives the probes); every other
route requires `Authorization: Bearer <key>`. **TLS belongs at the edge** — terminate
it at your Ingress / load balancer and route cleartext HTTP to the `Service`; fak
speaks plain HTTP.

## Apply

1. **Build and push the image.** There is no public registry image yet — build the
   repo [`Dockerfile`](../../Dockerfile) and push to a registry your cluster can pull
   from, then point `image:` (or the kustomize `images:` override) at it:

   ```bash
   docker build -t REGISTRY/fak:0.32.0 .
   docker push REGISTRY/fak:0.32.0
   ```

2. **Fill the Secret.** Replace the placeholders in `fak.yaml` (or, in production,
   prefer a real secret manager / `kubectl create secret generic fak-secrets ...`):

   ```bash
   openssl rand -hex 32   # use this for gateway-key
   ```

3. **Apply** (either form):

   ```bash
   kubectl apply -k deploy/k8s            # via kustomize
   kubectl apply -f deploy/k8s/fak.yaml   # the raw manifest
   kubectl rollout status deploy/fak
   ```

4. **Smoke test:**

   ```bash
   kubectl port-forward svc/fak 8080:80
   curl -s http://127.0.0.1:8080/healthz   # {"ok":true,...}  (no auth)
   ```

## Notes

- **Proxy vs in-kernel.** This manifest is proxy mode. For **in-kernel** mode
  (`--gguf`), mount the weights via a `PersistentVolume`, size memory (and GPU) to the
  model, raise `FAK_HTTP_WRITE_TIMEOUT_S` / `FAK_PLANNER_TIMEOUT_S` (write ≥ planner),
  and expect a longer `initialDelaySeconds`. See the guide.
- **Reload policy without a restart.** Edit the `fak-policy` ConfigMap, let the mount
  refresh, then `POST /v1/fak/policy/reload` (with the bearer token) to each pod.
- **No Helm chart yet.** This raw manifest (+ kustomize) is the supported path; a Helm
  chart is not shipped.

## See also

- [deployment-guide.md](../../docs/fak/deployment-guide.md) — Docker, Compose, k8s, bare metal, and the readiness checklist
- [server-config.md](../../docs/fak/server-config.md) — every flag, env var, route, and default
- [security.md](../../docs/fak/security.md) — threat model and hardening for a network deploy
