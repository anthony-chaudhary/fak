---
title: "fak deployment guide: Docker, Kubernetes, bare metal"
description: "Production deployment for fak serve across Docker, Compose, Kubernetes, and bare metal, with a readiness checklist for auth, policy, and binding."
---

# fak Deployment Guide

Production deployment for `fak serve` — the kernel gateway that fronts a model
(local or remote) and adjudicates every proposed tool call before the client sees
it. This guide covers four targets — **container image / Docker**, **Docker
Compose**, **Kubernetes**, and **bare metal** — plus a **production-readiness
checklist** you should clear before exposing the gateway beyond loopback.

Every flag, env var, route, and default below is read from this repository
(`Dockerfile`, `install.sh`, `cmd/fak/main.go`, `internal/gateway/`). For the full
flag/env catalog see [server-config.md](server-config.md); for the fast local path
see [server-quickstart.md](server-quickstart.md); for the threat model and
hardening see [security.md](security.md).

> **The one rule that bites you first.** `fak serve` binds **loopback with no
> authentication** by default. On a non-loopback bind (`0.0.0.0`) with no key it
> still serves, but logs:
> `WARNING: binding 0.0.0.0:8080 with NO --require-key set — the kernel gateway is exposed without authentication`.
> Never run a network-facing gateway without `--require-key-env` and a policy
> floor. The [checklist](#production-readiness-checklist) makes this concrete.

---

## How fak runs in production

`fak serve` is a single static Go binary (CGO off, no shell, no libc) that
listens on one HTTP port (default `127.0.0.1:8080`) and exposes:

- OpenAI-compatible `/v1/chat/completions` and Anthropic `/v1/messages` (both
  adjudicated),
- fak-native `/v1/fak/*` (syscall, adjudicate, admit, policy reload, …),
- `/healthz` (always unauthenticated) and `/metrics` (Prometheus).

It runs in one of two modes:

| Mode | How | Footprint |
|---|---|---|
| **Proxy** | `--base-url` points at an upstream OpenAI/Anthropic/Gemini/xAI provider or a local server (Ollama, vLLM, llama.cpp). fak adjudicates; the upstream generates. | Light — CPU/RAM for HTTP + adjudication only. |
| **In-kernel** | `--gguf PATH` loads GGUF weights into the in-kernel engine; fak generates and adjudicates in one process. | Heavy — size RAM (and GPU, if used) to the model. |

Proxy mode is the common production shape and is what the Kubernetes and
bare-metal examples below use.

---

## Production readiness checklist

Clear every item before a network-facing deploy. Sources for each are in
[server-config.md](server-config.md) and [security.md](security.md).

- [ ] **Authentication on.** Set `--require-key-env VAR` with a strong secret in
  `$VAR` (e.g. `export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"`). Every route
  except `/healthz` then requires `Authorization: Bearer <key>` or
  `x-api-key: <key>`. An empty `$VAR` silently starts **unauthenticated** —
  confirm it is exported and non-empty in the serving process's environment.
- [ ] **Policy floor pinned.** Ship an explicit `--policy policy.json`. The floor
  is fail-closed: anything not affirmatively allowed and not explicitly denied
  resolves to `DEFAULT_DENY`. Validate with `fak policy --check policy.json`
  before it gates traffic.
- [ ] **Bind intentionally.** Use `--addr 0.0.0.0:8080` only behind a firewall,
  load balancer, or reverse proxy that terminates TLS and restricts ingress. fak
  speaks plain HTTP — put TLS in front (LB / Ingress / nginx).
- [ ] **Timeouts sized to the backend.** Keep the conservative defaults for a fast
  hosted upstream; raise `FAK_HTTP_WRITE_TIMEOUT_S` **and** `FAK_PLANNER_TIMEOUT_S`
  together for a slow local model (the write timeout must be ≥ the planner
  timeout). See [Timeout tuning](../../docs/serve-config.md#timeout-tuning-remote-upstream-vs-slow-local-model).
- [ ] **Audit journal enabled** (recommended). Set `FAK_AUDIT_JOURNAL=/path/to/audit.jsonl`
  to a durable, writable path for a tamper-evident record of every adjudicated
  syscall.
- [ ] **Rate limiting** (optional). `FAK_RATELIMIT_MAX_CALLS` / `FAK_RATELIMIT_MAX_COST`
  with `FAK_RATELIMIT_KEY` (`trace`|`tool`|`global`) cap per-key load.
- [ ] **Health + metrics wired.** Probe `/healthz`; scrape `/metrics`
  (Prometheus). See [observability.md](observability.md).
- [ ] **Run as non-root.** The container image already runs as `nonroot`; on bare
  metal use a dedicated service user (the systemd unit below uses `DynamicUser`).
- [ ] **Version pinned.** Pin a release (`FAK_VERSION` for the installer, an image
  tag for containers) rather than tracking `latest`. This guide tracks
  **v0.30.0**.

---

## 1. Container image (Docker)

The repo ships a production [`Dockerfile`](../../Dockerfile) at its root. It is a
two-stage build: stage one compiles `cmd/fak` static (`CGO_ENABLED=0`); the final
image is `gcr.io/distroless/static-debian12:nonroot` plus the single binary — no
shell, no package manager, runs as `nonroot`, exposes `8080`.

> **No public registry image yet.** There is no official image on a public
> registry; you build from this Dockerfile and push to a registry you control.
> Building the static binary is the documented Docker adopter path (the
> `static-binary / Docker` route).

### Build

```bash
# From a clone (repo root, where the Dockerfile lives):
docker build -t fak:0.30.0 .

# Stamp a specific version into the binary:
docker build --build-arg APP_VERSION=0.30.0 -t fak:0.30.0 .

# Without cloning — build straight from the Git remote:
docker build -t fak:0.30.0 https://github.com/anthony-chaudhary/fak.git
```

The default `CMD` is `serve --addr 0.0.0.0:8080` (containers must bind `0.0.0.0`,
not loopback). The `ENTRYPOINT` is the `fak` binary, so override the command to run
`agent`, `policy`, etc.

### Run

```bash
# Reach a model server running on the host (Ollama here) from the container:
docker run --rm -p 8080:8080 fak:0.30.0 serve --addr 0.0.0.0:8080 \
  --base-url http://host.docker.internal:11434/v1 \
  --model qwen2.5:1.5b
```

`host.docker.internal` resolves the host from inside the container on Docker
Desktop. On Linux, add `--add-host=host.docker.internal:host-gateway` or point
`--base-url` at the upstream's real address.

### Run hardened (auth + policy + audit)

The image runs as `nonroot` with no shell, so mount the policy file and pass
secrets via the environment:

```bash
docker run --rm -p 8080:8080 \
  -e FAK_GATEWAY_KEY="$(openssl rand -hex 32)" \
  -e OPENAI_API_KEY="sk-..." \
  -e FAK_AUDIT_JOURNAL=/var/lib/fak/audit.jsonl \
  -v "$PWD/policy.json:/etc/fak/policy.json:ro" \
  -v fak-audit:/var/lib/fak \
  fak:0.30.0 serve --addr 0.0.0.0:8080 \
    --provider openai --base-url https://api.openai.com/v1 \
    --model gpt-4o --api-key-env OPENAI_API_KEY \
    --policy /etc/fak/policy.json \
    --require-key-env FAK_GATEWAY_KEY
```

Verify:

```bash
curl -s http://127.0.0.1:8080/healthz                 # {"ok":true,...}  (no auth)
curl -s http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $FAK_GATEWAY_KEY"
```

---

## 2. Docker Compose

A minimal Compose stack with fak fronting a host Ollama, plus a place to grow into
the observability stack:

```yaml
# compose.yaml
services:
  fak:
    image: fak:0.30.0          # built from the repo Dockerfile, pushed to your registry
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - FAK_GATEWAY_KEY=${FAK_GATEWAY_KEY:?set a strong key}
      - FAK_AUDIT_JOURNAL=/var/lib/fak/audit.jsonl
      - FAK_HTTP_WRITE_TIMEOUT_S=300   # raise for a slow local model
      - FAK_PLANNER_TIMEOUT_S=300
    volumes:
      - ./policy.json:/etc/fak/policy.json:ro
      - fak-audit:/var/lib/fak
    extra_hosts:
      - "host.docker.internal:host-gateway"   # reach a host Ollama on Linux
    command:
      - serve
      - --addr=0.0.0.0:8080
      - --base-url=http://host.docker.internal:11434/v1
      - --model=qwen2.5:1.5b
      - --policy=/etc/fak/policy.json
      - --require-key-env=FAK_GATEWAY_KEY

volumes:
  fak-audit:
```

```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"
docker compose up -d
```

For Prometheus + Grafana, the repo already ships a ready stack at
[`tools/grafana/docker-compose.yml`](../../tools/grafana/docker-compose.yml) that
scrapes `fak serve` on `:8080`; see [observability.md](observability.md).

---

## 3. Kubernetes

The example below runs proxy-mode fak as a stateless `Deployment` — the secret in a
`Secret`, the policy in a `ConfigMap`, `/healthz` driving the probes, and a
hardened `securityContext` that matches the distroless `nonroot` image. Apply it to
your cluster after pushing the image to a registry the cluster can pull from.

> TLS belongs at the edge. Terminate TLS at your Ingress / load balancer and route
> cleartext HTTP to the `Service` — fak speaks plain HTTP.

```yaml
# fak.yaml
apiVersion: v1
kind: Secret
metadata:
  name: fak-secrets
type: Opaque
stringData:
  # Generate with: openssl rand -hex 32
  gateway-key: "REPLACE_WITH_A_STRONG_KEY"
  # Upstream provider key, if using a hosted model:
  openai-api-key: "sk-..."
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: fak-policy
data:
  policy.json: |
    {
      "version": "fak-policy/v1",
      "posture": "fail_closed",
      "allow_prefix": ["read_", "get_", "list_", "search_"],
      "deny": { "bash": "SINK_EXEC", "write_file": "SINK_EGRESS" },
      "redact_fields": ["api_key", "token", "password"]
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fak
  labels: { app: fak }
spec:
  replicas: 2
  selector:
    matchLabels: { app: fak }
  template:
    metadata:
      labels: { app: fak }
    spec:
      securityContext:
        runAsNonRoot: true
      containers:
        - name: fak
          image: REGISTRY/fak:0.30.0
          args:
            - serve
            - --addr=0.0.0.0:8080
            - --provider=openai
            - --base-url=https://api.openai.com/v1
            - --model=gpt-4o
            - --api-key-env=OPENAI_API_KEY
            - --policy=/etc/fak/policy.json
            - --require-key-env=FAK_GATEWAY_KEY
          ports:
            - containerPort: 8080
          env:
            - name: FAK_GATEWAY_KEY
              valueFrom: { secretKeyRef: { name: fak-secrets, key: gateway-key } }
            - name: OPENAI_API_KEY
              valueFrom: { secretKeyRef: { name: fak-secrets, key: openai-api-key } }
          volumeMounts:
            - name: policy
              mountPath: /etc/fak
              readOnly: true
          livenessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            initialDelaySeconds: 3
            periodSeconds: 5
          resources:
            requests: { cpu: "250m", memory: "128Mi" }
            limits:   { cpu: "1",    memory: "512Mi" }
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
      volumes:
        - name: policy
          configMap: { name: fak-policy }
---
apiVersion: v1
kind: Service
metadata:
  name: fak
spec:
  selector: { app: fak }
  ports:
    - port: 80
      targetPort: 8080
```

```bash
kubectl apply -f fak.yaml
kubectl rollout status deploy/fak
kubectl port-forward svc/fak 8080:80      # local smoke test
curl -s http://127.0.0.1:8080/healthz
```

Notes:

- **`/healthz` answers `200 {"ok":true,...}` once the listener is bound and the
  model is loaded**, so it is a valid liveness *and* readiness signal. It is the
  only health route; there is no separate `/readyz`. It is always unauthenticated,
  so probes need no token.
- **Resource requests are starting points** for proxy mode. Tune from real
  `/metrics`. **In-kernel mode** (`--gguf`) is a different shape: mount the weights
  via a `PersistentVolume`, size memory (and GPU) to the model, raise the timeouts
  (below), and expect a longer `initialDelaySeconds` for the model load.
- **`readOnlyRootFilesystem: true`** is safe because the binary needs no writable
  root. If you enable `FAK_AUDIT_JOURNAL`, mount a writable volume for it and point
  the path there.
- **Reload policy without a restart:** edit the `ConfigMap`, let the mount refresh,
  then `POST /v1/fak/policy/reload` (with the bearer token) to each pod. The reload
  re-reads the same file passed to `--policy` and keeps the warm caches.

For a slow in-kernel/CPU model, add to the container `env` (write timeout ≥ planner
timeout):

```yaml
            - { name: FAK_HTTP_WRITE_TIMEOUT_S, value: "600" }
            - { name: FAK_PLANNER_TIMEOUT_S,    value: "600" }
```

---

## 4. Bare metal

### Install the binary

**One-line installer** (downloads the prebuilt static binary, verifies its
SHA-256, installs to PATH — no Go, no clone):

```bash
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

Installer knobs (environment): `FAK_VERSION` pins a version (e.g. `0.30.0`;
default latest release), `FAK_INSTALL_DIR` sets the target (default
`/usr/local/bin` if writable, else `~/.local/bin`).

```bash
FAK_VERSION=0.30.0 FAK_INSTALL_DIR=/usr/local/bin \
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh)"
fak version
```

**Prebuilt release assets.** Releases attach static binaries for **linux/amd64**,
**darwin/amd64**, **darwin/arm64**, and **windows/amd64** (each with a `.sha256`,
plus an aggregate `SHA256SUMS`) at
<https://github.com/anthony-chaudhary/fak/releases/latest>. The
`curl | sh` installer covers macOS (amd64/arm64) and linux/amd64; Windows users
download the `.zip` manually.

**linux/arm64 has no prebuilt binary** — build it from source:

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak                        # the Go module is the repository root
go build -trimpath -o /usr/local/bin/fak ./cmd/fak   # Go 1.26+, auto-fetched via GOTOOLCHAIN=auto
```

### Run as a service (systemd)

Store secrets in a root-only environment file, then run the gateway as an
unprivileged dynamic user:

```ini
# /etc/fak/fak.env   (chmod 600, root-owned)
FAK_GATEWAY_KEY=<openssl rand -hex 32 output>
OPENAI_API_KEY=sk-...
FAK_HTTP_WRITE_TIMEOUT_S=300
FAK_PLANNER_TIMEOUT_S=300
FAK_AUDIT_JOURNAL=/var/lib/fak/audit.jsonl
```

```ini
# /etc/systemd/system/fak.service
[Unit]
Description=fak serve — agent tool-call adjudication gateway
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/fak/fak.env
ExecStart=/usr/local/bin/fak serve --addr 0.0.0.0:8080 \
  --provider openai --base-url https://api.openai.com/v1 \
  --model gpt-4o --api-key-env OPENAI_API_KEY \
  --policy /etc/fak/policy.json \
  --require-key-env FAK_GATEWAY_KEY
Restart=on-failure
RestartSec=2
# Run unprivileged with a hardened sandbox.
DynamicUser=yes
StateDirectory=fak                 # /var/lib/fak for the audit journal
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

```bash
fak policy --check /etc/fak/policy.json     # validate before enabling
sudo systemctl daemon-reload
sudo systemctl enable --now fak
curl -s http://127.0.0.1:8080/healthz
journalctl -u fak -f                         # watch for the no-auth WARNING
```

### Bare-metal with a real local model (Ollama + fak)

The common single-box pattern is a model server on the GPU with fak adjudicating in
front of it. Run the model server, warm it, then start fak as the service above but
pointed at the local server:

```bash
# Model server (separate process / unit):
ollama serve
ollama pull qwen2.5:14b

# fak in front (point --base-url at the local server; raise timeouts for big models):
FAK_HTTP_WRITE_TIMEOUT_S=300 FAK_PLANNER_TIMEOUT_S=300 \
fak serve --addr 0.0.0.0:8080 \
  --provider openai --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:14b \
  --policy /etc/fak/policy.json \
  --require-key-env FAK_GATEWAY_KEY
```

The same shape runs the **in-kernel** engine instead of an upstream — drop
`--base-url` and pass `--gguf PATH` (the GGUF's embedded tokenizer is used by
default); see [server-quickstart.md](server-quickstart.md) Scenario 4.

---

## Operating the deployment

**Verify after every deploy:**

```bash
KEY="$FAK_GATEWAY_KEY"
curl -s http://HOST:8080/healthz                                   # liveness (no auth)
curl -s http://HOST:8080/v1/models -H "Authorization: Bearer $KEY" # auth works
curl -s http://HOST:8080/metrics -H "Authorization: Bearer $KEY"   # Prometheus scrape
# Prove the floor: an allow-listed read vs a denied exec.
curl -s -X POST http://HOST:8080/v1/fak/adjudicate \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"tool":"Bash","arguments":{"command":"git push origin main"}}'
```

**Upgrades.** Pin a new release tag / image and roll forward (`kubectl set image`
or pull the new tag and restart the unit). Policy changes need no restart — rewrite
the policy file and `POST /v1/fak/policy/reload`.

**Observability.** Scrape `/metrics` for verdict counts (`fak_verdict_total`),
operation latency (`fak_operation_duration_seconds`), and startup/model-load
timings. The repo's [`tools/grafana/`](../../tools/grafana/docker-compose.yml)
stack wires Prometheus + Grafana to a `fak serve` on `:8080`. Full details in
[observability.md](observability.md).

**Troubleshooting.** Slow models tripping a timeout, auth rejections, and bind
errors are covered in [server-troubleshooting.md](server-troubleshooting.md).

---

## See also

- [server-quickstart.md](server-quickstart.md) — fastest path to a running gateway
- [server-config.md](server-config.md) — every flag, env var, route, and default
- [security.md](security.md) — threat model and hardening for a network deploy
- [observability.md](observability.md) — metrics, logs, and traces
- [server-troubleshooting.md](server-troubleshooting.md) — when something breaks
- [policy-guide.md](policy-guide.md) and [`fak/POLICY.md`](../../POLICY.md) —
  authoring the capability floor and the refusal vocabulary
- [`Dockerfile`](../../Dockerfile) and [`install.sh`](../../install.sh) — the
  build and install sources this guide describes
