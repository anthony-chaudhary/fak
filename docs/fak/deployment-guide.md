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

## Capacity & sizing (how big a box, and when it saturates)

Two questions decide the box: **how many concurrent governed sessions does this host
hold**, and **how do I see it saturating before the in-flight loops degrade?** fak
answers the second directly — configure a ceiling and the runtime publishes live
saturation and backpressures past it — and gives you the inputs to answer the first.

### The sizing model

A host holds `N` concurrent governed sessions, bounded by whichever resource runs out
first:

```
N = min( RAM_usable / per_session_RAM , cores × loops_per_core )
```

The two serving shapes have very different `per_session_RAM` and CPU profiles:

| Shape | What each session costs | Dominant limit | Where the numbers come from |
|---|---|---|---|
| **Proxy** (`--base-url`) | HTTP + adjudication + per-session cache/ctxmmu state on the fak box; the model (weights + KV) lives upstream. Per-session RAM is small (low MB); throughput is CPU-bound on adjudication. | cores × `loops_per_core` | Measure per-session overhead with #3292's governed-overhead harness; watch `fak_gateway_operation_duration_seconds` for the per-op CPU cost. |
| **In-kernel** (`--gguf`) | Resident model weights (shared across all sessions) **plus** a per-session KV cache = context_tokens × bytes/token. Per-session RAM is dominated by KV. | `RAM_usable / per_session_KV` (and VRAM, if GPU) | `fak serve --gguf … --plan-json` prints the header-derived weight + KV footprint before any load (#4361); size `per_session_RAM` from its KV demand at your context budget. |

**Illustrative only** — replace with a measured run on *your* host and model (the
numbers below are shape, not a claim about any host):

| Shape | Host | per-session RAM | Derived `N` | Set `FAK_MAX_SESSIONS` |
|---|---|---|---|---|
| Proxy | 8 vCPU / 16 GB | ~8 MB | CPU-bound: ~4 loops/core → ~32 | `32` |
| In-kernel, 7B Q4, 8k ctx | 16-core / 64 GB + 24 GB VRAM | ~1.1 GB KV | (64−`weights`) / 1.1 → size to VRAM/RAM | measured `N` |

Derive the real figures from #3292's harness (per-session overhead), `--plan-json`
(memory footprint), and the live saturation gauge below — do not ship these placeholders
as a capacity promise.

### The live saturation signal

Set the derived ceiling and the runtime exposes headroom and fails **closed** past it —
no silent cap:

- **Configure** the ceiling: `FAK_MAX_SESSIONS=<N>` in the serving environment. Unset or
  `0` is unbounded (the historical default — no gauge, no backpressure).
- **Watch** it on `/metrics`: `fak_gateway_session_saturation` (live governed sessions ÷
  ceiling, in `[0,1+]`), alongside `fak_gateway_session_ceiling` and
  `fak_gateway_sessions_live`. Alert/scale out as it approaches `1.0`.
- **Or probe** it on the readiness surface: `/healthz` carries the same readout as a
  `session_saturation` object (`ceiling`, `live`, `ratio`, `headroom`, `bounded`).
  Saturation does **not** flip `ok` to `false` — a box at its ceiling is healthy, it is
  shedding new load by design.
- **Backpressure** past the ceiling: a **new** session's admission is refused with
  HTTP `503` + `Retry-After` and the closed reason `SESSION_CEILING_SATURATED`
  (`dos man wedge SESSION_CEILING_SATURATED --explain`). Sessions **already in flight are never
  refused** — saturation sheds *new* load rather than degrading the loops already running.

This is the deployment-boundary projection of the fleet's admission machinery, scoped to
the single all-in-one deployment. It is inert until you set a ceiling, so it never
surprises an existing deployment.

> **Generation:** `gen/next` foundation. The saturation gauge, the ceiling, and the
> backpressure are shipped and gated behind `FAK_MAX_SESSIONS`; the *measured* sizing
> table is the promotion evidence still owed — replace the illustrative rows above with
> witnessed per-session overhead/KV numbers from a real run. Invalidating assumption: that
> the governed-session count (`fak_sessions`) is the right saturation axis for your
> workload — if a single session fans out to many concurrent upstream requests, size on
> request concurrency (the native scheduler's `fak_sched_*` family) instead.

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
- [ ] **Session ceiling sized** (optional, recommended for the all-in-one). Set
  `FAK_MAX_SESSIONS=<N>` from the [Capacity & sizing](#capacity--sizing-how-big-a-box-and-when-it-saturates)
  model and alert on `fak_gateway_session_saturation` before it hits `1.0`. Past the
  ceiling, new sessions get `503` + `SESSION_CEILING_SATURATED`; in-flight loops are
  never sacrificed. Unset = unbounded.
- [ ] **Run as non-root.** The container image already runs as `nonroot`; on bare
  metal use a dedicated service user (the systemd unit below uses `DynamicUser`).
- [ ] **Version pinned.** Pin a release (`FAK_VERSION` for the installer, an image
  tag for containers) rather than tracking `latest`. This guide tracks
  **v0.46.0**.

---

## 1. Container image (Docker)

The repo ships a production [`Dockerfile`](https://github.com/anthony-chaudhary/fak/blob/main/Dockerfile) at its root. It is a
two-stage build: stage one compiles `cmd/fak` static (`CGO_ENABLED=0`); the final
image is `gcr.io/distroless/static-debian12:nonroot` plus the single binary — no
shell, no package manager, runs as `nonroot`, exposes `8080`.

> **Published image.** Every `v*` tag publishes
> `ghcr.io/anthony-chaudhary/fak:<version>` and `:latest` to GitHub Container
> Registry ([`release-container.yml`](https://github.com/anthony-chaudhary/fak/blob/main/.github/workflows/release-container.yml)),
> and the package is public — no login needed to pull it. Building the
> Dockerfile yourself is the alternative for an air-gapped cluster or a
> private-registry mirror.

### Pull (recommended)

```bash
docker pull ghcr.io/anthony-chaudhary/fak:0.46.0
# or track the moving tag:
docker pull ghcr.io/anthony-chaudhary/fak:latest
```

### Build (alternative: air-gapped / private registry)

```bash
# From a clone (repo root, where the Dockerfile lives):
docker build -t fak:0.46.0 .

# Stamp a specific version into the binary:
docker build --build-arg APP_VERSION=0.46.0 -t fak:0.46.0 .

# Without cloning — build straight from the Git remote:
docker build -t fak:0.46.0 https://github.com/anthony-chaudhary/fak.git
```

The default `CMD` is `serve --addr 0.0.0.0:8080` (containers must bind `0.0.0.0`,
not loopback). The `ENTRYPOINT` is the `fak` binary, so override the command to run
`agent`, `policy`, etc.

### Run

```bash
# Reach a model server running on the host (Ollama here) from the container:
docker run --rm -p 8080:8080 fak:0.46.0 serve --addr 0.0.0.0:8080 \
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
  fak:0.46.0 serve --addr 0.0.0.0:8080 \
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

### GPU image (in-kernel CUDA serving)

The default image above is GPU-free (proxy mode only). To run the **in-kernel**
engine (`--gguf ... --engine inkernel --backend cuda`) on a rented or local GPU,
use [`Dockerfile.cuda`](https://github.com/anthony-chaudhary/fak/blob/main/Dockerfile.cuda)
instead — a CUDA-runtime variant of the same two-stage build, published alongside
the default image as the `-cuda` tag:

```bash
# Pull (a tagged release publishes both):
docker pull ghcr.io/anthony-chaudhary/fak:0.46.0-cuda

# Or build locally, picking the arch that matches the rented card
# (sm_80 A100, sm_89 Ada/L4 default, sm_90 H100/H200, sm_100 B200/GB200):
docker build -f Dockerfile.cuda --build-arg CUDA_ARCH=sm_90 -t fak:cuda .

# Run with the NVIDIA Container Toolkit (--gpus requires it on the host):
docker run --rm --gpus all -p 8080:8080 \
  -v /models:/models:ro \
  ghcr.io/anthony-chaudhary/fak:0.46.0-cuda \
  serve --addr 0.0.0.0:8080 --gguf /models/model.gguf \
  --engine inkernel --backend cuda
```

This image does **not** support sm_120 (RTX 50-series / consumer Blackwell) —
`internal/compute` has no sm_120 kernel variant yet, so rent a datacenter card
(A100/L4/H100/H200/B200/GB200). Model coverage in-kernel is exactly what
[`internal/compute`](https://github.com/anthony-chaudhary/fak/tree/main/internal/compute)
already proves — this image ships the runtime, not new kernel coverage.

---

## 2. Docker Compose

A minimal Compose stack with fak fronting a host Ollama, plus a place to grow into
the observability stack:

```yaml
# compose.yaml
services:
  fak:
    image: ghcr.io/anthony-chaudhary/fak:0.46.0   # published image; swap for your own to build from source
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
[`tools/grafana/docker-compose.yml`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/docker-compose.yml) that
scrapes `fak serve` on `:8080`; see [observability.md](observability.md).

---

## 3. Kubernetes

The example below runs proxy-mode fak as a stateless `Deployment` — the secret in a
`Secret`, the policy in a `ConfigMap`, `/healthz` driving the probes, and a
hardened `securityContext` that matches the distroless `nonroot` image. It
defaults to the published `ghcr.io/anthony-chaudhary/fak` image; point `image:`
at your own registry instead if you build from source.

> This manifest is also **committed at [`deploy/k8s/`](https://github.com/anthony-chaudhary/fak/tree/main/deploy/k8s)**
> — apply it directly with `kubectl apply -k deploy/k8s` (or `kubectl apply -f
> deploy/k8s/fak.yaml`) after filling the `Secret`. It pulls the published ghcr
> image by default; override `image:` (or the kustomize `images:` patch) to
> point at your own registry. See [`deploy/k8s/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/deploy/k8s/README.md).

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
      "deny": { "bash": "POLICY_BLOCK", "write_file": "POLICY_BLOCK" },
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
          image: ghcr.io/anthony-chaudhary/fak:0.46.0   # or REGISTRY/fak:0.46.0 if you build from source
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

Installer knobs (environment): `FAK_VERSION` pins a version (e.g. `0.46.0`;
default latest release), `FAK_INSTALL_DIR` sets the target (default
`/usr/local/bin` if writable, else `~/.local/bin`).

```bash
FAK_VERSION=0.46.0 FAK_INSTALL_DIR=/usr/local/bin \
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh)"
fak version
```

**Prebuilt release assets.** Releases attach static binaries for **linux/amd64**,
**linux/arm64**, **darwin/amd64**, **darwin/arm64**, and **windows/amd64** (each with a
`.sha256`, plus an aggregate `SHA256SUMS`) at
<https://github.com/anthony-chaudhary/fak/releases/latest>. The
`curl | sh` installer covers macOS (amd64/arm64) and linux (amd64/arm64); Windows users
download the `.zip` manually.

**linux/arm64 is a first-class published target** — the same pure-Go binary on a
Raspberry Pi / Jetson / arm64 edge gateway as on a datacenter host (`CGO_ENABLED=0`, so
nothing to port). Install it the same way as any other target (the one-line installer, or
the manual download). To build a specific commit from source instead:

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

**Observability.** Scrape `/metrics` for verdict counts (`fak_gateway_operations_total`),
operation latency (`fak_gateway_operation_duration_seconds`), and startup/model-load
timings. The repo's [`tools/grafana/`](https://github.com/anthony-chaudhary/fak/blob/main/tools/grafana/docker-compose.yml)
stack wires Prometheus + Grafana to a `fak serve` on `:8080`. Full details in
[observability.md](observability.md).

**Troubleshooting.** Slow models tripping a timeout, auth rejections, and bind
errors are covered in [server-troubleshooting.md](server-troubleshooting.md).

---

## Forge-side enforcement (required)

fak's client-side hook floor (`tools/githooks/{pre-commit,pre-push,commit-msg,reference-transaction}`)
enforces the trunk laws — `OFF_TRUNK`, DCO sign-off, the Conventional-Commits
`(fak <leaf>)` stamp, and the leak scan — but it lives **inside each clone and is
bypassable by design.** Three independent client-side escapes defeat it: `--no-verify`
skips client hooks; a `core.hooksPath` override repoints them at an empty directory; and
shell-laundering (`alias`, a wrapper script, `eval`, `$()`, backticks) evades the
conservative argv tokenizer so a laundered `git push --force` never presents recognizable
argv to fak at all. `internal/gitgate` refuses `--no-verify` and the `core.hooksPath`
knob precisely *because* hooks are bypassable — but that refusal is itself a per-process
client-side check.

A **forge-side ruleset never sees client argv.** It evaluates the actual ref update the
forge receives, after all laundering has collapsed into a concrete
`<old-sha> <new-sha> <refname>`. That is the one layer fak structurally cannot reach from
inside the clone, and it is where a fleet's trunk laws need a backstop no client can
disarm. **For a multi-tenant fleet the trunk guarantees only hold if the forge ruleset is
also applied.** The client floor is best-effort defense-in-depth; the ruleset is the
non-bypassable companion.

The templates and a one-command apply wrapper live in
[`tools/forge-rulesets/`](https://github.com/anthony-chaudhary/fak/tree/main/tools/forge-rulesets):

```bash
# GitHub (needs `gh auth login`); edit the selected status-check contexts first:
tools/forge-rulesets/apply.sh github  <owner>/<repo> current

# Branch-regime cutover: create dev first, then apply the role-specific templates:
tools/forge-rulesets/apply.sh github  <owner>/<repo> dev
tools/forge-rulesets/apply.sh github  <owner>/<repo> main

# GitLab (needs GITLAB_TOKEN with api scope):
tools/forge-rulesets/apply.sh gitlab  <project-id>
```

- `github-ruleset.json` — the current no-cutover Repository Ruleset targeting `main`:
  non-fast-forward (no force-push), deletion protection, required linear history, required
  signatures, and required status checks (`ci`). Mirrors `OFF_TRUNK` and the no-force-push
  law server-side for today's single hot trunk.
- `github-dev-ruleset.json` — the branch-regime development template targeting `dev`:
  deletion and force-push protection, linear history, signatures, and the development CI
  check (`ci-fast`). This is the high-churn integration branch ordinary workers target
  after `[branch_roles].development_branch` moves to `dev`.
- `github-main-ruleset.json` — the branch-regime public/release template targeting `main`:
  deletion and force-push protection, linear history, signatures, and release/front-door
  checks (`ci`, `release-artifacts`). Ordinary workers should not push to this branch; if
  the forge needs a release-promotion bot/app exception, add that bypass actor explicitly
  before active enforcement.
- `gitlab-push-rules.json` — Push Rules with a `commit_message_regex` mirroring the
  Conventional-Commits + `(fak <leaf>)` stamp (the same shape `tools/commit_stamp_doctor.py`
  recognizes) and `prevent_secrets` mirroring the leak scan.
- `apply.sh` — the `gh api` / GitLab Push Rules API wrappers plus a Terraform stub so the
  ruleset can live in IaC and not drift silently.

Branch-regime ordering is deliberate:

1. Create `dev` from the verified current trunk.
2. Apply `github-dev-ruleset.json` to protect the hot development branch.
3. Shadow-run agents and CI while `[branch_roles]` still names the no-cutover regime.
4. Switch `[branch_roles].development_branch` and worker prompts to `dev`.
5. Apply `github-main-ruleset.json` only after the release-promotion path is ready, so
   `main` becomes the clean public front door instead of an ordinary worker target.

This is pure defense-in-depth that **composes with, and does not overlap,** fak's core
value: fak adjudicates *before the call runs* (it refuses a hazard with a reason,
in-process, no round-trip); the ruleset adjudicates *the resulting ref update at the
forge* (it cannot reason about intent or refuse pre-call, but it cannot be laundered).
Neither replaces the other.

**Per-forge parity residual.** GitHub Rulesets and GitLab Push Rules do not express an
identical predicate set. The commit-message regex is a first-class Push Rule on GitLab but
is status-check / signature-shaped on GitHub; conversely no-force-push and linear-history
are Ruleset rules on GitHub but **protected-branch** settings on GitLab (configured
separately from push rules — see the note in `gitlab-push-rules.json`). The template
mirrors each law on whichever forge can express it and documents the residual; it does not
promise a byte-identical mirror of every client hook. It also does **not** make fak's
guarantees cross-clone or atomic — a ruleset validates a single ref update per repository;
cross-machine commit atomicity remains the collective-commit barrier's separate concern.

---

## See also

- [server-quickstart.md](server-quickstart.md) — fastest path to a running gateway
- [server-config.md](server-config.md) — every flag, env var, route, and default
- [security.md](security.md) — threat model and hardening for a network deploy
- [observability.md](observability.md) — metrics, logs, and traces
- [hosted-control-plane.md](hosted-control-plane.md) — architecture brief (RFC) for a multi-tenant hosted policy + audit control plane over the audit stream the binary emits
- [server-troubleshooting.md](server-troubleshooting.md) — when something breaks
- [policy-guide.md](policy-guide.md) and [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) —
  authoring the capability floor and the refusal vocabulary
- [`Dockerfile`](https://github.com/anthony-chaudhary/fak/blob/main/Dockerfile) and [`install.sh`](https://github.com/anthony-chaudhary/fak/blob/main/install.sh) — the
  build and install sources this guide describes

## All-in-one runtime (`fak up`)

`fak up` is the product entry point for the existing unified serve lifecycle. It
starts one process that owns the gateway, governed agent-session route, policy
adjudication, quarantine, hash-chained audit, and metrics:

```sh
fak up --engine mock --native
curl http://127.0.0.1:8080/readyz
curl -N -H "Content-Type: application/json" \
  -d '{"goal":"book the requested task"}' \
  http://127.0.0.1:8080/v1/fak/agent/sessions
```

`/readyz` reuses the gateway's existing startup and health gates. It returns
HTTP 503 until the listener has reached `MarkReady`, and it also reflects the
same model warmup, degenerate-decode, recent served-failure, and deep provider
checks as `/healthz`. `fak up` accepts the complete `fak serve` flag surface;
it does not add a second agent loop or a process supervisor.

One artifact does not remove host prerequisites. Native accelerator execution
still requires compatible hardware, OS/vendor drivers, supported ISA and enough
RAM; local execution also requires the selected model files. Remote engines and
future hosted fak placement require explicit endpoints, network reachability,
and credentials. Unsupported local compute must fail as an explicit placement
or capability result. The runtime never silently changes fak-native or
performance work to llama.cpp, Ollama, or a cloud provider.

Examples keep placement visible:

```sh
# CPU-safe deterministic/reference placement
fak up --engine mock --native

# Optional in-kernel model; load is part of measured startup readiness
fak up --gguf model.gguf --native

# Hardened network deployment; external placement remains explicit in serve flags
fak up --addr 0.0.0.0:8080 --policy floor.json --require-key-env KEY
```

Restart-on-crash, a `fak down` command, and durable in-flight checkpoint drain
are separate lifecycle features; `fak up` does not claim them.
