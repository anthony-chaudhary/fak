---
title: "Team node deployment: one fak node for many engineers"
description: "Stand up one always-on fak node that a team of engineers shares over a protected network: per-person gateway keys, per-person audit attribution, the ops plane, and dashboards."
---

# Team node deployment — one node, many engineers

> **Audience.** An operator standing up ONE always-on node for a TEAM of engineers,
> each running their own agents against it. By the end: the gateway runs with
> per-person keys, every session is attributed to its person, the repo-maintenance
> ops plane ticks on schedule, and dashboards show what is happening.

This page joins pieces that already exist but no single doc wires together: the
`fak node` lifecycle ([node-setup.md](node-setup.md)), the multi-tenant keyset
(`--key-principal`, [server-config.md](server-config.md#authentication) ·
[serve-config.md](../serve-config.md)), the ops plane
([verb-surface.md](../generated/verb-surface.md)), and the observability stack
([observability.md](observability.md)). Everything below is shipped; anything not
yet is a labeled known gap (§9).

## 1. Topology

N engineers, one node, a protected network:

```
engineer A ─┐
engineer B ─┼──► Tailscale / VPN ──► TLS edge ──► THE NODE
engineer N ─┘                        ├─ fak serve gateway :8080 (per-person keys + floor)
                                     ├─ fak ops daemon (repo maintenance sweeps)
                                     └─ Prometheus :9091 → Grafana :3000 (+ Alertmanager :9093)
```

Each engineer's bearer authenticates AND attributes their sessions (§2–§3); the
ops plane maintains the repo the node hosts (§4); Prometheus scrapes `/metrics`
and Grafana serves the public dashboards (§5).

## 2. Install the gateway — two auth tiers

### Tier A — quickstart: one shared bearer

The one-command path from [node-setup.md](node-setup.md):

```bash
fak node install --remote \
  --base-url http://127.0.0.1:8131/v1 --model your-model-id
# installs the service (systemd fak-serve-gateway / launchd com.fak.serve-gateway /
# schtasks FakServeGateway), binds 0.0.0.0:8080, prints ONE generated bearer key
```

Fine for a trusted small group sharing one key. For a team, use tier B so every
person is individually attributable.

### Tier B — team tier: per-person keys (`--key-principal`)

Run `fak serve` directly under the hardened unit
[`deploy/systemd/fak.service`](../deploy/systemd/fak.service) with the root-only env
file [`deploy/systemd/fak.env.example`](../deploy/systemd/fak.env.example):

```bash
sudo cp deploy/systemd/fak.service /etc/systemd/system/fak.service
sudo cp deploy/systemd/fak.env.example /etc/fak/fak.env   # chown root:root, chmod 600
```

Extend the unit's `ExecStart` with one `--key-principal` per engineer (repeatable;
each value names the ENV VAR holding that person's key, never the key itself):

```ini
ExecStart=/usr/local/bin/fak serve --addr 0.0.0.0:8080 \
  --provider openai --base-url https://api.openai.com/v1 \
  --model gpt-4o --api-key-env OPENAI_API_KEY \
  --policy /etc/fak/policy.json \
  --require-key-env FAK_GATEWAY_KEY \
  --key-principal alice=FAK_KEY_ALICE \
  --key-principal bob=FAK_KEY_BOB
```

…and add the per-person secrets to `/etc/fak/fak.env`:

```bash
FAK_KEY_ALICE=<output of openssl rand -hex 32>
FAK_KEY_BOB=<output of openssl rand -hex 32>
```

Then `sudo systemctl daemon-reload && sudo systemctl enable --now fak`.

Properties of this setup:

- **Authenticates AND attributes.** A matching inbound key (`x-api-key` or
  `Authorization: Bearer`) stamps that principal onto the access log and
  `/v1/fak/events` (§3); keys are hashed to SHA-256 at construction, plaintext
  dropped.
- **Additive to `--require-key-env`.** The shared bearer still authenticates the
  anonymous single-tenant caller; no `--key-principal` specs leave that path
  byte-for-byte unchanged.
- **Refuses to boot, loudly,** on a malformed spec, an unset/empty env var, or two
  principals sharing one key (`KEY_PRINCIPAL_UNRESOLVED`, exit 2) — a
  half-resolved keyset is a tenant-isolation hole, not a cosmetic bug.
- **No unauthenticated off-host bind.** `0.0.0.0` with neither token door is
  refused at startup.

## 3. Per-person audit & attribution

With `--key-principal` armed, the principal is authoritative-from-context: it lands
in the JSON access log and on `/v1/fak/events` (as `X-Fak-Principal`), joinable by
`X-Trace-Id` across metrics, log, and events ([observability.md](observability.md)).
Without a matched key, attribution falls through to the caller-supplied header —
exactly why a half-resolved keyset refuses to boot.

Tamper-evident journal: set `FAK_AUDIT_JOURNAL` (the committed env example points it
at `/var/lib/fak/audit.jsonl`) and verify the chain:

```bash
fak audit verify /var/lib/fak/audit.jsonl
```

> **Honest gap — per-person cost and rate limits.** The usage ledger
> (`internal/gatewayusageledger/ledger.go`) has **no principal field yet**, so
> per-person *cost* attribution today is access-log/events only. Per-principal rate
> limits are **not yet enforced** (#5379): one noisy tenant can eat the shared
> provider budget. The global caps `FAK_RATELIMIT_MAX_CALLS` /
> `FAK_RATELIMIT_MAX_COST` (bucketed by `trace`|`tool`|`global`) bound the whole
> gateway, not one person.

## 4. Install the ops plane

`fak ops` maintains the repo the node hosts: dead locks, cold worktrees, runaway
processes, storage reclaim. `fak ops daemon` is a foreground ticker loop
(`--interval`, default 60s; `--once` for a single tick) — **there is no service
installer verb yet**, so hand-write a minimal unit:

```ini
# /etc/systemd/system/fak-ops.service — HAND-WRITTEN: no installer verb yet.
[Unit]
Description=fak ops daemon — periodic repo maintenance sweeps
After=network-online.target

[Service]
# The engine discovers the repo root from WorkingDirectory; the event ledger
# lands at <repo>/.fak/ops-events.jsonl, so the user must be able to write it.
WorkingDirectory=/opt/fak-repo
User=fakops
ExecStart=/usr/local/bin/fak ops daemon --interval 5m
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Human-facing verbs (all repo-scoped, same `WorkingDirectory`):

```bash
fak ops status            # disk headroom + recent event count (healthy/warning/refuse)
fak ops sweep --dry-run   # preview one maintenance pass without mutations
fak ops log --since 24h   # the ledger tail
```

Every mutation is recorded to the repo-scoped ledger `<repo>/.fak/ops-events.jsonl`.
Note: `fak ops schedule` enrolls OpenCode orchestrators, NOT the daemon. The
ops-plane exporters and "Fak Ops" dashboard (`fak-ops-dashboard` :9101, `fak-sync`
ops fold :9094) are companion-repo (fak-private) owned today; public teams get the
gateway / fleet / cachevalue dashboards (§5).

## 5. Dashboards

The repo ships a ready-to-run stack ([`tools/grafana/`](../tools/grafana/README.md)):

```bash
tools/grafana/up.sh       # Prometheus :9091 + Grafana :3000 (anonymous Viewer) + Alertmanager :9093
tools/grafana/down.sh     # tear down (--purge drops volumes)
```

Point the scrape job at the node: edit the `fak_gateway` target in `prometheus.yml`
to the gateway address. Scrape security when `--require-key-env` is on: `/metrics`
then requires a bearer — add Prometheus auth to the job (e.g. `bearer_token_file`)
or front it with a loopback-only scrape proxy ([observability.md §5](observability.md)).
The gateway config also carries a read-scoped `ReadBearer` accepted ONLY on
`/healthz`, `/debug/vars`, `/metrics` (`internal/gateway/config.go`) — `fak guard`
mints one for its own session gateway; a standalone `fak serve` does not expose it
as a flag yet.

Provisioned public dashboards: `fak-gateway-observability`, `fak-startup-load`,
`fak-dogfood-slow-requests`, plus the fleet and cachevalue views; alert rules ship
in `prometheus-alerts.yml` (gateway-down on `fak_gateway_up == 0`).

## 6. Client onboarding (per engineer)

On each engineer's machine, once they hold their personal key:

```bash
fak node use <host>:8080 --key 'THEIR_PERSONAL_KEY'   # saves router + key, probes /healthz
fak node run -- claude                                # agent pointed at the node
```

`use` writes `~/.config/fak/node.json`; bare `fak agent` / `fak chat` use the saved
router. Key rotation:

- **Tier A (shared key):** `fak node install --rotate-key` mints a fresh bearer —
  the installer prints a loud note that **every client must re-run `fak node use`
  with the new key**.
- **Tier B (per-person):** change that person's env-var value in
  `/etc/fak/fak.env`, restart the service, and have that engineer re-run
  `fak node use` with the new key; everyone else keeps working.

## 7. Network posture

- **Protected network only.** Keep the node behind Tailscale/VPN; HTTP on an
  unprotected LAN carries keys in cleartext ([node-setup.md](node-setup.md)).
- **TLS at the edge.** fak speaks plain HTTP — put a TLS-terminating reverse proxy
  in front of the `0.0.0.0` bind ([deployment-guide.md](deployment-guide.md)).
- **Never expose without a required key.** Off-host binds without a token door are
  refused at startup; keep `--allow-lan` off and restrict the `0.0.0.0` listener to
  trusted clients with the host firewall ([node-setup.md](node-setup.md)).
- **Scrape surfaces stay gated.** With auth on, `/metrics` and `/debug/vars`
  require the bearer; confirm `gateway.auth_required` in `/debug/vars`
  ([observability.md](observability.md)).

## 8. Go-live checklist

Merged from the [production readiness checklist](deployment-guide.md#production-readiness-checklist)
and the [security deployment checklist](security.md#deployment-checklist) — clear
both before pointing a team at the node:

- [ ] Auth on: `--require-key-env` with strong per-person values in a root-only env
      file; confirm the vars are non-empty in the serving process.
- [ ] Capability floor pinned: explicit `--policy` validated with
      `fak policy --check`; irreversible tools absent from the allow-list.
- [ ] Bind intentional: `0.0.0.0` only behind TLS + firewall; `/metrics` and
      `/debug/vars` auth-gated.
- [ ] Audit journal enabled (`FAK_AUDIT_JOURNAL`) and verified with
      `fak audit verify`.
- [ ] Health + metrics wired: `/healthz` probed, `/metrics` scraped, gateway-down
      alert live.
- [ ] Run as non-root, version pinned (not `latest`), timeouts sized to the
      backend; every engineer onboarded via `fak node use` with their own key,
      rotation procedure (§6) known.

## 9. Known gaps & follow-ups

- **No principal field in the usage ledger** — per-person cost attribution is
  access-log/events only today (`internal/gatewayusageledger/ledger.go`).
- **Per-principal rate limits not enforced** (#5379) — one noisy tenant can eat the
  shared provider budget; only global caps exist today.
- **No `fak ops` service installer** — hand-written unit needed (§4); SOURCE-ONLY
  row in [verb-surface.md](../generated/verb-surface.md).
- **Ops-plane dashboards/exporters are companion-repo owned** — public teams get
  gateway/fleet/cachevalue only.
- **`ReadBearer` not exposed on `fak serve`** — read-scoped scraping uses
  `bearer_token_file` or a loopback proxy until a flag lands.

## See also

- [node-setup.md](node-setup.md) — the `fak node` lifecycle (install/use/run/status/forget)
- [server-config.md](server-config.md) — every `fak serve` flag and env var
- [observability.md](observability.md) — metrics, access log, trace correlation
- [always-on-dogfood-server.md](always-on-dogfood-server.md) — the shared-gateway design (one shared key)
- [deployment-guide.md](deployment-guide.md) — production deployment
- [security.md](security.md) — threat model and deployment checklist
