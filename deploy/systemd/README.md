---
title: "fak as a systemd service — committed unit"
description: "A committed, hardened systemd unit for fak serve: DynamicUser, StateDirectory, ProtectSystem=strict, and an env-file template that keeps secrets out of the unit."
---

# fak as a systemd service

The hardened unit the [deployment guide's systemd
section](../../docs/fak/deployment-guide.md#run-as-a-service-systemd) describes,
materialized as files you can copy directly instead of retyping out of a fenced
block — the same rationale as [`deploy/k8s/`](../k8s). The
[deployment guide](../../docs/fak/deployment-guide.md) is the **source of truth**
for every flag, env var, and default; read its production-readiness checklist
before exposing the gateway beyond loopback.

## What's here

| File | What it is |
|---|---|
| [`fak.service`](fak.service) | The unit: `fak serve` behind `DynamicUser`, `StateDirectory=fak`, `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`. |
| [`fak.env.example`](fak.env.example) | Root-only environment-file template (`EnvironmentFile=/etc/fak/fak.env`) — every secret lives here, never in the unit. |

## Install

```bash
sudo cp deploy/systemd/fak.service /etc/systemd/system/fak.service
sudo install -d -m 700 /etc/fak
sudo cp deploy/systemd/fak.env.example /etc/fak/fak.env
sudo chown root:root /etc/fak/fak.env && sudo chmod 600 /etc/fak/fak.env
# edit /etc/fak/fak.env — at minimum FAK_GATEWAY_KEY (openssl rand -hex 32)

fak policy --check /etc/fak/policy.json   # validate before enabling
sudo systemctl daemon-reload
sudo systemctl enable --now fak
curl -s http://127.0.0.1:8080/healthz
journalctl -u fak -f                     # watch for the no-auth WARNING
```

## The one rule that bites first

A network-facing gateway **must** set `--require-key-env` (done in the committed
unit) and pin a policy floor (`--policy /etc/fak/policy.json`). `/healthz` is the
only unauthenticated route. **TLS belongs at the edge** — terminate it at a front
proxy or load balancer; `fak serve` speaks plain HTTP.

## Notes

- **`DynamicUser=yes` + `StateDirectory=fak`**: the service runs as an unprivileged
  dynamic user; `/var/lib/fak` is created for it, which is where the unit's env file
  points `FAK_AUDIT_JOURNAL`.
- **`ProtectSystem=strict`** makes the filesystem read-only for the service except
  for `/var/lib/fak` (and `/run`, `/tmp` via `PrivateTmp`). The policy file at
  `/etc/fak/policy.json` is read-only access, which is allowed.
- **Edit the `ExecStart` line for your upstream**: `--provider`/`--base-url`/`--model`
  name the upstream model endpoint; for a local Ollama or an in-kernel `--gguf`
  model, see the guide's "Bare-metal with a real local model" section.
- macOS equivalent: [`tools/com.fak.serve-gateway.plist`](../../tools/com.fak.serve-gateway.plist)
  (launchd). Container/Kubernetes: [`deploy/k8s`](../k8s), [`deploy/helm`](../helm).
