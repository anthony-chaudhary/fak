# GCP Tier-2 always-on dogfood control VM (issue #732)

Audience: operators bringing up the always-on Tier-2 dogfood lane after reading
[always-on-dogfood-server.md](always-on-dogfood-server.md). Prerequisite: a GCP
project with `gcloud` access for apply mode. You will be able to review the VM
plan safely, apply it from an authenticated host, and verify the systemd units.

[always-on-dogfood-server.md](always-on-dogfood-server.md) describes Tier 2: a
cheap `e2-small` running the guarded fleet plus a shared `fak serve` gateway
24/7, reachable over Tailscale. This is the script that **stands it up**:
`scripts/gcp-dogfood-control-vm.sh`.

A VM never sleeps, so Tier 2 is the steady-state overflow lane next to the Mac (Tier 1).
The units are **systemd** — the Linux analogue of the Tier-1 launchd plists:

| Tier 1 (Mac, launchd) | Tier 2 (GCP VM, systemd) | role |
|---|---|---|
| `tools/com.fak.serve-gateway.plist` | `fak-serve-gateway.service` | 24/7 `fak serve` anthropic-passthrough gateway on `127.0.0.1:8080` |
| `tools/com.fak.dogfood-fleet.plist` | `fak-dogfood-fleet.service` + `.timer` | one preflight-gated guarded dispatch tick every 30 min (journals to `.dispatch-runs/guard-audit/`) |

Both systemd units are embedded in the VM startup script the deploy script renders, so the
VM converges on first boot with no further SSH.

## Plan it (no creds needed)

The script is **plan-by-default** — it prints the exact `gcloud` command and the full VM
startup script and exits, so the whole deploy is reviewable without a GCP project:

```bash
./scripts/gcp-dogfood-control-vm.sh            # prints gcloud + the cloud-init startup script
```

Knobs (env): `GCP_ZONE` (default `us-central1-a`), `GCP_MACHINE` (default `e2-small`),
`VM_NAME` (default `fak-dogfood-control`), `FAK_REPO_URL`, optional `TAILSCALE_AUTHKEY`.

## Apply it (deferred — needs GCP creds)

```bash
GCP_PROJECT=<your-project> ./scripts/gcp-dogfood-control-vm.sh --apply
```

`--apply` requires an authenticated `gcloud` and a `GCP_PROJECT`. **The actual apply is
deferred:** no GCP credentials / project are available from the implementing host, so the
VM is not created here. The script + units land now; running `--apply` on an authenticated
host is the remaining step.

After apply, verify the same way as Tier 1:

```bash
gcloud compute ssh fak-dogfood-control --zone us-central1-a --command '
  systemctl is-active fak-serve-gateway.service
  systemctl list-timers fak-dogfood-fleet.timer
  ls /opt/fak/.dispatch-runs/guard-audit/*.jsonl
'
```

A journal under `.dispatch-runs/guard-audit/` on the VM is the same `audit_journal_evidence`
witness `tools/dogfood_coverage.py` counts (issue #731) — so a live Tier-2 VM is a second
way to flip coverage to grade A, independent of the Mac node (#729).

## GPU burst (separate lane)

Tier 2 steady state is CPU-only (the gateway does no model compute — the upstream does). To
exercise fak's own in-kernel decode under load, **burst** to a GPU VM via the accelerator
ladder in `tools/gcp_accel.py` (cheapest `n1-t4` first to de-risk the plumbing). That is a
separate, on-demand lane, not part of this always-on control VM.

## Refs

- `scripts/gcp-dogfood-control-vm.sh` — the plan/apply bring-up (this issue)
- `docs/fak/always-on-dogfood-server.md` — the Mac/GCP always-on tiers design
- `docs/fak/node-macos-a-activation.md` — the Tier-1 launchd analogue (#729)
- `tools/gcp_accel.py` — the GPU-burst accelerator ladder
