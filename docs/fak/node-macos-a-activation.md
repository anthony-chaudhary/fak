---
title: "fak node-macos-a dogfood fleet activation"
description: "Runbook to activate the always-on guarded dogfood fleet on node-macos-a via launchd; the on-device launchctl load is deferred."
---

# Activating the always-on guarded dogfood fleet on node-macos-a (issue #729)

Audience: operators activating the Tier-1 Mac dogfood node from the always-on
dogfood plan. Prerequisite: read
[always-on-dogfood-server.md](always-on-dogfood-server.md) and have shell access
to `node-macos-a`. You will be able to install the launchd units, keep the node
awake, and verify that guarded audit journals are real.

This is the runbook that turns node-macos-a into a 24/7 **guarded** dogfood node — the
one that finally flips `audit_journal_evidence` from a configured wire into an exercised
one, taking `tools/dogfood_coverage.py` from grade **B** to **A** (issue #731).

## What "activated" means

Two launchd units run together on the node:

| Unit | Role | Witness it produces |
|---|---|---|
| `tools/com.fak.serve-gateway.plist` | keeps the shared `fak serve` gateway alive 24/7 on `127.0.0.1:8080` | `always_on_serve_plist` (configured) |
| `tools/com.fak.dogfood-fleet.plist` | fires ONE guarded dispatch tick every 30 min (`issue_dispatch.py --live`) | **`audit_journal_evidence`** — a guarded worker journals kernel verdicts under `.dispatch-runs/guard-audit/` |

The gateway proves the wire is **built**; the dogfood-fleet tick proves it is **exercised**.
The coverage scorecard only counts the second as evidence, so both are required for grade A.

## Quick setup — one command (preferred)

```bash
cd <repo>

# Set the upstream key, then run the installer — it builds fak, fills all
# template placeholders, loads both launchd units, and starts caffeinate.
# The gateway plist now wraps fak serve under caffeinate so sleep prevention
# is part of the launchd service itself (no separate keep-awake step).
ANTHROPIC_API_KEY="sk-ant-..." ./tools/install-mac-node.sh

# Check status any time:
./tools/install-mac-node.sh --status

# For off-host access (e.g. connecting a Windows/Mac client over Tailscale):
ANTHROPIC_API_KEY="sk-ant-..." ./tools/install-mac-node.sh --bind-all
# The script prints FAK_GATEWAY_KEY + exact env lines to paste on the client.
```

The installer runs `tools/mac_keep_awake.sh start` as a belt-and-suspenders for
the dispatch units; the gateway itself wraps `fak serve` under `caffeinate -is`
so a reboot also picks it up without any extra step.

## Manual steps (fallback — single unit at a time)

```bash
cd <repo>

# 1. Build fak and install both launchd units (fill the template placeholders).
go build -o tools/.bin/fak ./cmd/fak

sed -e "s#__FAK__#$(pwd)/tools/.bin/fak#" \
    -e "s#__REPO__#$(pwd)#" \
    -e "s#__ADDR__#127.0.0.1:8080#" \
    tools/com.fak.serve-gateway.plist > ~/Library/LaunchAgents/com.fak.serve-gateway.plist
sed -e "s#__PYTHON__#$(command -v python3)#" -e "s#__REPO__#$(pwd)#" \
    tools/com.fak.dogfood-fleet.plist  > ~/Library/LaunchAgents/com.fak.dogfood-fleet.plist

# 2. Set the upstream credential in the login environment (NOT in the template).
launchctl setenv ANTHROPIC_API_KEY "sk-ant-..."

# 3. Load both units (RunAtLoad fires the first tick immediately).
#    caffeinate -is is now baked into the gateway plist — no separate keep-awake needed.
launchctl load -w ~/Library/LaunchAgents/com.fak.serve-gateway.plist
launchctl load -w ~/Library/LaunchAgents/com.fak.dogfood-fleet.plist
```

## Verify the flip

```bash
# A guarded worker has journaled at least one kernel decision:
ls .dispatch-runs/guard-audit/*.jsonl
wc -l .dispatch-runs/guard-audit/*.jsonl

# audit_journal_evidence is now met; coverage should read grade A:
python tools/dogfood_coverage.py
#   dogfood-coverage: 100.0% (9/9 KPIs)  grade A  dogfood_debt 0  audit_rows N>0
python tools/dogfood_coverage.py --check    # exit 0 (the #731 gate)
```

The same coverage payload is emitted on a cadence by `.github/workflows/dogfood-coverage.yml`
(issue #731) and tailed live from `tools/_watchdog/launchd_dogfood_fleet.log`.

## Deferred: the on-device activation

The **config + units + runbook land here.** The actual `launchctl load` on node-macos-a is
**deferred** — the Mac is asleep / unreachable from the implementing host, so the units
can't be installed and the first guarded tick can't be fired from here. Activation is the
four steps above, run on the node once it is reachable; the verify block then confirms
`audit_journal_evidence` flips and coverage holds at A.

Until then, `audit_journal_evidence` stays soft-unmet (coverage grade B), which is the
honest state — no fabricated journal rows. The #731 gate (`--check`, HARD debt) stays green
throughout, because grade A depends only on this soft KPI.

## Refs

- `tools/com.fak.dogfood-fleet.plist` — the guarded-tick cadence unit (this issue)
- `tools/com.fak.serve-gateway.plist` — the 24/7 gateway daemon
- `tools/issue_dispatch.py` — the preflight-gated guarded dispatch tick
- `tools/dogfood_coverage.py` — counts `.dispatch-runs/guard-audit/*.jsonl` as `audit_journal_evidence`
- `docs/fak/always-on-dogfood-server.md` — the Mac/GCP always-on tiers design
