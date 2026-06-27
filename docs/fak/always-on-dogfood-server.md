---
title: "Always-On Dogfood Server: 3x the Kernel on the Real Dev Loop"
description: "How to run the fak kernel in front of the real dev workflow 24/7 — the guarded dispatch fleet plus a shared fak serve gateway — across a laptop, an always-on Mac, and GCP. Setup, measurement, and the kill switch."
---

# Always-On Dogfood Server

Read this if you operate fak's own agent fleet or want interactive Claude Code
sessions to cross the same kernel boundary as unattended dispatch workers. You
will be able to pick the right always-on tier, run the guarded worker loop,
expose a shared authenticated gateway, measure coverage from audit journals, and
use the kill switches when the dev loop must bypass dogfood. For the basic server
setup first, start with [server-quickstart.md](server-quickstart.md); for the
production knobs this page relies on, see [server-config.md](server-config.md)
and [observability.md](observability.md).

## 1. Thesis: dogfood the kernel on the REAL dev loop

"Dogfooding" only counts when our own daily dev work crosses the kernel boundary —
not when a demo does. fak's whole claim is that the kernel (`fak serve` / `fak guard`)
belongs in front of *every* tool call an agent proposes: deny the dangerous ones by
structure, repair malformed args, quarantine poisoned results, and write every verdict
to a durable, tamper-evident record. The honest test of that claim is to put it in
front of the highest-volume agent work we actually run — the dispatch fleet and our
own Claude Code sessions — and leave it there.

The highest-volume dev work on a fleet node is a dispatch worker: a full agentic
`claude -p` session that runs unattended. Until recently every one of those workers
talked **straight to the provider API** — the kernel adjudicated **none** of it. That
is the inverse of dogfooding.

A just-shipped change closes that gap. `tools/dispatch_worker.py` now fronts every
worker with `fak guard` **by default** (`guarded_launch_command`, gated by
`FLEET_DOGFOOD_GUARD`), and `tools/issue_dispatch.py` routes its detached spawn through
the same path. So:

| | Fleet workers through the kernel |
|---|---|
| **Before** | 0% — workers called the provider API directly |
| **After** | 100% of claude workers, **default-on**, fail-open when `fak` is not built |

That is the **first** of the three multipliers. The other two are *time* (run the
guarded loop 24/7, not only when a laptop happens to be awake) and *surface* (put a
shared `fak serve` gateway in front of hand-driven Claude Code sessions too, so even
interactive coding is kernel-adjudicated). This doc is how you get all three.

The interactive front door is the one-command, productized form of the same boundary
(`cmd/fak/guard.go`): `fak guard -- claude` starts the same gateway `fak serve` runs,
points the child agent's base URL at it through a **child-only** env var (never your
shell, never `settings.json`), defaults the upstream to the real Anthropic API in
passthrough mode, uses your Claude Pro/Max **subscription** OAuth token by default when
no `ANTHROPIC_API_KEY` is set, and turns a **durable, hash-chained decision journal on
by default** that you can replay with `fak audit verify`.

```
 ┌─────────────┐  POST /v1/messages  ┌────────────────────────┐  /v1/messages  ┌──────────────────┐
 │ claude (-p) │ ─────────────────▶  │  fak guard / fak serve  │ ─────────────▶ │ api.anthropic.com │
 │  the worker │ ◀──── SSE stream ─  │  adjudicates every tool │ ◀──────────── │   (real Claude)   │
 └─────────────┘                      └────────────────────────┘                └──────────────────┘
   ANTHROPIC_BASE_URL set on the CHILD only      every tool call crosses the floor; every verdict journaled
```

---

## 2. The three always-on tiers

You can dogfood at three levels of "always-on". Each tier is additive — Tier 1 and
Tier 2 just keep the same guarded loop running longer and reach more sessions.

| | Tier 0 — Laptop (Windows) | Tier 1 — Always-on Mac | Tier 2 — GCP always-on |
|---|---|---|---|
| **What runs** | scheduled-task fleet | guarded fleet 24/7 + shared `fak serve` gateway | guarded fleet 24/7 + shared `fak serve` gateway |
| **Uptime** | intermittent (whenever the laptop is on) | 24/7 (launchd `KeepAlive` + `caffeinate`) | 24/7 (VM never sleeps) |
| **Cost** | $0 | $0 (hardware you own) | ~ a few $/month for an `e2-small`; GPU only on burst |
| **Local-model in-kernel path** | CPU-slow (proof only) | CPU/Metal | burst to a GPU VM (see `tools/gcp_accel.py`) |
| **Reachable by other machines** | no | yes, over Tailscale | yes, over Tailscale / private IP |
| **Role** | the status quo | the recommended dev server | overflow + GPU bursts |

The kernel boundary is **identical** on every tier — it is the same `fak guard` /
`fak serve` gateway. The tiers differ only in *how long it stays up* and *who can reach
it*.

### Tier 0 — Laptop (Windows): the status quo

The fleet already runs on the laptop on a Windows Scheduled Task that ticks a watchdog
every 5 minutes; the watchdog respawns the `dos loop --enact` dispatch supervisor when
one is not alive, and each spawned worker is now guarded by default. This is real
dogfooding, but only while the laptop is awake and the task is enabled — so coverage is
*intermittent*.

Nothing new is required here; the guarded path is on by default. Confirm it:

```powershell
# Build the in-tree fak binary the guard path resolves (tools/.bin/fak.exe)
.\scripts\dogfood-claude.ps1 --install

# See exactly what one worker would launch — note `fak ... guard ... -- claude`
python tools\dispatch_worker.py --lane demo --dry-run

# Plan one safe, switcher-routed, bounded dispatch tick (dry-run)
python tools\issue_dispatch.py
```

### Tier 1 — Always-on Mac (M-series mini): the recommended dev server

A small Apple-silicon Mac that stays on is the best always-on dogfood host: it is quiet,
cheap to run, and Metal makes even the local-model in-kernel path usable. It does two
jobs.

**Job A — run the guarded fleet 24/7.** One command wires all three launchd units
(serve-gateway, dogfood-fleet, dispatch-supervisor), builds the binary, and starts
the sleep guard. The `fak serve` gateway plist now wraps `fak serve` under `caffeinate -is`
so the machine cannot idle-sleep while the gateway is running — no separate
keep-awake step needed:

```bash
# ONE COMMAND — builds fak, fills all plist templates, loads units, starts caffeinate:
ANTHROPIC_API_KEY="sk-ant-..." ./tools/install-mac-node.sh

# For off-host access from another machine (Windows or Mac over Tailscale):
ANTHROPIC_API_KEY="sk-ant-..." ./tools/install-mac-node.sh --bind-all
# The script prints FAK_GATEWAY_KEY + exact env lines to paste on each client.

# Check status or uninstall:
./tools/install-mac-node.sh --status
./tools/install-mac-node.sh --uninstall
```

To also put `fak` on PATH system-wide (so `fak guard -- claude` resolves anywhere),
run `scripts/dogfood-claude.sh --install` once after the node installer.
Full runbook: [`docs/fak/node-macos-a-activation.md`](node-macos-a-activation.md).

**Stay awake.** The `caffeinate -is` wrapper in `tools/com.fak.serve-gateway.plist`
holds the idle-sleep and system-sleep assertions while `fak serve` runs. Together with
launchd `KeepAlive=true`: launchd keeps the process alive, caffeinate keeps the *machine*
alive — no more 25-minute flap. If only the dispatch supervisor is running (no gateway),
`tools/mac_keep_awake.sh start` provides the same assertion separately.

**Job B — host a shared `fak serve` gateway for hand-driven sessions.** The fleet covers
*unattended* workers. To cover *interactive* Claude Code too — yours and anyone else on
the network — run one shared gateway on the Mac that other machines point
`ANTHROPIC_BASE_URL` at. Then a normal `claude` on a laptop is kernel-adjudicated without
running its own gateway.

```bash
# On the Mac: a shared, authenticated gateway in front of the real Anthropic API.
# Bind beyond loopback ONLY with a required key — an unauthenticated off-host kernel
# gateway is an open door (fak guard and fak serve both warn about this).
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"
export ANTHROPIC_API_KEY="sk-ant-..."        # or use the subscription-OAuth default
fak serve --addr 0.0.0.0:8080 \
  --provider anthropic --base-url https://api.anthropic.com \
  --policy examples/dogfood-claude-policy.json \
  --require-key-env FAK_GATEWAY_KEY
```

Reach it over **Tailscale** (the Mac and your laptop on one tailnet — no public exposure).
On any other machine:

```bash
export ANTHROPIC_BASE_URL="http://<mac-tailscale-ip>:8080"
export ANTHROPIC_API_KEY="$FAK_GATEWAY_KEY"   # the gateway's bearer; auth is over Tailscale
claude                                          # your normal Claude Code, now kernel-adjudicated
```

For a *single* laptop session you do not need the shared gateway at all —
`fak guard -- claude` runs its own in-process gateway on a private loopback port and
needs no key. The shared gateway is the lever for covering *many* machines / sessions
from one always-on host.

### Tier 2 — GCP always-on: overflow + GPU burst

When the Mac is busy or you want a second always-on lane, a small GCP VM runs the same
two jobs. Use a cheap, always-on instance for the steady state, and **burst** to a GPU
VM only when you want to exercise fak's own in-kernel decode (`fak serve --gguf`, the
pure-Go forward) under real Claude Code load.

**Steady state — a tiny always-on VM.** An `e2-small` is enough to run the guarded
fleet and a shared `fak serve` anthropic-passthrough gateway (the gateway does no model
compute itself — the upstream does). Install the in-tree binary, arm the same watchdog
cron, and run the same shared gateway as Tier 1 Job B. Reach it over Tailscale or the
VPC's private IP; never bind `0.0.0.0` without `--require-key-env`.

**Burst — a GPU VM for the local-model in-kernel path.** The CPU-only in-kernel forward
is too slow for a full interactive Claude Code turn (a real turn sends ~5–6K tokens of
system prompt and tool schemas; see the honest caveat in `DOGFOOD-CLAUDE.md`). For that
path you want a GPU. `tools/gcp_accel.py` is the registry of GCP accelerator machine
types fak can run on — a Blackwell-first fallback ladder (`a4-b200` → `a4x-gb200` →
`a3-ultra-h200` → `a3-high-h100` → `g2-l4` → `n1-t4`) with the cheapest tier
(`n1-t4`, ~$0.55/hr) reserved for de-risking the plumbing before spending on a big node.

```bash
# Inspect the accelerator ladder (pure data; no gcloud / network call)
python tools/gcp_accel.py
```

Provision the cheapest tier first to prove the loop end-to-end, then burst up. Run
`fak serve --gguf <weights>` on the GPU node and point the guarded fleet (or a Tier-1
gateway) at it; tear the GPU VM down when the burst is done so it is not always-on cost.

---

## 3. How to measure the 3x

You measure dogfooding with two things: a **coverage scorecard** and the **audit
journals** the guarded workers leave behind. Configuration is not evidence — a flag can
say "on" while nothing ran — so both checks cross-check reality on the live host.

**The scorecard.** `tools/dogfood_coverage.py` imports `dispatch_worker` and calls the
live `guarded_launch_command` on *this* host, so the score reflects what would actually
launch — not what a config claims. It folds its KPIs into one `coverage` percent, a
`dogfood_debt` integer (count of unmet HARD affordances), an A–F grade, and a
control-pane JSON payload.

```bash
python tools/dogfood_coverage.py            # human report
python tools/dogfood_coverage.py --json      # control-pane payload
python tools/dogfood_coverage.py --check     # exit 1 if any HARD KPI is unmet
```

The HARD KPIs are the ones that must hold for the fleet to be kernel-adjudicated at
all:

- `fleet_leaf_guarded` — the leaf launcher really fronts a claude worker with `fak guard`
  on this host (a behavior check, not a grep).
- `bin_resolvable` — a `fak` binary resolves, so the fail-open path is not silently
  dropping coverage to 0%.
- `guard_default_on` — `FLEET_DOGFOOD_GUARD` is not disabled in the live environment.
- `issue_dispatch_wired` — the scheduled-task lane routes its spawn through the guard path.
- `guard_verb_present` — `fak guard` exists as the one-command front door.

**The journals are the witness.** Every guarded worker writes its verdicts to a durable,
hash-chained JSONL journal. The fleet uses a **per-session** journal under the gitignored
`.dispatch-runs/guard-audit/`, named `<lane>-<backend>-<pid>-<id>.jsonl` — keyed on the
lane and backend (for separability and globbing) **plus a per-process token**. That
per-session key is deliberate: the hash-chained journal has no inter-process lock, so two
concurrent same-lane workers sharing one file would braid two independent chains into a
forked, unverifiable journal. A per-session file lets each `fak guard` own its own valid
chain; the interactive `fak guard` default writes one under your user config dir.
`dogfood_coverage.py` counts the decision rows across those journals (`audit_rows` in the
payload) — that is the proof the wire was *exercised*, not merely wired. Verify any one
chain is intact (glob the lane prefix to find them):

```bash
fak audit verify .dispatch-runs/guard-audit/<lane>-claude-<pid>-<id>.jsonl
```

Run `dogfood_coverage.py` on a `/loop` cadence to keep the number from rotting; watch
`audit_rows` climb as the always-on fleet works, and watch `coverage` hit and hold A.

---

## 4. The kill switch and safety

Dogfood-by-default never means dogfood-no-matter-what. There are three release valves,
all already in the code.

**Kill switch — `FLEET_DOGFOOD_GUARD=0`.** Set this on a node and its workers launch
**unguarded** (straight to the provider), no code change, no `dos.toml` edit. Any of
`{0, off, false, no, disable, disabled}` (and empty) turns it off; unset means on.

```bash
FLEET_DOGFOOD_GUARD=0 python tools/issue_dispatch.py --live   # this node: workers unguarded
```

**Fail-open when `fak` is not built.** `resolve_fak_bin` looks for `$FAK_BIN`, then the
in-tree `tools/.bin/fak[.exe]` the dogfood launcher builds, then `fak` on PATH. If none
resolves it returns nothing and the worker launches **unwrapped** rather than failing.
A host that has never built `fak` still dispatches — it just dogfoods 0% until you build
the binary (which is exactly what `dogfood_coverage.py`'s `bin_resolvable` KPI flags).
The fleet must keep moving; coverage is a goal, never a gate on getting work done.

**Timeout floors so the gateway never truncates a long turn.** `fak guard` fronts the
real provider in passthrough, and a frontier Claude Code turn with extended thinking can
run well past `fak serve`'s default 60 s planner / 90 s write timeouts — which would cut
the turn off at the gateway. So a guarded worker raises both floors to a generous
600 s (`GUARD_TIMEOUT_FLOOR_S`) via `FAK_PLANNER_TIMEOUT_S` / `FAK_HTTP_WRITE_TIMEOUT_S`,
**without** clobbering an explicit operator value. A spawned worker is also wall-clock
bounded (default 1800 s, opt out with `--timeout-s 0`) so a wedged session cannot burn
tokens forever.

One more safety note for the shared-gateway tiers: a gateway bound beyond loopback with
no required key is an unauthenticated kernel reachable off-host. Both `fak guard` and
`fak serve` warn loudly about this. On Tier 1 / Tier 2 always pair a non-loopback
`--addr` with `--require-key-env`, and keep the gateway on a private network (Tailscale
or the VPC), never the public internet.

---

## See also

- [`DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md) — the one-command dogfood launcher and the `/v1/messages` adjudication proxy
- [`docs/fak/server-quickstart.md`](server-quickstart.md) — every way to start a `fak serve` gateway (auth, policy, in-kernel, cloud)
- `cmd/fak/guard.go` — the `fak guard` front door (child-only base URL, subscription default, default-on hash-chained journal)
- `tools/dogfood_coverage.py` — the coverage scorecard (run it to measure the 3x)
- `tools/gcp_accel.py` — the GCP accelerator ladder for the GPU-burst in-kernel path
