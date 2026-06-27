---
title: "Lab dev loop: develop fak on a remote box from Slack"
description: "Run and develop fak on a remote lab GPU, driven from Slack: the kernel and dev turn stay public while the lab transport stays private."
---

# Lab dev loop — develop fak ON a lab box, drive it from Slack

This is the end-to-end loop for **running and developing fak on lab compute you choose**,
driven out-of-band so you can start it from anywhere (a phone, a laptop) while every byte
of the actual work runs on the box. It ties together four pieces that already exist: a
model served on a lab GPU, a kernel-adjudicated dev turn pointed at it, a Slack control
channel to drive it, and the public fleet view that folds the result.

The split that makes it safe to keep public: the **kernel + the dev turn are public**
(this repo); the **Slack transport is private** (the lab protocol carries lab identifiers,
so it lives in `fak-private`). The seam between them is a data contract — a per-box report
JSON — not a code import. See [the GPU-server / Slack boundary](../dgx-slack-boundary.md).

```
  you (Slack, from anywhere)
        │  post a task line
        ▼
  private bridge ──▶ lab box you chose
        ▲               │  runs:  fak guard --remote-serve <box>:8080 -- <agent> <task>
        │               │           ├─ kernel adjudicates every tool call (local, on the box)
        │  postback      │           └─ INFERENCE runs on the lab GPU (the remote fak serve)
        └───────────────┘  + writes one fak.fleet.report/v1 line
                                     │
                                     ▼
                            fleetctl status  (public fold + readiness score)
```

## The one new public piece: `fak guard --remote-serve`

`fak guard` runs its own kernel gateway on a local loopback port and execs the agent;
`--remote-serve HOST[:PORT]` points that agent's **inference** at a `fak serve` running on
a lab box you chose. The kernel still adjudicates every tool call locally on the box, but
the model forward runs on the lab GPU. Port defaults to `8080` (the documented `fak serve`
addr). It is shorthand for the OpenAI-compatible wire `fak serve` exposes, with the `/v1`
suffix the chat route lives under added to the upstream base for you, and it **preflights
`GET /healthz` AND `GET /v1/models`** so a box that is down — or that answers health but is
not serving the `/v1` surface — fails loud before the gateway binds rather than 404-ing on
the first turn.

Because `--remote-serve` forces the OpenAI-compatible wire, the wrapped agent must be one
that reads `OPENAI_BASE_URL` (Codex, OpenCode, Aider) — not Claude Code, which speaks the
Anthropic wire (guard rejects `--remote-serve` with `--provider anthropic`).

```bash
# on the lab box: serve a model in fak's own kernel on the GPU
FAK_Q4K=1 fak serve \
  --gguf /srv/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --engine inkernel --backend cuda \
  --addr 0.0.0.0:8080

# from the box (or the bridge session on it): run a kernel-adjudicated dev turn,
# inference on this box's GPU, kernel local. The agent reads OPENAI_BASE_URL.
fak guard --remote-serve localhost:8080 -- codex
```

The banner shows the upstream as a **remote fak serve on a lab box** so you can see at a
glance that the turn's compute is where you put it, not on a public API.

## Driving it from Slack (private bridge)

The Slack control channel that reaches the lab boxes is private — it speaks a lab protocol
and carries a host, a channel id, and a token, none of which ever enter this repo. The
entry point is [the private comms stub](../private-comms-channel.md); the live runbook is
in `fak-private`. The shape of the loop, with no lab identifiers, is:

1. Post a task line in the control channel.
2. The bridge runs, on a persistent session on the box you chose:
   `cd <repo> && fak guard --remote-serve localhost:8080 -- <agent> '<task>'`.
3. The box posts the guard exit summary back to the channel and writes one
   `fak.fleet.report/v1` line into the reports directory (via `fak lab report`, or the
   bridge's own writer) so `fak lab status` folds it into the public fleet view.

Because the work runs in a session on the box, the body of the work never crosses Slack —
only the task line in and the summary out. You drive it from anywhere; the compute stays
on the machine you picked.

## Folding the result (public)

The fast front door is `fak lab status` — one command, no flags, that answers "which
lab nodes are alive right now?" It ships a **generic** default roster (the lab boxes
written down as `dgx-a`/`a100x8`/`lab`, never a real host or channel), folds the per-box
report JSON against it, and renders the same bounded view + 0–100 readiness score
`fleetctl` does (they share `internal/fleet`):

```bash
fak lab status            # the embedded roster, reports from ~/.config/fak/fleet/reports
fak lab status --all      # add a per-box table
fak lab ls                # just list the boxes in the roster
```

When no live reports exist yet, `fak lab status` degrades **honestly** — every box reads
`unknown` (not down) and it tells you how to populate liveness. The reports dir resolves
`--reports` → `$FAK_FLEET_REPORTS` → `~/.config/fak/fleet/reports` (the bridge's drop
path). The standalone `fleetctl status --roster R --reports DIR` is still there for an
explicit roster/reports pair — see [fleet.md](../fleet.md).

### Self-reporting a box (no bridge needed)

A box can write its own `fak.fleet.report/v1` line with `fak lab report`, closing the
loop for that box without the private bridge — useful on a box you can run `fak` on
directly (the CPU GLM host, a Mac verify node):

```bash
fak lab report --id da-cpu --state live --version "$(fak version)"
```

Keep `--note` generic (no host/IP/channel/token) — it is rendered verbatim in the public
fleet view.

## Boundary rules (do not trip)

- The Slack control plane stays in `fak-private`. Never add `cmd|internal/*dgx*` or
  `*slack*bridge*` paths here — the commit gate (`tools/check_committed_files.py`) refuses
  them, and `internal/pythongate` refuses a new `tools/*.py`.
- No real host, IP, channel id, or token in any tracked file. Real values live in
  gitignored local files (`fak-mac.local.ps1`, `.env.slack.local`) and resolve through
  `FAK_*` overrides — the convention in [scrubbing real values](scrubbing-real-values.md).

## See also

- [Always-on dogfood server](always-on-dogfood-server.md) — the 24/7 framing this loop is
  the lab-GPU lane of; section 2 covers the GPU-burst ladder via `tools/gcp_accel.py`.
- [GPU-server / Slack boundary](../dgx-slack-boundary.md) — what is public vs private.
- [private-comms-channel](../private-comms-channel.md) — how to reach the bridge.
- [fleet.md](../fleet.md) — the public fleet fold + readiness score.
