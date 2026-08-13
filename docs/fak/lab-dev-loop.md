---
title: "Lab dev loop: develop fak on a remote box from Slack"
description: "Run and develop fak on a remote lab GPU, driven from Slack: the kernel and dev turn stay public while the lab transport stays private."
---

# Lab dev loop — develop fak ON a lab box, drive it from Slack

> **Audience.** Operators who develop fak on remote lab compute and drive it out-of-band from Slack. By the end you'll understand the four-piece loop and the public/private split that keeps it safe to ship.

This is the end-to-end loop for **running and developing fak on lab compute you choose**,
driven out-of-band so you can start it from anywhere (a phone, a laptop) while every byte
of the actual work runs on the box. It ties together four pieces that already exist: a
model served on a lab GPU, a kernel-adjudicated dev turn pointed at it, a Slack control
channel to drive it, and the public fleet view that folds the result.

The split that makes it safe to keep public: the **kernel + the dev turn are public**
(this repo); the **Slack transport is private** (the lab protocol carries lab identifiers,
so it lives in `fak-private`). The seam between them is a data contract — a per-box report
JSON — not a code import. See [the GPU-server private boundary](../gpu-server-private-boundary.md).

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

For a user-visible GLM lab setup, do not hand-copy private host or tunnel coordinates
into the command. The private bridge/tunnel producer writes a local, gitignored target
config at `$FAK_LAB_TARGETS` or the default fak config path (`fleet/lab-targets.json`):

```json
{
  "schema": "fak.lab_targets/v1",
  "targets": [
    {
      "alias": "@lab/glm-5.2",
      "base_url": "http://localhost:PORT",
      "model": "glm-5.2",
      "roster": "C:/local/private/roster.json"
    }
  ]
}
```

The tracked docs and tests use only placeholder aliases and `localhost`. The real file
stays local. `box_id` is optional and should be a display-safe generic ID if present;
when the target validates against a private roster, omit `box_id` so the resolver does
not print the private report key. Validate the alias without printing coordinates:

```bash
fak lab target @lab/glm-5.2 --json
```

Then run the guarded turn through the alias:

```bash
fak manage --remote-serve @lab/glm-5.2 --probe -- codex
```

Alias resolution fails closed unless `fak lab readiness --json` is `READY_FOR_DEV_WORK`,
the local target config exists, and scrubbed reports contain a fresh healthy
`inference.ready`/`inference.degraded` row for the requested model.

A status 200 is not enough: a route can answer and still be too slow to develop on.
The resolver applies a scrubbed **latency budget** (default 30s) to the fresh report's
`probe_latency_ms`. A `ready` row whose probe latency exceeds the budget resolves to
`LATENCY_DEGRADED` with next action
`route-latency-exceeds-dev-budget-refresh-report-or-use-fallback`, `fak lab readiness`
holds lab-backed dispatch closed instead of advertising `READY_FOR_DEV_WORK`, and
`fak guard --remote-serve @lab/glm-5.2` refuses the alias with `LAB_TARGET_NOT_READY`
rather than binding a route a coding worker cannot use. The producer records the scrubbed
sample when self-reporting the box:

```bash
fak lab report --id mac-a --state live \
  --inference ready --engine fak --model glm-5.2 --probe-latency-ms 143737
```

`fak lab target @lab/glm-5.2 --json` echoes the decision as `latency_budget_ms` and
`probe_latency_ms` so the degraded status is visible without printing route coordinates.

Because `--remote-serve` forces the OpenAI-compatible wire, the wrapped agent must be one
that reads `OPENAI_BASE_URL` (Codex, OpenCode, Aider) — not Claude Code, which speaks the
Anthropic wire (guard rejects `--remote-serve` with `--provider anthropic`).

```bash
# on the lab box: serve a model in fak's own kernel on the GPU.
# Linux/NVIDIA box (a -tags cuda build):
FAK_Q4K=1 fak serve \
  --gguf /srv/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --engine inkernel --backend cuda \
  --addr 0.0.0.0:8080

# Apple-Silicon box (darwin/arm64+cgo) — GPU prefill + resident Q8 decode on a
# dense Qwen-class Q8 GGUF. --metal is the Metal seam, mutually exclusive with --backend:
fak serve \
  --gguf /srv/models/qwen2.5-coder-7b-instruct-q8_0.gguf \
  --engine inkernel --metal \
  --addr 0.0.0.0:8080

# from the box (or the bridge session on it): run a kernel-adjudicated dev turn,
# inference on this box's GPU, kernel local. The agent reads OPENAI_BASE_URL.
fak manage --remote-serve localhost:8080 -- codex
```

The banner shows the upstream as a **remote fak serve on a lab box** so you can see at a
glance that the turn's compute is where you put it, not on a public API.

## The Codex prompt-size latency envelope

Small OpenAI-compatible clients already complete through the lab route: a minimal
`fak guard --remote-serve @lab/glm-5.2 --provider openai --probe -- python <smoke>` returns
model text well inside the probe window. The **Codex** shape is different, and it is worth
naming why so the next operator does not re-diagnose it:

```bash
fak manage --remote-serve @lab/glm-5.2 --probe -- \
  codex exec --ephemeral --ignore-user-config --ignore-rules --skip-git-repo-check \
    -C <tmp> --sandbox read-only -m glm-5.2 \
    "Reply with exactly GLM52_OK. Do not inspect files. Do not call tools."
```

Even a no-repo, no-tools Codex turn carries Codex's own system + developer prompt. After
proxy shaping caps `max_tokens` and strips tool schemas, the forwarded request is still on
the order of ~20 KB across a few messages — the bulk is Codex's prompt, not repo context.
Through the current bridge/model latency envelope that prompt does **not** decode inside a
10-minute probe, while the small Python smoke does. The blocker is prefill+decode latency
on a large fixed prompt, not the guard wiring, and not repo context leaking in.

### Honest routes to close it

Pick by what the lab box can give you; each is honest about any request shaping:

- **Faster serving endpoint.** Serve the model on a box/engine whose prefill+decode clears
  the ~20 KB prompt inside the probe window (a resident GLM-5.2 lane, or a smaller/faster
  coder GGUF for the smoke). This is the route that closes the acceptance path without
  changing what Codex sends.
- **Direct tunnel.** Point `--remote-serve` at a lower-latency localhost tunnel to the
  serving box so bridge round-trips are not added to the decode budget. Keep the tunnel
  coordinates in the gitignored local target config, never in tracked files.
- **Prompt compaction.** Reduce Codex's fixed prompt with its own flags before it reaches
  the wire (fewer instructions, no tool schemas) so less prefill has to decode. Any shaping
  the proxy performs (the `max_tokens` cap, the tool-schema strip) must be recorded in the
  witness so the smoke stays honest about what was sent.
- **Explicit Codex smoke mode.** A dedicated smoke that sends the minimal Codex-compatible
  request and states plainly what it trimmed — honest shaping over a green-looking full run.

The explicit smoke route is now witnessed from this machine: the private loopback proxy
collapsed the no-repo/no-tools Codex request to the final user prompt, capped output at 16
tokens, stripped tool schemas, and recorded that shaping as a **smoke-only** route. With
that shaping, the command above returned `GLM52_OK` through `fak guard
--remote-serve @lab/glm-5.2` in 282.8s. This proves the end-to-end wiring and lab inference
for a Codex-shaped client.

The unshaped full Codex coding turn is still **not yet** witnessed: the fixed Codex
system+developer prompt remains too large for the current bridge/model latency envelope
inside the probe window. Closing that stronger path still requires a faster serving lane,
a lower-latency direct tunnel, or real prompt compaction before the wire. See #2974 for the
tracking issue and #2952/#2953 for the resolver and live-witness siblings.

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
written down as `gpu-server-a`/`gpu-server-class`/`lab`, never a real host or channel), folds the per-box
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

If the box is also serving a model, publish the separate inference-usefulness status so
`fak lab status` can answer whether the lab machine is useful for inference, not just
reachable:

```bash
fak lab report --id mac-a --state live \
  --inference ready --engine fak --model qwen --output-tps 1.75
```

Use `ready` or `degraded` only when the serving stack can take inference work. Use
`warming`, `blocked`, or `unknown` when it cannot. The model/engine/reason labels must stay
generic; never publish a URL, host, channel id, token, private model path, or raw bridge
transcript.

When you have a scrubbed end-to-end probe latency, record it with `--probe-latency-ms`.
A `ready`/`degraded` box whose sample exceeds the developer-useful budget (default 30s) is
held out of `READY_FOR_DEV_WORK` and surfaced as a latency-degraded route, so a 200-but-slow
lane never advertises itself as usable:

```bash
fak lab report --id mac-a --state live \
  --inference ready --engine fak --model glm-5.2 --probe-latency-ms 143737
```

After reports are populated, derive the dispatch gate from the scrubbed status instead of
hand-marking readiness:

```bash
fak lab readiness --from-reports --write-default --json
```

`--from-reports` derives the link-state phase (`CLEAR`/`WAITING`/`WORKING`) from the
scrubbed lab status + inference reports; `--from-status` is still accepted as a deprecated
alias. This writes `READY_FOR_DEV_WORK` only when at least one healthy reported box has useful
fresh inference (`ready` or `degraded`). Warming, stale, or missing inference reports hold
lab-backed dispatch closed.

## Boundary rules (do not trip)

- The private control plane stays in `fak-private`. Never add `private bridge/control packages` or
  `*slack*bridge*` paths here — the commit gate (`tools/check_committed_files.py`) refuses
  them, and `internal/pythongate` refuses a new `tools/*.py`.
- No real host, IP, channel id, or token in any tracked file. Real values live in
  gitignored local files (`fak-mac.local.ps1`, `.env.slack.local`) and resolve through
  `FAK_*` overrides — the convention in [scrubbing real values](scrubbing-real-values.md).

## See also

- [Always-on dogfood server](always-on-dogfood-server.md) — the 24/7 framing this loop is
  the lab-GPU lane of; section 2 covers the GPU-burst ladder via `tools/gcp_accel.py`.
- [GPU-server private boundary](../gpu-server-private-boundary.md) — what is public vs private.
- [private-comms-channel](../private-comms-channel.md) — how to reach the bridge.
- [fleet.md](../fleet.md) — the public fleet fold + readiness score.
