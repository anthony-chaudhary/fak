---
title: "Fleet self-discovery spine"
description: "How machines and the agents running on them find each other in fak, and where that fleet-wide view surfaces for operators to act on."
---

# Fleet self-discovery spine

How machines and the agents on them find each other, and where that view surfaces.
This is the design behind `internal/fleetspine` — the minimal working spine — plus the
iteration path (agent-view / A2A / other transports / Slack overlap) it opens.

## The problem

The `fak info` fleet panel shows cross-machine health, but discovery was **git-mediated
shared-filesystem only**: each machine writes `tools/_registry/machines/<id>.json` and peers
see it only after a `git pull`. A box that shares no git remote (a fresh cloud node, a Mac
verify node) never appears; even peers that do share it lag by a git round-trip. "Show all
active sessions in the fleet" was answerable only for repo-wired, recently-pushed machines.

## The spine (shipped: local-only, UDP multicast)

`internal/fleetspine` adds a **live network discovery source** that unions with the disk fold:

- **Heartbeat** — a compact, display-only wire message (id, host, verdict, sessions, version,
  timestamp). Scalars only; it mirrors `gateway.SessionFleetMachine` field-for-field, so it can
  never smuggle a token or payload onto the wire.
- **Transport** (interface) — `Advertise` / `Listen`. The shipped concrete transport is
  **UDP multicast** (stdlib `net` only — no new dependency), same-subnet (TTL 1) = "the LAN".
  The interface is the seam every future transport plugs into (below).
- **Registry** — a *passive* peer store: peers push heartbeats, so liveness is pure
  time-since-last-heartbeat (fresh → verdict, past miss-window → STALE, past hard-expiry →
  dropped) — the network twin of the fleetpane snapshot-staleness rule, NOT the gateway's active
  probe/hysteresis loop. Drops self-echo, bounds the map, rate-limits per id, rejects malformed.
- **Merge** — `guardFleetProvider` unions live peers into the disk `FleetDoc` keyed by machine
  id (newest `generated_utc` wins), recomputes the aggregate, and folds. The panel and
  `SessionFleet` are unchanged; a network-only peer renders, and both-empty stays silent.

Enabled by default (zero-config); degrades silently to disk-only when multicast is unavailable.

### Trust / scope (v1)

Cooperative LAN, no heartbeat auth. A spoofed id only affects a **display** panel — never
routing, never tokens. Hardened against malformed/flood, not against a hostile LAN. An optional
shared-secret HMAC is the documented follow-up for untrusted segments.

## Two views the same spine serves

The goal named two audiences. They read the same peer registry through different lenses.

### Operator view (shipped)

The `fak info` fleet panel: who is up, how many sessions, who is stale/needs-action. This is the
`SessionFleet` fold above. It is the human's "is the fleet healthy?" glance.

### Agent view (iteration — the plumbing this unlocks)

An agent/worker wants a different question answered: **"who else is working, and can I talk to
them?"** The spine is the discovery layer *beneath* the agent-to-agent bus (`internal/a2achan`),
which today is an *addressed* send/recv channel (`a2a.to.id` / `a2a.to.locale`, capability-gated)
that presumes you already know who to address. The spine supplies exactly that missing input:

- **Discovery → addressing.** A spine peer's `id` (and, once carried, a locale/endpoint) is the
  address `a2achan` needs. "See other workers" = read the registry; "message one" = hand its id
  to `a2a.send`. The spine answers *who is out there*; A2A answers *send to them*.
- **What a heartbeat should carry for the agent view.** The operator view needs only display
  scalars. The agent view additionally wants a **reachable endpoint** (host:port of the peer's
  gateway) and a **liveness/role hint** (is it serving? draining? what is it working on, at the
  coarse level the payload-free contract allows — e.g. a lane id, never prompt/result text). This
  is an additive heartbeat field set behind the same schema-versioned wire, gated so the
  agent-reachable fields are only populated when the operator opts into agent-visibility.
- **Kernel-first, not a side channel.** A2A already folds through the kernel's capability floor
  (`a2achan/floor.go`: `a2a.send` / `a2a.recv` tokens). Discovery should stay symmetric — a
  worker learns of a peer through the spine, then any message it sends is still adjudicated. The
  spine never grants a capability; it only makes a peer *addressable*.

### Overlap with Slack (iteration — dedupe the publish)

Some of this is already published to Slack (the `#capacity` scoreboard, fleet status history).
The spine and the Slack publish are **two renderings of one fleet fact**, and should converge on
one source so they cannot disagree:

- Today the Slack scoreboard is fed from the disk/registry fold; the spine adds a fresher, wider
  membership set. The iteration is to let the same union (disk ∪ spine) feed **both** the local
  info panel and the Slack post, so a machine discovered live shows up in the scoreboard too —
  and a machine that has gone dark drops from both on the same expiry rule.
- Guard against double-publish: the spine is peer-to-peer (every box sees every box), but Slack
  should be posted by ONE owner (the scoreboard-owning session), not N boxes each posting the
  same roster. The spine gives each box the full view; a single elected publisher renders it.

## Iteration roadmap (other transports)

The `Transport` interface is the whole point — new reach is a new implementation, nothing above
it changes:

- **UDP multicast (shipped)** — LAN, zero-config, same subnet.
- **HTTP rendezvous** — each guard POSTs its heartbeat to a static peer list or a shared
  registry endpoint; crosses subnets and cloud VPCs; reuses the gateway's HTTP server.
- **SSH fan-out** — for boxes reachable only over SSH (the fleet-compute model): a collector SSHes
  each host's `/debug/vars` and folds the results as spine peers.
- **Cloud/gossip** — a rendezvous service or gossip mesh for a fleet with no shared broadcast
  domain and no static list.

Each keeps the Heartbeat wire, the Registry, and the merge intact; only the reach changes.

## Files

- `internal/fleetspine/spine.go` — Heartbeat, Registry, MachineMaps (the union adapter).
- `internal/fleetspine/transport.go` — Transport interface, UDP multicast transport, chan fake.
- `internal/fleetspine/loop.go` — advertiser / listener / expiry loops + the composed Spine.
- `cmd/fak/guard_fleet.go` — the disk ∪ spine union at the FleetDoc layer.
- `cmd/fak/guard_spine.go` — builds + starts the spine on the guard-lifetime context.
- `internal/fleetpane/config.go` — `spine_*` config keys (`FLEET_SPINE_*` env overrides).

## Verification status (honest)

- **Unit + merge logic: PROVEN.** `internal/fleetspine` (registry expiry via injected clock,
  self-drop, bounded map, rate-limit, malformed-drop, MachineMaps↔fold contract; chan-transport
  round-trip + malformed-drop; loop start/stop) and `cmd/fak` fold union (disk∪spine, newest-wins,
  spine-only-renders, both-empty-silent) are green.
- **Live cross-machine discovery: NOT YET verified from a single box.** The real-socket tests
  self-skip because same-host multicast loopback is unreliable on Windows (the dev box) — this is
  an OS loopback limitation, not a code defect: the shipped transport sends out the NIC and a
  packet reaches a *different* machine's socket normally. Verifying it needs two machines on one
  LAN (`TestEndToEndTwoSpinesDiscoverEachOther` is the ready live probe; run it on two boxes, or
  run two `fak manage` sessions on two hosts sharing the group/port and confirm each appears in the
  other's `fak info` fleet panel with no shared git remote).
