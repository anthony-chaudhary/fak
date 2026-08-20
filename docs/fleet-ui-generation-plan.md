---
title: "Fleet terminal UI and UX generation plan"
description: "A staged plan for supervising fak machines, sessions, loops, objectives, approvals, and fleet health from the existing terminal surfaces."
---

# Fleet UI/UX generation plan

> Status: initial plan refreshed 2026-08-11 in [#6477](https://github.com/anthony-chaudhary/fak/issues/6477). The G0 spine is tracked by
> [#6476](https://github.com/anthony-chaudhary/fak/issues/6476).

This plan organizes the first generations of terminal UI/UX for maintainer fleet concepts:
machines, sessions, loops, objectives, approvals, and related fleet health. It starts from the
TUI and command surfaces that already work; it does not propose a new frontend stack.

## Value frame

- **For:** an operator supervising several long-running fak loops from one terminal.
- **Problem:** machine posture is visible in `fak info`, but loop inventory and recovery evidence
  live behind separate `fak fleet-pane` commands. The operator must switch surfaces and mentally
  join two read models before deciding what needs attention.
- **Today:** use the Overview or Agents tab for machine/session posture, then run
  `fak fleet-pane loop-list`, `loop-check`, or `loop-audit` for loops.
- **Better because:** a single attention-first Fleet workspace answers “what needs attention and
  what is the next check?” while retaining the existing commands as the source of truth.

The real next-best alternative is improving those separate CLI reports. A UI generation earns its
keep only when it removes an operator decision or a surface switch; decorative duplication does not.

## Current-state inventory

| Concept | Authoritative source today | Existing surface | G0 treatment |
|---|---|---|---|
| Machines and sessions | guard-published `/debug/vars.fleet` / `gateway.SessionFleet` | `fak info` Overview and Agents | Reuse the aggregate and bounded machine sample. |
| Loop inventory and typed health | `internal/fleetpane` config and checks | `fak fleet-pane loop-list|loop-check|loop-audit` | Join read-only checks into the Fleet workspace. |
| Recovery | `fleetpane.RecoveryAction` and existing CLI verbs | copy/run a `fak fleet-pane ...` command | Show a copyable next command; do not mutate in the TUI. |
| Objectives | trajectory-control hierarchy | planned by #2571 | Defer to G1; do not fake progress from worker activity. |
| Out-of-band control | closed control vocabulary | planned by #2753 | Defer mutations until capability and witness semantics exist. |
| Durable work queue | dispatch queue | planned by #2928 | Keep separate from loop runtime state. |
| External/mobile read model | gateway aggregate | planned by #4231 and #4227 | Do not make G0 depend on a new network API. |

Two seeds are already shipped: the fleet concepts front door in `docs/concepts/fleet.md` and the
machine-posture rendering from #6060. The missing spine is not “build a dashboard”; it is joining
that machine seed to the existing loop read model inside the terminal surface operators already use.

## Generation map

| Generation | Operator question | Surface and scope | Exit witness |
|---|---|---|---|
| **G0 — See and triage** | What needs attention, and what should I inspect next? | Read-only Fleet workspace in `fak info`: headline, machines, attention-first loops, typed reason, age/source, copyable command. | Captured render plus a live/selfcheck run; #6476. |
| **G1 — Understand progress** | Is the fleet advancing the objective or merely busy? | Add objective hierarchy and progress/evidence drill-in from #2571; preserve unknown rather than infer progress. | Cross-session fixture with objective → loop → worker evidence and a stale/unknown case. |
| **G2 — Act safely** | Can I steer or recover this item without racing another operator? | Capability-aware actions from #2753, explicit confirmation, CAS/precondition, receipt, and independent read-back. | Race/denial/replay witnesses; no optimistic “clicked means done.” |
| **G3 — Reach remotely** | Can I inspect and act away from the terminal? | External/mobile clients over #4231/#4227/#4229/#4230 after the read and control contracts stabilize. | Contract tests plus remote read-back; no terminal-only state. |

This ordering is intentional: **read model → progress model → control model → additional clients**.
Starting with buttons, a web shell, or mobile cards would freeze an unproved vocabulary and duplicate
the terminal seam.

## G0: minimal working spine

### User journey

1. Start `fak info` against a fixture or live guard endpoint.
2. Select **Fleet** using the existing keyboard or mouse tab model.
3. Read one headline that states overall posture and freshness/source.
4. Scan loops ordered by attention first, then stable name.
5. For an attention row, read the typed state and evidence, then copy the existing next command.
6. Return to Overview or Agents without losing their current behavior.

### Wireframe

The renderer should degrade by dropping detail, not by hiding state. Exact glyphs are implementation
choices; the information order is the contract.

```text
 Fleet  ACTION   machines 3 · sessions 14 · loops 4 (1 attention)   observed 8s ago

 NEEDS ATTENTION
 ! dispatch-main   STALLED   no progress 18m   next: fak fleet-pane loop-check dispatch-main

 LOOPS
 ✓ research        RUNNING   progress 2m ago
 ✓ release         HEALTHY   tick 41s ago
 ? nightly         UNKNOWN   source unavailable   next: fak fleet-pane loop-audit

 MACHINES
 ! build-02        STALE     3 sessions   r81+g12ab34
 ✓ control-01      OK        8 sessions   r84+g98cd76

 source: guard fleet + local fleetpane checks · read-only · Enter copies/expands, it does not act
```

At narrow widths the invariant is `name + typed state + next check`; counts, versions, and explanatory
detail may wrap or collapse. Color is redundant: words and glyphs must carry the same distinction.

### Information architecture

- **Headline:** verdict, machine/session/loop counts, attention count, freshness, provenance.
- **Attention:** all actionable loops first, bounded only with an explicit “N more” row.
- **Loops:** name, typed state, age/progress evidence, action hint or existing CLI command.
- **Machines:** reuse the existing bounded attention-first machine sample rather than inventing a
  second machine renderer.
- **Empty/error states:** distinguish no configured loops, unavailable source, stale snapshot, and
  healthy zero-attention. Never render all four as an empty table.

### Interaction contract

G0 is read-only. Navigation, expand/collapse, and copy are allowed; start, stop, recover, approve,
retry, and steer are not. The row must tell the truth about the boundary (`read-only`) and route the
operator to a command that already owns the operation. This keeps keyboard, mouse, and headless use
on one semantic seam and avoids an unwitnessed TUI-only control path.

### Acceptance and proof

The issue is complete only when all of these are witnessed:

- Fleet is reachable through the existing tab and focus model by keyboard and mouse.
- Overview and Agents captured renders remain unchanged unless an intentional migration is asserted.
- Healthy, attention, stale/unavailable, and no-config/empty fixtures render distinct outcomes.
- State is consumed from `internal/fleetpane`; the renderer does not reclassify loop health.
- An actionable row contains a copyable existing `fak fleet-pane` next step.
- A captured render asserts the whole pane, including attention order and narrow-width behavior.
- A captured live or `-selfcheck` run demonstrates the real seam, not only component formatting.

## Backlog organization

### G0 — spine now

- [#6476](https://github.com/anthony-chaudhary/fak/issues/6476) — add the first-generation
  read-only Fleet workspace. This is the only required implementation issue for the first spine.
- [#3571](https://github.com/anthony-chaudhary/fak/issues/3571) — land-throughput and CAS-contention
  row. Useful observability, but not a blocker for the Fleet workspace.

### G1 — comprehension after the spine

- [#2571](https://github.com/anthony-chaudhary/fak/issues/2571) — objective board and cross-session
  hierarchy. Its progress vocabulary must remain separate from loop liveness.
- [#2928](https://github.com/anthony-chaudhary/fak/issues/2928) — durable kanban queue. Join it only
  after its state machine is authoritative; do not model queue state in the renderer.

### G2 — controls after comprehension

- [#2753](https://github.com/anthony-chaudhary/fak/issues/2753) — closed out-of-band control
  vocabulary. It supplies the operation semantics before buttons exist.
- Approval UI belongs to the same generation and must include capability, precondition, receipt, and
  read-back states rather than a generic confirmation modal.

### G3 — additional clients

- [#4231](https://github.com/anthony-chaudhary/fak/issues/4231) — external fleet-status endpoint.
- [#4227](https://github.com/anthony-chaudhary/fak/issues/4227) — mobile fleet read model.
- [#4229](https://github.com/anthony-chaudhary/fak/issues/4229) — mobile steer/control.
- [#4230](https://github.com/anthony-chaudhary/fak/issues/4230) — mobile approval queue.

## Decisions and non-goals

1. **Extend the TUI; do not start a web application.** The TUI already has polling, fixtures,
   keyboard/mouse navigation, color modes, and captured-render tests.
2. **Reuse typed backend states.** Presentation may order and shorten; it may not invent health,
   progress, or recovery classifications.
3. **Attention first, not activity first.** Running worker count is context, not success.
4. **Progress and liveness stay distinct.** A healthy loop can make no objective progress; an
   advancing objective can temporarily have no live worker.
5. **Mutations wait for receipts.** G0 links to existing commands. G2 actions require explicit
   capability and witnessed read-back.
6. **No private infrastructure details.** Public fixtures use scrubbed machine and account identities.

## Planning freshness rule

Refresh this plan when one of the source contracts changes (`SessionFleet`, `fleetpane` loop checks,
trajectory hierarchy, or control vocabulary), or when a generation exit witness lands. A refresh
must update the current-state table and issue links; adding a new mockup without reconciling the
source contract is not a refresh.

## Drill-in constraint: Fleet observes; a session attachment operates

This generation map governs the Fleet **overview**. It must not become a second, progressively
richer session product. The cross-client invariant is defined in
[`session-client-contract.md`](session-client-contract.md): selecting a first-party session
resolves its logical `session_id` and opens a full attachment to the same authoritative state a
terminal client operates.

Consequences for every generation:

- G0 rows remain projections and copyable diagnostics.
- G1/G2 may improve selection and attention ordering, but session controls belong to the shared
  attachment protocol rather than Fleet-only handlers.
- G3 external/mobile clients discover and attach by logical session ID. A gateway URL, PID,
  provider thread, account, model, or compute host is a replaceable execution-epoch binding, not
  the bookmark or identity shown as “the session.”
- A deliberately reduced dashboard or integration must identify itself as a projection and offer
  a handoff to a full client; it must not present drill-in as though it were the session.

The required drill-in witness is cross-client: select one Fleet row, attach at the row's event
address, act through the shared capability set, and observe that addressed effect from the
reference terminal client without transcript copying or state reconstruction in the renderer.
