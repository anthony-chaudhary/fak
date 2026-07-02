---
title: "Meaningful control, one sidecar, automatic context: the three-plane split"
description: "Disambiguation + strategy binding the operator's 2026-07-01 goal to the tree: an INTENT plane the user controls (epic #2208), a LEGIBILITY plane that renders the same experience and shared items across agents/surfaces/platforms (epic #2209), and a HOUSEKEEPING plane that is zero-knob automatic by default (owned by the existing #2198/#1844 programs). One litmus test decides which plane a knob belongs to; two ratchets keep the boundary honest."
date: 2026-07-01
---

# Meaningful control, one sidecar, automatic context: the three-plane split

Status: disambiguation + strategy note for epics
[#2208](https://github.com/anthony-chaudhary/fak/issues/2208) (meaningful
control; children #2210–#2212) and
[#2209](https://github.com/anthony-chaudhary/fak/issues/2209) (sidecar/rollup;
children #2213–#2216). Nothing ships from this note; every gap it names is a filed
issue with a witness. Companions:
[CONCEPT-AUTOMATIC-CONTEXT](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md) (#2198,
the automatic plane's doctrine),
[RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY](RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md)
(the durable store the shared-items plane reads).

## The operator's ask

> 1) users [get] more control for things they should meaningfully control
> 2) roll up / side car view so [the] same experience and shared items across
> various agents, surfaces, platforms etc. 3) the management of context and
> cache is super automatic by default. … a user may wish to reference an
> existing session to start another one, but they shouldn't have to do that
> for routine context management.

## The disambiguation: one litmus test, three planes

The three asks are not three features; they are one boundary and the two
surfaces it creates.

> **A knob belongs to the user iff it encodes intent the system cannot infer
> from telemetry.** WHICH session to branch from. WHAT must survive eviction.
> HOW MUCH to spend (tokens, wall-clock, dollars). WHAT needs consent before
> it is irreversible. Everything else — residency, warmth, TTLs, breakpoints,
> compaction, hygiene — is housekeeping the system can decide from what it
> already measures, and must be zero-knob by default.

- **INTENT plane (control — epic #2208).** Promote the intent levers to
  first-class, consistent verbs. The operator's own example sits here:
  referencing an existing session to start another is `fak session branch`
  (#1200, lifecycle epic #1193) — a deliberate act, never a chore.
- **HOUSEKEEPING plane (automatic — epic #2198 + #1844/#1490/#1570/#1860).**
  Routine context and cache management never requires a user action. This
  plane already has its doctrine (seven laws), its keystone ratchet (R1
  #2199), and its cache posture (`--managed-cache`, epic #1844). This note
  adds nothing to it — by design.
- **LEGIBILITY plane (sidecar — epic #2209).** Where both planes become
  visible, identically, everywhere: the same session/account/lane/posture
  facts and the same shared items (task records, lessons, verdicts) no matter
  which agent (the `harnessprofile` set: claude/codex/opencode/aider/hermes),
  which surface (CLI, TUI, Slack, gateway HTTP), which host. Control
  exercised once applies everywhere; an item is the same item everywhere.

The worked example, fully disambiguated: *branch-from-session* is INTENT
(#1200); *routine carryover across resets/relays* is HOUSEKEEPING (#2198
R5/R7, #1860 batons); the sidecar renders the lineage either way (the
`continuation=`/`parent=`/`gen=` fields already exist on session state).

Two ratchets police the boundary, in opposite directions:

| Ratchet | Counts | Direction | Owner |
|---|---|---|---|
| R1 manual-overlay counter (#2199) | user-required HOUSEKEEPING knobs | down, by automating them away | #2198 |
| Knob census (A1, epic #2208) | INTENT levers with missing/partial routes | down, by promoting them to full verbs | #2208 |

A knob that moves the wrong counter is misfiled; the census (A1) adjudicates
both, so the boundary is a generated table, not a taste.

## What exists vs what was filed (verified against HEAD, 2026-07-01)

**Already real** (scouted, with file evidence): `fak session`
stop/pause/resume/throttle/run/budget/pace/envelope/priority with `--if-rev`
concurrency (`cmd/fak/session_cmd.go`); shipped steer (#760/#850); default-on
automatic context (ctx-view 8000, compact-history 48000, tool-result elision,
precompact shadow — `cmd/fak/guard.go`); `--managed-cache` posture
(`cmd/fak/guard_managed_cache.go`, #1844); ctxplan pins as mechanism
(`internal/ctxplan/plan.go`); the cross-host session substrate
(`internal/leaseref/session.go` `ListSessions` over
`refs/fak/locks/session-*`); the shared-task-record contract
(`docs/shared-task-record-contract.md`).

**Verified gaps → filed children:**

| Gap (evidence) | Plane | Child |
|---|---|---|
| Every user-facing knob unclassified intent-vs-housekeeping | boundary | A1 [#2210](https://github.com/anthony-chaudhary/fak/issues/2210) knob census (epic #2208) |
| Pins have mechanism, no user verb; live planner takes no pin source | INTENT | A2 [#2211](https://github.com/anthony-chaudhary/fak/issues/2211) `fak session pin` (epic #2208, binds #844) |
| Envelope `wall=`/`spend=`/`throughput=` parse and apply to nothing (`session_cmd.go`) | INTENT | A3 [#2212](https://github.com/anthony-chaudhary/fak/issues/2212) envelope routes (epic #2208) |
| Session health is Claude-only; codex/opencode/aider invisible (`fleet.go`, `fleet_sessions.py`) | LEGIBILITY | B1 [#2213](https://github.com/anthony-chaudhary/fak/issues/2213) cross-agent census (epic #2209) |
| Four identity spaces (trace_id / drive-state id / leaseref ref / harness identity) never join | LEGIBILITY | B2 [#2214](https://github.com/anthony-chaudhary/fak/issues/2214) `fak.session.descriptor.v1` (epic #2209) |
| No two surfaces render the same fields the same way (`fak fleet` vs `fleet_top.py` vs `rollup` vs 14 Slack silos) | LEGIBILITY | B3 [#2215](https://github.com/anthony-chaudhary/fak/issues/2215) sidecar pane v0 (epic #2209) |
| Shared-task-record contract is a fixture wired to no live surface | LEGIBILITY | B4 [#2216](https://github.com/anthony-chaudhary/fak/issues/2216) shared items live (epic #2209) |

**Adopted by reference, not re-filed:** #1200/#1193 (branch + lifecycle),
#844 (pin mechanics), #2156 (reversibility consent), #1203 (fleet session
fold — the keystone reader), #748 (agent-OS process table), #1601/#1612
(usage rollup), #2141–#2145 (lessons ledger), #2035 (monitor fold).

## Strategy: sequencing, worst-regret-first

1. **B2 #2214 descriptor schema first** — the join-key spine; data-only, no
   new state; everything in the legibility plane (B1, B3, #1203) consumes it.
2. **A3 #2212 envelope routes** — smallest promoted verb; finishes a lever
   that already half-exists, and its structured refusal reasons exercise the
   closed-vocabulary discipline the other ceilings will reuse.
3. **B1 #2213 census, then B3 #2215 pane v0** — census before pane so the
   pane's first render already spans agents; parity contract (one render
   model, two serializers) from day one.
4. **A2 #2211 pin verb** after #2198's R2 one-ledger join stabilizes, so the
   pin lands on the joined residency+warmth record rather than on today's
   split ledgers.
5. **A1 #2210 census + ratchet** ships data-only immediately; the CI ratchet
   arms only after two stable runs (same discipline as R1).

Theme 3 continues unchanged under #2198/#1844 — the strategy here is
explicitly to add zero knobs to that plane.

## Honesty fences

- Nothing in this note is shipped by being written here; every child carries
  its own witness and closes only on a resolving commit.
- The sidecar renders adjudicated facts; it never re-adjudicates them, and
  WITNESSED is never blended with OBSERVED (same fence as `fak rollup`).
- The census counts are generated, deterministic, and non-self-reported; a
  hand-maintained knob list would be exactly the drift this note exists to
  kill.
- #2175–#2184 and #2185–#2194 are pairwise-duplicate filings of the same
  managed-cache follow-ons (noted upstream); this note cites the epic #1844
  rather than either batch.
