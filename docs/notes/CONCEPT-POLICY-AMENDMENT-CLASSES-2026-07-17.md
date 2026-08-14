---
title: "Policy amendment classes: who may amend the guard floor, in which direction, through which channel"
description: "Makes the fak manage floor's implicit per-knob mutability discipline explicit as a capability = (direction × channels). Four amendment classes — FROZEN, RATCHET, GATED-WIDEN, SELF-AMENDABLE — replace the scattered clamps and doc comments with one model, a registry, and a conformance test that fails the build if a knob is unclassified."
date: 2026-07-17
---

# Policy amendment classes

Status: concept note for epic #5170. Track A ships the model as a registry
(`internal/amendclass`) plus a reflection conformance test; Tracks B–F build the
observability, ratchet-completion, and frontier work on top of it. This note is
the committed prose the code points back to.

## The problem

`fak manage` already enforces a rich, per-knob mutability discipline — but it is
**implicit**. It lives in scattered doc comments (`decide.go`, `guard_allow.go`,
`policy_reload_widen.go`) and is enforced by scattered clamps (`sanitizeProfile`,
`sanitizeAdvisoryReasons`, `AdvisoryEligible`, `mustRun`, `diffPolicyWidening`,
`protectGuardPolicyConfig`). There is no single model that answers, for any
policy knob: **who may change it, in which direction, and through which channel.**

That makes it hard to (a) reason about what an autonomous run can and cannot
loosen, (b) safely add flexibility on the axes we *want* to be flexible, and (c)
prove the core protections stay frozen.

The user ask that motivates the epic: *classify policy so we know which parts can
be (1) self-modified by the agent, (2) online-modified by an external process,
(3) the current heaviest/immutable setup — so the floor is flexible where it
should be, yet the core protections never move.*

## Why not a linear tier

This is deliberately **not** an ordered rung / tier / level. Those words are
already taken: adjudicator sub-`rung`s are evaluation order; architest and
dispatch `tier`s are the import graph; risk / lock / reversibility `class`es are
their own ladders; repo-guard severity `level`s are yet another. And per
[`EXPLAINER-least-privilege-across-fak-2026-07-02.md`](EXPLAINER-least-privilege-across-fak-2026-07-02.md),
*least privilege wants capability checks, not level comparisons.*

So a knob's amendment class is a **capability = (direction it may move) × (set of
channels authorized to move it)**, aligned with the lock-class ladder in
[`CONCEPT-CORE-SOFT-LOCKS-2026-06-30.md`](CONCEPT-CORE-SOFT-LOCKS-2026-06-30.md).

## The four amendment classes

- **FROZEN (core).** Compiled-in floor; **no channel may weaken it**, and it is
  self-protected — the policy + overlay files are injected into `SelfModifyGlobs`
  by `protectGuardPolicyConfig`, so the wrapped agent cannot edit its own guard
  config. Members: the hardwired cloud-metadata SSRF egress floor (`mustRun` for
  every class), the reversibility preview-confirm gate (#2156), the structural
  genuine-danger arg-rules (rm -rf / RCE-pipe / out-of-tree-write / shell-dialect
  / fork-bomb), the write-class self-modify rungs (`mustRun` on `classWrite`), the
  `AdvisoryEligible` clamp (POLICY_BLOCK / SECRET_EXFIL / EGRESS_BLOCK can never
  soften), and the gitgate hazard table.

- **RATCHET (tighten-only).** Any channel may *strengthen*; none may *weaken*.
  Members: `EgressExtraDenyHosts`, `EgressBlockHosts` / `EgressBlockLists`,
  `SecretPatterns` (unioned, extend-never-replace), `ArgPredicates`
  (restrict-only), added `SelfModifyGlobs`, `RedactFields`, `LintWrites` (opt-in),
  the `ResearchEgressAllowHosts` restrict allowlist, and the `fak manage deny`
  overlay.

- **GATED-WIDEN (out-of-band only).** May loosen the *convenience* surface — but
  only via a channel the wrapped agent cannot reach, and every widening is
  journaled (and, per Track D, optionally TTL'd and scope-bounded). Members: the
  `fak manage allow` overlay (`Allow` / `AllowPrefix`), `POST /v1/fak/policy/reload`
  under `FAK_POLICY_RELOAD_ALLOW_WIDEN`, `Complain`, `AdvisoryReasons` (clamped to
  heuristic reasons), `Posture` fail_closed→admit_and_log, `SecretPosture`, the
  read-class `RungProfile`, `EgressAllowHosts`, `EgressRestrict`, and
  `AutoRepairSidestep`.

- **SELF-AMENDABLE (frontier — empty today).** What the agent may amend about its
  *own* policy. Today: **nothing** — self-modification is refused by design ("an
  agent can never grant itself a capability"). The epic opens a provably-safe
  envelope: **monotone-tighten only** (a self-write admitted only when
  `diffPolicyWidening` is empty) and/or **propose-only** (the agent queues a
  widening request that an out-of-band operator ratifies). See
  [`RESEARCH-MONOTONE-TIGHTEN-SELF-AMENDMENT-2026-07-17.md`](RESEARCH-MONOTONE-TIGHTEN-SELF-AMENDMENT-2026-07-17.md).

## The five amendment channels (the "how")

1. **compiled-in / shipped** — `guard-default-policy.json` embed,
   `DefaultPolicy` / `DevAgentPolicy`. The current, heaviest baseline.
2. **on-disk operator overlay** — `.fak/guard/allow.json`, `.fak/guard/deny.json`,
   env→user→repo layering; `--policy`. Out-of-band.
3. **live gateway reload** — `POST /v1/fak/policy/reload`. Gated + journaled
   (`AppendConfigSwap`).
4. **operator escalation** — Slack operator-question / `OPERATOR_GATE`. The human
   residue of a held decision.
5. **agent-self** — closed today (the frontier).

## The three questions, answered by the cross-product

| Question | Amendment class | Channel(s) |
|---|---|---|
| **Self-modifiable?** | SELF-AMENDABLE (empty today; frontier) | 5 (agent-self) — via monotone-tighten or propose-only |
| **Online-modifiable by an external process?** | RATCHET + GATED-WIDEN | 2 (overlay), 3 (reload), 4 (operator) |
| **Current setup / heaviest / immutable?** | FROZEN + shipped `DevAgentPolicy` | 1 (compiled-in) |

## Invariants

- No channel weakens a FROZEN knob (a red-team amendment matrix test, #5174,
  asserts this against every channel).
- A RATCHET knob only ever tightens; a GATED-WIDEN loosening is never reachable
  by the wrapped agent and is always journaled (a `CapabilityGrant` record, #5178).
- Every new knob must declare its amendment class — the reflection conformance
  test in `internal/amendclass` fails the build otherwise, so the model can't
  silently rot back into scattered comments.

## The registry as source of truth

`internal/amendclass` holds the classification as pure data: one `Knob` per
`adjudicator.Policy` field (plus the FROZEN non-field floor elements), each with
its `Class`, `Direction`, authorized `Channels`, and a one-line `Doc`. It imports
nothing from `internal/adjudicator` (keeping it a clean, cycle-free leaf); the
conformance test imports the adjudicator only to reflect over `Policy` and prove
coverage. Every observability verb built on top — `fak manage policy explain`
(#5172), `fak manage policy diff` (#5173), the exit-summary amendment posture
(#5184) — reads the registry, so the model is described in exactly one place.

## Anchor files

- `internal/amendclass/amendclass.go` — the registry + conformance test.
- `internal/adjudicator/decide.go` — the `Policy` struct the registry classifies.
- `cmd/fak/guard_startup.go` — floor assembly + precedence (`loadGuardCapabilityFloor`).
- `cmd/fak/guard_allow.go` / `cmd/fak/guard_deny.go` — the overlay channels.
- `cmd/fak/policy_reload_widen.go` — the widening gate (`diffPolicyWidening`).
- `internal/journal/capgrant.go` — the `CapabilityGrant` provenance record.
