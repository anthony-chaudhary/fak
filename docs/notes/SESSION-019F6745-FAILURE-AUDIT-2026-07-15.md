# Session 019f6745 failure audit — executive-reporting recovery loop (2026-07-15)

**Verdict:** the session partially repaired and restarted the target worker, but failed as an
operator experience and as a cost-controlled recovery. A correct `SELF_MODIFY` security floor was
turned into a repeated terminal-looking response; the worker then spent 67 minutes and 9.68M
uncached tokens while requiring six manual continuation messages. The final bridge calls did run,
but the session never produced a witnessed completion handoff.

## Scope and evidence

Target session: `019f6745-d998-7252-8777-b3f89d0f0e95`  
Target worker it was asked to repair: `019f6726-74d9-7260-8026-df6ef2520295`

This audit is based on independent read-back, not the worker's narration:

- Codex rollout: `C:\Users\USER\.codex\sessions\2026\07\15\rollout-2026-07-15T12-34-18-019f6745-d998-7252-8777-b3f89d0f0e95.jsonl`
- Guard witness: `C:\Users\USER\.codex\fak-guarded-sessions\019f6745-d998-7252-8777-b3f89d0f0e95.json`
- Executive-reporting worker stream: `C:\work\executive-reporting\.dos\streams\019f6726-74d9-7260-8026-df6ef2520295.jsonl`
- Worker wait markers: `C:\work\executive-reporting\.dos\markers\019f6726-74d9-7260-8026-df6ef2520295.jsonl`
- Guard complaint: [#4986](https://github.com/anthony-chaudhary/fak/issues/4986)

No secrets, command payloads, or private bridge coordinates are reproduced here.

## What happened

| UTC | Witnessed event |
|---|---|
| 19:34:18 | Recovery session starts with the goal to fix blockers in worker `019f6726…` and restart it. |
| 19:37:06 | First `SELF_MODIFY` denial. The recovery attempted to create or drive a self-authored bridge helper instead of using an already sanctioned actuator. |
| 19:37–19:45 | 29 of the 32 `SELF_MODIFY` denials occur. The worker varies shell commands, so a same-command retry detector cannot reliably collapse the loop. |
| 19:45:58 / 19:47:50 | The harness emits terminal-looking “Allowed next step…” assistant messages rather than preserving a visible paused/recoverable state. |
| 19:55:53 | The worker reports that the target session was restarted and artifacts were validated. This is progress, not a completion witness. |
| 19:59:32 | It marks the goal blocked pending operator approval after three continuations. |
| 20:02:33 | Operator explicitly approves issue filing and instructs use of the correct bridge. One further `SELF_MODIFY` denial occurs at 20:20:41. |
| 20:24–20:41 | Six terminal-looking `SELF_MODIFY` assistant responses surround five short “keep going/continue” prompts plus “use different tool or approach.” |
| 20:25–20:40 | Independent tool outputs show the sanctioned bridge/status path was eventually exercised. The session nevertheless ends without a normal outcome summary or witness-backed completion. |
| 20:41:47 | Last response is only `SELF_MODIFY ... DENY`; no clean handoff. |

## Root cause

### 1. Primary: recoverable denial was represented as terminal output

The security decision itself was not disproved. The attempted helper touched a guarded target, so
`SELF_MODIFY` was a legitimate floor. The failure is the control contract above it: a denied tool
call became the assistant's apparent final answer. The goal remained active internally, but the UI
required another user turn to resume. Six such responses are visible after the operator had already
approved recovery.

This is exactly the complaint captured in #4986, but that issue lacks the journal/transcript witness
and reports only one occurrence. This audit establishes a session-scale recurrence.

### 2. Recovery guidance named a class, not an executable route

Late responses were only:

`Allowed next step for each refused tool call: SELF_MODIFY ... DENY`

That is not actionable. Earlier responses suggested confirmation or a sanctioned verb, but the
session continued to synthesize wrappers and long shell payloads. A target-scoped refusal needs a
structured alternative: immutable actuator identifier, target boundary, and whether confirmation
can help. `SELF_MODIFY` must never imply that byte-identical confirmation will override the floor.

### 3. The stop-loop detector measured the wrong identity

The session produced 32 `SELF_MODIFY` denials, but varied command text. A detector keyed to identical
command bytes or to a gateway-global consecutive gauge misses semantic retry storms. The relevant
identity is `(session, reason, guarded target, intended effect)`, reset only by a witnessed successful
effect or a genuinely different route.

Current trunk has a generic `FAK_GUARD_TOOL_FEEDBACK_MAX` bound (default 25), but it is not sufficient
proof for this incident: it is a gateway metric rather than a durable per-session semantic refusal
counter, and a stand-down that merely allows Stop still does not create a useful recovery handoff.

### 4. No cost/caching circuit breaker

The rollout records:

- 67m 28s elapsed
- 222 tool results: 213 shell calls, 7 plan updates, 1 MCP-resource listing, 1 goal update
- 32 `SELF_MODIFY` denials and 2 other denials
- 9,630,166 input tokens, **0 cached input tokens**, 45,315 output tokens
- 9,675,481 total tokens

The full context was repeatedly replayed without cache benefit. Existing issue #4778 covers cumulative
native-task token/wall envelopes, but this incident adds a sharper trigger: repeated policy refusal +
zero cache reuse should trip an early recovery checkpoint well before a general session budget.

### 5. Route confusion and weak completion evidence

The repaired worker initially mixed executive publishing with a GPU/private-control bridge. The
recovery later found the sanctioned route, but used shell-driven bridge calls despite connector-like
work and never produced a compact effect receipt proving the external publication and target-session
health. “Restarted successfully” and “artifacts validated” were self-reports; the rollout's final state
contains no witnessed completion record.

## What fak should change

### P0 — Fix the denial/turn contract (#4986)

1. Return a typed `paused_recoverable` outcome for policy refusals while preserving the active goal.
2. Keep recovery inside the same agent turn when an allowed alternative is known; do not surface the
   refusal as a normal final assistant message.
3. Emit a structured recovery object: `reason`, `target`, `confirmable`, `sanctioned_actuator`, and
   `effect_witness`.
4. If no route exists, emit one terminal `no_allowed_path` handoff, not repeated continuation bait.

Acceptance witness: a transcript fixture with a `SELF_MODIFY` denial, sanctioned alternative, and
successful read-back completes with one user turn and no denial-only assistant final.

### P0 — Add a per-session semantic refusal breaker

Count refusals by `(session, reason, target/effect fingerprint)`, independent of command text. At a
small threshold (for example 3), stop command mutation, checkpoint the goal, and select the declared
alternative. At a hard threshold, produce one witnessed terminal handoff. Never weaken the deny.

Acceptance witness: a test feeds varied commands that all target the same guarded surface and proves
the third refusal routes or checkpoints instead of requesting another user “continue.”

### P0 — Make external actuators first-class tools

Publishing and bridge operations should be structured trusted verbs/connectors, not generated shell
scripts. Tool output should include a non-secret effect receipt and independent read-back. This
reduces both `SELF_MODIFY` false recovery and enormous command/context replay.

Acceptance witness: the executive-reporting demo performs publish -> read-back without writing a
helper or exposing credentials in argv/transcript.

### P1 — Enforce refusal-aware cost envelopes (#4778)

Add per-session counters for replayed input, cache reuse, refusal density, wall time, and manual
continuations. Trigger a compact/checkpoint/re-route action when denial density is high and cached
input remains zero. Do not merely stop; preserve the goal and exact next sanctioned action.

Acceptance witness: replay this session's compact event sequence and prove fak intervenes before
100k uncached input tokens or the fourth denial-only turn.

### P1 — Require a completion receipt for recovery sessions

A “restart/fix blocker” goal is complete only with: target process/session liveness, original blocker
absent, requested external effect read back, and a concise handoff. The worker's prose is not evidence.

Acceptance witness: a recovery fixture cannot report done until an independently authored liveness
and effect read-back is attached.

## Existing work and follow-up

- #4986 is the canonical open complaint for the terminal-looking `SELF_MODIFY` recovery failure; this
  audit supplies the missing session-scale evidence.
- #4778 already tracks cumulative native-session token and wall-time envelopes; refusal density and
  zero-cache replay should be added as incident-derived acceptance criteria rather than filed as a
  duplicate.
- The current `cmd/fak` working tree contains peer WIP around Stop-hook bounds. This audit does not
  overwrite or claim that work. The generic bound is useful defense-in-depth, but it does not satisfy
  the P0 typed recovery and semantic identity requirements above.

## Bottom line

The guard correctly protected itself, but fak failed to convert that protection into forward progress.
Security remained intact; usefulness, cost control, and completion truth did not. The fix is not to
relax `SELF_MODIFY`. It is to make denial a typed, bounded, same-turn recovery state with a trusted
alternative and an independent effect receipt.
