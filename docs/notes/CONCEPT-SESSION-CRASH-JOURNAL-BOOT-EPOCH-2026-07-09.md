---
title: "The crash-survivable session-registration journal — a boot-epoch recovery spine (2026-07-09)"
description: "A lightweight, durable, machine-global sidecar that records session registration + lifecycle to disk so a system-wide infra crash (a Windows update reboot, a WindowsTerminal 0xc0000005 that kills every terminal at once) can be recovered from: the fleet re-enumerated and each session resumed. The keystone is one primitive fak has never had — the machine boot epoch — which turns today's best-effort transcript-scan discovery into a definitive `started-before-the-current-boot ∧ not-cleanly-closed ⇒ died in the reboot` verdict. Maps the design against the shipped pieces (guard_sessions.jsonl, session-registry.json, resume_sweep, CrashInterrupted) and lays out the epic + DoD child tickets. Foundation slice (boot-epoch primitive + Classify fold + `fak sessionjournal report`) ships with this note."
---

# The crash-survivable session-registration journal (boot-epoch recovery spine)

> Date: 2026-07-09. Status: design note + shipped foundation slice (C1). Companions:
> [RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY](RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md)
> (this is its U1/U6 rung at the *registration/liveness* altitude, not the working-state altitude),
> [RESUME-REHOME-RUNBOOK](RESUME-REHOME-RUNBOOK-2026-06-26.md),
> [VERIFIED-RESUME-PACKET-CHECKPOINT (#636)](VERIFIED-RESUME-PACKET-CHECKPOINT-2026-06-26.md).
> Prior art it composes with (does not duplicate): epic **#1193** (guard-lifecycle),
> **#1197** (`session-registry.json` C1 registry), **#3461** (`guard_sessions.jsonl`
> discoverability index), **#2418** (`sessionledger` world-state witness at resume).

## 0. The trigger

On 2026-07-09 an in-place `WinAppRuntime.Main.1.8` update crashed WindowsTerminal
(`0xc0000005` in `Microsoft.Terminal.Control.dll`, ~1s after the update), taking every
terminal — and every `fak manage` / `fak c` session running inside them — down at one
instant. This is the *machine-wide* failure class: not one session hitting a rate limit,
but the whole fleet dying together. A Windows-update reboot is the same shape.

The operator's ask: a **lightweight global journal for session registration** — a sidecar
for monitoring that, after such an event, lets the fleet be re-enumerated and resumed.

## 1. What already survives a crash — and why it is not enough

The recovery pipeline today ([RESUME-REHOME-RUNBOOK](RESUME-REHOME-RUNBOOK-2026-06-26.md))
is **best-effort reconstruction**, not a durable ledger read:

| Piece | What it is | Why it falls short of a machine-wide crash |
|---|---|---|
| `resume_sweep` / `internal/resume/sweep` | discovers sessions by scanning `~/.claude*/projects/*/<sid>.jsonl` transcripts | reboot-surviving, but cwd is recovered from the **irreversible** project slug (`re.sub([^A-Za-z0-9],"-")`) — a repo outside the `<workspace-root>\*`/`<windows-users-root>\*` globs lands the resume in the wrong tree |
| live-vs-crashed test | scan the live process table (`Win32_Process`) for the sid | **erased by a reboot** — after a machine-wide crash you cannot tell which sessions *were* alive |
| `guard_sessions.jsonl` (`internal/guardsessions`, #3461) | append-only index: `{handle, trace_id, agent, pid, cwd, started_utc, nonce}` | **register-only, latest-per-handle snapshot** — no lifecycle, no clean-close, no liveness fold; a session that crashed and one that ran for hours look identical |
| `session-registry.json` (`internal/session`, #1197) | durable descriptor registry with `PID/host/cwd/StartSHA` | **opt-in** (a plain `fak manage -- claude` writes no row; gated on `guardDurabilityWanted`), **per-user-config**, and its only staleness signal is `LastSeen` vs a TTL — never reconciled against a reboot |
| `internal/resume/scan.go` `CrashInterrupted` | per-transcript "killed mid-turn" verdict | the exact fingerprint a reboot stamps on every live transcript at once — but per-session, with **no boot correlation** and no "the whole fleet died together" fan-out |

The common gap has one name: **there is no machine boot epoch anywhere in fak.** A grep for
`boot_id|LastBootUpTime|GetTickCount|kernel uptime|unclean-shutdown` returns nothing. Every
liveness answer is derived from the volatile process table, which a reboot wipes.

## 2. The keystone primitive: the boot epoch

The machine's last boot instant is the one durable fact that distinguishes a clean restart
from a machine-wide wipe. With it, crash recovery becomes a comparison, not a scan:

```
for each recorded session S:
    if S has a clean close event                      -> CLOSED   (ignore)
    else if S.started_at  <  current_boot_time        -> CRASHED  (MACHINE_REBOOT) -> resume it
    else if S.pid is not alive (when checkable)       -> CRASHED  (PID_DEAD)
    else if S.last_seen is stale                       -> STALE    (ambiguous)
    else                                               -> LIVE
```

`S.started_at < current_boot_time` is exact and needs nothing new on the write side — every
existing `guard_sessions.jsonl` row already carries `started_utc`. A session that started
before the machine's current boot **cannot** still be running; if it was not cleanly closed,
it died in the reboot. That single line is the whole machine-wide-crash detector, and it
works over the journal fak *already writes today*.

**Acquiring the boot time, dependency-free** (the module is stdlib-only, no `x/sys`):
- Windows: `GetTickCount64` via `syscall.NewLazyDLL("kernel32.dll")` (the idiom
  `internal/procguard` / `dispatch_tick_os_windows.go` already use); `boot = now - uptime`.
- Linux: `/proc/stat` `btime` (exact boot epoch in seconds).
- Elsewhere: degrade to "unknown" — the fold then skips `MACHINE_REBOOT` and falls back to
  PID / stale-beat, never a false crash verdict.

A `BootID` (the boot time bucketed to 60 s to absorb NTP jitter) gives the lifecycle journal
a stable per-boot equality key; the 60 s bucket is safe because a genuine reboot moves the
boot instant by the prior uptime (minutes–hours), and the sub-bucket edge case is backstopped
by the PID check. Exactness under laptop-sleep/NTP drift is a C6 hardening item (correlate
against WMI `LastBootUpTime`), fenced in §5.

## 3. The shape: a lightweight lifecycle journal that composes with what exists

Not a new database — an append-only JSONL **event** log (the repo's idiomatic local DB;
there is no sqlite anywhere) with three lifecycle kinds and a boot stamp, folded on read:

- `open`  — registration at start: `{id, boot, pid, host, cwd, model, account, argv, start_sha, gateway, ts}`
- `beat`  — heartbeat: `{id, pid, ts}` (bounds crash loss to one heartbeat interval)
- `close` — clean deregister at graceful exit: `{id, reason, ts}`

The reader folds events per id to the latest lifecycle state, joins in any legacy
`guard_sessions.jsonl` rows as `open`-only records, applies §2's classifier, and emits the
**CRASHED set** — each with the real `cwd`/`model`/`account` needed to relaunch, sourced from
the record instead of slug-recovered. This is the durable, authoritative input the resume
pipeline reconstructs best-effort today.

Durability follows the shipped conventions: `O_APPEND` single-line writes (atomic at line
granularity, as `guardsessions.Record` already relies on), tolerant fold-on-read (skip torn
lines — a monitoring/recovery journal favors availability over strict integrity), state path
via the `env → os.UserConfigDir()/fak → .fak` idiom. Cross-process `flock` hardening (the
`internal/usagelog` pattern) is a C6 item, not a foundation blocker.

## 4. Epic + DoD child tickets

**Epic — Crash-survivable session-registration journal (boot-epoch recovery spine).**
Child of / composes with #1193; extends #1197 and #3461 with the one thing they lack — a
durable liveness fold that survives a machine-wide reboot and emits the resume set.

- **C1 — boot-epoch primitive + `Classify` fold + `fak sessionjournal report`.** *(this note ships it.)*
  DoD: `BootTime`/`BootID` on Windows (`GetTickCount64`) + Linux (`/proc/stat btime`),
  degrading to "unknown" elsewhere; a pure `Classify` with unit tests covering
  LIVE / CRASHED(MACHINE_REBOOT) / CRASHED(PID_DEAD) / STALE / CLOSED and the reopen edge;
  a `report` verb that folds `guard_sessions.jsonl` + the lifecycle journal into a classified
  table + `--json`, surfacing the CRASHED (resume-candidate) set with each cwd; verified
  end-to-end (`open` a session, `report` shows LIVE; inject a pre-boot start, `report` shows
  CRASHED). No hot-path edits.
- **C2 — lifecycle events (heartbeat + clean close) with boot stamps.** DoD: `open/beat/close`
  write boot-stamped events; a heartbeat cadence bounds crash-loss to one interval; graceful
  exit writes `close`; the fold distinguishes clean CLOSED from crashed. Grade the append cost
  (must be O(1) per event).
- **C3 — unconditional register-on-start via the SessionStart hook.** DoD: every `fak manage` /
  `fak c` start appends a boot-stamped `open` — *not* opt-in like `session-registry.json`; the
  currently-inert `guard-sessionstart` hook (persists nothing today) writes the row so a plain
  interactive `fak manage -- claude` is finally recorded. Flag-gated first.
- **C4 — recovery-on-restart feeds the resume pipeline.** DoD: on SessionStart after a detected
  reboot, `report` hands the CRASHED set to `resume_sweep` / `fleet_resume_watchdog` as a
  durable input; witnessed that a resume launches from the journal's recorded cwd, not the
  slug-recovered cwd (closes the wrong-tree failure in §1).
- **C5 — name the cause: WER Event-1000 correlation.** DoD: correlate the boot-epoch crash set
  with a WER/Event-1000 (`0xc0000005`) ingest — the native-crash class `toolproc console-faults`
  deliberately skips (it reads .NET Event 1026 only) — so the report names *why* the fleet died
  (WinAppRuntime update, WT crash) beside *which* sessions. Dedupe with #3668 / console-faults.
- **C6 — cross-process + multi-user hardening.** DoD: `flock`-serialized append (usagelog
  pattern); a truly host-global path (not per-user-config); share-delete/compaction; a persisted
  boot marker correlated against WMI `LastBootUpTime` for exactness under sleep/NTP drift.
- **C7 — unify the three registries.** DoD: one journal is the source of truth; the
  discoverability index (#3461) and durable descriptor registry (#1197) read its folded view;
  retire the parallel TTL/process-scan liveness inference.

## 5. Relationship to agent-to-agent comms (and where it deliberately isn't)

The journal is adjacent substrate to `internal/a2achan` (the in-kernel, capability-floored,
Ref-backed agent-to-agent mailbox), **not** the same layer — and the layering enforces that.

**Where it genuinely relates:**

- **Shared identity namespace.** `a2achan`'s Session locale addresses a mailbox by the peer's
  `ToolCall.TraceID` — "the session identity already on every call." The journal's `Event.ID`
  is that same session/trace id (its join + fold key). So the journal is, for free, the
  liveness/registration index for exactly the traces A2A addresses; the join is a shared key,
  no new plumbing.
- **The journal is the liveness oracle A2A lacks.** `a2achan` can only drop a message to an
  abandoned mailbox once it hits a blunt length cap (the per-channel bus bound, #3480) —
  because it *cannot distinguish a dead recipient from a slow one*. The journal's
  CRASHED(MACHINE_REBOOT) verdict is precisely that missing signal: a read-only oracle an A2A
  layer (or the fleet resume watchdog) can consult to GC dead-trace mailboxes deliberately
  instead of leaking-then-capping.
- **Same "cite evidence, not self-report" DNA.** The journal, the `dos_status` fail-closed
  digest, and `a2achan`'s correction protocol (an orchestrator corrects a worker only by citing
  its live status row) are one family of peer-readable *evidence* surfaces. The journal is the
  **liveness** member; `dos_status` is the **progress** member. A fleet supervisor wants both.
- **The crash-path twin of the graceful handoff.** `a2achan`'s Window locale is the *graceful*
  continuity path (a summarizing window SENDs a handoff before teardown, the resuming window
  RECVs it). The journal is the *ungraceful* twin: when the box died with no teardown to send a
  handoff, the journal is what remains to reconstruct who existed. Graceful handoff and crash
  reconstruction are the two halves of continuity.

**Where it deliberately does not relate (the fences that keep it lightweight):**

- **Not a transport.** `a2achan` delivers adjudicated *values* under a taint/scope capability
  floor; the journal records *lifecycle facts* (open/beat/close, boot-stamped). Agent payloads
  never enter the journal — that would break its lightweight property *and* smuggle data around
  `a2achan`'s floor. The journal needs no capability floor because it carries only
  infrastructure facts (PID, boot, cwd, argv), not agent-authored content.
- **Not coordination, not correction.** Lane admission stays in `dos_arbitrate`/`leaseref`;
  worker correction stays in `a2achan`'s status/correction protocol. Presence in the journal
  grants no lane and carries no message.
- **One-way dependency, structurally enforced.** A2A layers *consume* the journal read-only; the
  journal never imports A2A. `internal/sessionjournal` is a stdlib-only foundation leaf (it
  imports nothing internal); `a2achan` is a higher mechanism. architest's no-upward-imports gate
  makes "journal is substrate, A2A rides on top" a compile-time fact, not a convention.

Net: the journal does not extend, replace, or depend on A2A messaging; it supplies the durable
liveness floor an A2A supervisor needs and that `a2achan`'s in-process, dies-with-the-process
mailboxes cannot provide themselves. The relationship surfaces concretely at the **C1 `report`
view** (already a peer-readable, gateway-independent liveness oracle) and matures at **C7**
(#3791), where the unified journal becomes the single source that view reads. An `a2achan`
dead-trace-GC hook that consults the oracle is a plausible future *consumer*, explicitly out of
scope for this epic.

## 6. Honest fences

- C1 ships **detection, not enforcement**: `report` classifies and surfaces the resume set; it
  does not launch resumes (that is C4, deferred to the vetted `fleet_resume_watchdog` path).
- The boot time from `GetTickCount64` is `now - uptime`; a long laptop **sleep** plus an NTP
  step can drift it past the 60 s bucket and falsely re-mint the boot id. The 60 s bucket +
  the PID backstop keep this from producing a false `MACHINE_REBOOT` in the common case, but
  the exact fix (WMI `LastBootUpTime` + a persisted marker) is C6, not C1.
- `started_at < boot_time` is sound only when clocks are monotonic across the boot; a large
  backward clock correction at boot is the known adversarial case — C6's persisted marker
  removes the reliance on wall-clock subtraction.
- This journal records *registration and liveness*, **not working state**. It answers "which
  sessions existed and were alive when the box died, and how to relaunch each" — the resumed
  session still re-establishes its own context (the U2 carryover / #636 packet concern).
- Append favors availability: a torn line from a crash-at-write is skipped on read, not
  refused. A tamper-evident hash chain (the `internal/journal` pattern) is available if a
  future rung needs it, but a recovery journal does not.
