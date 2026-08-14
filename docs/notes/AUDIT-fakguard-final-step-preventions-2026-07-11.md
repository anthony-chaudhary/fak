# Audit: fakguard crashes & bad preventions at trajectory final steps (2026-07-11)

**Scope.** The guard surface that acts at the *end* of an autonomous `fak manage -- claude`
trajectory: the Stop-hook completion guard, the result-side livelock/doom-loop detector, the
gateway/policy tool-call refusal path, and the git commit/push gates. "Final step" = git commit,
git push, CI invocation, and the agent's own attempt to *stop*. We looked for two failure classes:
**crashes** (panics / unhandled paths that abort a guard) and **bad preventions** (false-positive
blocks that stop a legitimate final step or abort a healthy trajectory).

**Method.** Code read of the guard sources + empirical fold of the shipped ledgers
(`docs/nightrun/guard-stops.jsonl`, 950 rows; `docs/nightrun/gateway-usage.jsonl`, 3 292 exit
events). Three of the findings reproduced live in the audit session itself.

**Headline.** No guard *crashes* were found — every terminal guard fails **open** (allows the
stop / defers) on its own error, and the gateway log carries **0** panic/error markers across
3 292 exits. The risk is entirely on the **bad-prevention** side, and it is concentrated in four
places, one of them a confirmed false-positive with the fix already half-wired in the tree.

---

## Empirical ground truth (950 stop decisions)

| disposition | count | note |
|---|---:|---|
| `handoff_block` | 441 (46%) | **dominant** final-step block; 41 fired with `stop_hook_active:true` |
| `mode_off` | 129 | guard disabled — clean |
| `clean_completion` | 103 | real stops |
| `same_issue_continue` | 88 | genuine-repeat continues |
| `deny_all_continue` | 42 | |
| `tool_feedback_continue` | 32 | unbounded (F2) |
| `same_issue_give_up` / `blind_give_up` | 24 / 21 | bounded stand-downs — **working as designed** |
| `fail_open_gauge_unavailable` | 21 | gateway unreachable → **correctly allowed the stop** |
| operator_directed_* | 28 | |

- **57 rows blocked (exit 2) while `stop_hook_active:true`** — the harness signalling it is already
  in a stop-hook continuation loop, and the guard blocked anyway.
- **`signal:"same-issue"` on 589 rows but `stage:"allow"` on 572** — the `signal` field is a
  *path label*, not a repeat classification (F5).

---

## Findings

### A1 — Result-side livelock counter false-trips on passive transcript replay  ·  BAD_PREVENTION · **high**
*(the `LIVELOCK_DETECTED … ABORT=terminal` banner on forward-progressing sessions — reproduced live)*

The result-side detector counts an **unchanged, replayed** `tool_result` once per turn. Claude Code
replays the full history every turn; a quarantined result's bytes stay in that history permanently,
so its digest is stable and the admit-ledger keeps returning the recorded `QUARANTINE` verdict:

- `internal/gateway/gateway_admit.go:211` `admitInboundResults` loops every replayed result each turn;
  digest at `:234` is computed from `messages[i].Content` (stable across replays).
- `internal/gateway/result_livelock.go:11` `annotateResultLivelock` feeds that replayed verdict into
  `ObserveFailure` once per turn, **without** checking whether this exact held result was already seen.
- `internal/guardrsi/livelock.go:177` `observe` sees the same key each turn → `count++` monotonically;
  `sawObservation` stays true so `Clear` never fires. 3 = advisory, 6 = fuse, 9+ = escalate →
  `ABORT=terminal` banner (`internal/gateway/messages_resultnotes.go:135`).

So **one old paged-out result drives the counter up by one per turn regardless of what the agent
does** — a loop the agent never caused.

**The fix is declared but never wired.** `internal/gateway/gateway.go:1300-1317` declares
`resultLivelockObserved` (the intended "replay gate" — observe each held result at most once, keyed by
a stable per-result key; a genuinely re-issued call gets a *new* `tool_call_id` and still climbs) and
`resultLivelockRecorded` (dedups durable side-effects). **Both fields are never read or written
anywhere** — `annotateResultLivelock` consults neither. Consequences: the false trip is live; the
durable record `journal.AppendLivelock` (`internal/journal/livelock.go:59`) is never called; the
cross-trace `emitFleetObservation` (`internal/gateway/fleet_obs.go:35`) is never called.

Worse, `TestResultAdmissionLivelockSurfacesOnReplay`
(`internal/gateway/messages_result_note_test.go:185`) **pins the buggy behaviour** — it replays the
same admission three times and asserts `repeat=3` trips. The suite is green *with* the bug.

**Scope-limit (why "high" not "critical"):** the result-side trip is **advisory** — it injects the
misleading "you are looping, ABORT=terminal, stop re-reading" instruction but does not itself convert
anything to DENY and does not feed the session-ending same-issue gauge. Real enforcement lives on the
proposed-call side (`internal/gateway/adjudicate_proposed.go:247`), which correctly keys on genuinely
re-issued calls. The harm is a false abort instruction inside a healthy trajectory (a model may obey it
and drop good work), plus the missing durable/fleet records for *real* loops.

**Fix:** wire `annotateResultLivelock` to consult/populate `resultLivelockObserved` (observe each held
result at most once) and dedup durable side-effects via `resultLivelockRecorded` + call
`AppendLivelock`/`emitFleetObservation` on a real trip. Then change
`TestResultAdmissionLivelockSurfacesOnReplay` to replay with a *new* `tool_call_id` per turn.

### A2 — Task-handoff gate has no give-up bound and ignores `stop_hook_active`  ·  BAD_PREVENTION · **medium-high**

The deny-all ladder is carefully bounded (stands down at `max`, then allows the stop so a stuck model
cannot loop). The task-handoff gate is not: `runGuardTaskHandoffGate`
(`cmd/fak/guard_stophook.go:578-621`) returns exit 2 on **every** stop where the handoff artifact is
missing/invalid, with no ladder and **no check of `stop_hook_active`** (the field is parsed at
`:360-362` and only used for shadow-log text). Empirically this is the dominant block — **441/950
(46%)** `handoff_block`, **41 of them with `stop_hook_active:true`** (turns climbing to 91+). A run that
genuinely cannot produce a valid handoff (misconfigured file path, a `taskmgr.ReviewHandoff` edge) is
blocked at *every* stop attempt. Opt-in (`FAK_GUARD_TASK_HANDOFF_MODE` defaults off), so it only bites
harnesses that enable it — but the missing bound is a real robustness gap relative to the deny-all path.

**Fix:** give the handoff gate a bounded give-up rung and/or honour `stop_hook_active` (allow the stop
once the harness signals an active continuation loop), mirroring the deny-all ladder's stand-down.

### A3 — gitgate `core.hooksPath` detection is a substring match, not key-scoped  ·  BAD_PREVENTION · **medium**

`internal/gitgate/gitgate.go:653` sets `hasHooksPath = true` when **any** token
`strings.Contains(lt, "core.hookspath")`, not when `core.hooksPath` is the config **key**. Because
`tokenizeSegments` (`:919-925`) fuses a quoted value into one token, a legitimate write of a *different*
key whose value merely mentions the string is refused with `POLICY_BLOCK`:

```
git config alias.st "status; note core.hooksPath"
git config core.editor "vim ; echo core.hooksPath"
git config --add safe.directory /repo/core.hooksPathBackup
```

The `-c`/long-global form has the same substring flaw (`:558-565`, **A3b, low**). A read/`--unset` is
correctly exempted; a write of a different key is not.

**Fix:** set `hasHooksPath` only when `core.hooksPath` is the key operand (mirror the precise
`splitConfigKey`/`isGitFalse` handling already used for `commit.gpgsign` at `:657-665`).

### A4 — a repeated structural refusal is amplified into a session-ending `TERMINAL`  ·  BAD_PREVENTION · **medium**

A gitgate Deny carries an **empty** disposition (`internal/gitgate/gitgate.go:266`). When the same call
is re-proposed past the abort count, `internal/gateway/adjudicate_proposed.go:318-347` stamps
`Disposition = "TERMINAL"` / `By = "livelock-abort"`. This is the `POLICY_BLOCK/TERMINAL` disposition
seen in the wild. It is *correct* for a genuine loop — but if the underlying refusal is an A3 false
positive the agent **cannot** satisfy, the escalation converts a wrong refusal into a session stop.
Also (**A4b, low-med**): a lone `POLICY_BLOCK` with empty disposition is non-retryable
(`toolRejectionIsRetryableFeedback`, `:376-382`) → immediate `adjudicationOutcomeDenyAll` (`:373`),
heavier than retryable feedback. Both are subsumed by fixing A3 at the source.

### A5 — BARE_COMMIT_SWEEP cannot tell its own recommended safe form from a bare sweep  ·  BAD_PREVENTION · **low** (default) / **medium** (`FLEET_BARE_COMMIT_GUARD=block`)
*(reproduced live — an explicit-pathspec `git commit -- <paths>` still drew the advisory)*

`gateBareCommitSweep` (`internal/hooks/gate_barecommitsweep.go:56-84`) inspects `d.StagedPaths` — the
**entire staged index** — and keys only on the `FAK_SAFECOMMIT_VETTED` marker. A pre-commit hook never
receives the commit's pathspec, so the gate **cannot distinguish** a safe `git commit -- <yours>` (which
lands only the named paths) from a bare `git commit` (which sweeps the whole index). Yet the advisory's
own remedy text (`:110`) recommends `git commit -- <yours>` as the safe alternative — and that exact form
is unvetted, so it re-draws the same advisory, listing paths that will *not* be committed. A
self-referential loop that erodes trust in the advisory. Advisory-only by default (exit 0, safe); a
**false block** under `FLEET_BARE_COMMIT_GUARD=block`.

**Fix:** in the advisory text, drop `git commit -- <yours>` as an "equivalent" and point only at
`fak commit --path` (the sole form that sets the vetted marker); or, longer-term, have safecommit stamp
the vetted marker for any explicit-pathspec commit it can attribute.

### A6 — unbounded tool-feedback continue  ·  BAD_PREVENTION · **low-medium**

`cmd/fak/guard_stophook.go:419-430`: when `consecutive<=0 && feedbackConsecutive>0` the hook returns
exit 2 to let the model repair a malformed call — with **no max and no ladder**. A model stuck emitting
malformed tool calls is continued every turn (until the harness's own `stop_hook_active` cutoff).
Reasonable to continue *once*; unbounded continue is the same missing-bound gap as A2. **32** rows in the
ledger. **Fix:** apply a bound analogous to the deny-all ladder.

### A7 — `signal` telemetry is a path label, not a repeat classification  ·  OBSERVABILITY · **low**

`cmd/fak/guard_stophook.go:435,452-456`: `rec.Signal="same-issue"` whenever the gateway *emitted* the
same-issue gauge at all — not based on which signal drove the decision. On a current gateway every row —
including `clean_completion` (`depth=0`) — is stamped `same-issue` (589/950), while 572 rows are
`stage:"allow"`. An operator folding the ledger would read 589 "same-issue repeats" that never happened.
`depth` disambiguates, so this is labelling, not behaviour. **Fix:** gate the label on the driving path
(`useSame && depth>0`), else `"blind"`/`"clean"`.

---

## What is sound (verified, no action)

- **No crashes.** Every terminal guard fails **open**: the Stop hook returns 0 on bad args/mode/no-URL/
  unreachable-gauge (`guard_stophook.go:349,389,403,410`); the safecommit lock layer reaps dead-PID
  locks and is best-effort throughout (`internal/safecommit/runner.go`); `gateBareCommitSweep` never
  returns an error; the policy `deny_regex` compiles at **load** with returned errors (no `MustCompile`
  in the hot path, RE2 = no catastrophic backtracking) so a bad rule fails the manifest, not the
  gateway. Gateway log: **0** panic/error markers / 3 292 exits.
- **Refusal attribution is NOT broken.** The "Bash `core.hooksPath` refusal right after a Grep" specimen
  is **not** a kernel misattribution: every note entry's label is the refused call's own
  `tc.Function.Name`, built only from the current turn's `adjs`
  (`adjudicate_proposed.go:438`, `messages_stream_passthrough.go:130-133`,
  `refusal_notes.go:102`), with no cross-turn/cross-call carryover and no stored replay; and gitgate
  cannot classify a Grep (no shell command → `Adjudicate` defers, `gitgate.go:262`). The specimen is a
  co-proposed Bash call in a multi-tool turn (Grep allowed, Bash refused), not a phantom.
- **The give-up ladders are correctly bounded** — 45 give-up rows stood a stuck session down cleanly;
  21 gauge-unavailable stops correctly failed open.

---

## Priority

1. **A1** — wire the declared result-side livelock replay gate (kills the false `ABORT=terminal`
   on healthy trajectories; restores the missing durable/fleet records). Highest value.
2. **A3** — key-scope the gitgate `core.hooksPath`/`-c` detection (closes the only demonstrated
   false `POLICY_BLOCK`; A4 amplification then can't bite a false positive).
3. **A2 / A6** — bound the handoff and tool-feedback continue paths (and honour `stop_hook_active`).
4. **A5 / A7** — advisory-text + telemetry-label clarity.

## What is deliberately NOT done here

This is an investigation only — **no code changed, nothing committed**. Each finding names a precise
fix location; A1's fix additionally requires updating the test that currently pins the bug.
