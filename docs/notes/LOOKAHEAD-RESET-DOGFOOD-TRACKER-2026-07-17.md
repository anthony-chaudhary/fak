---
title: "Look-ahead reset dogfood tracker: P0 -> P1 -> P2 runbook + witnessed tracking"
description: >
  Dogfood tracker for the look-ahead reset (#5202 family). Sequences the three
  protocols cheapest-first — P0 PreCompact async mini-lookahead (#5207), P1
  doomloop OBSERVE fork-confirm (#5208), P2 full drift-triggered reset — with
  their witnessed done-conditions, owns the "one long real session" evidence
  capture, and tracks the 4-arm paired-fork A/B (#5210). Every status row is
  witnessed from ledger rows + transcripts + git evidence, never self-report.
date: 2026-07-17
family_key: fak-lookahead-reset-key
prior_art: ["#5202", "#5207", "#5208", "#5210"]
---

# Look-ahead reset dogfood tracker (P0 -> P1 -> P2)

Status: **tracking note under epic [#5202](https://github.com/anthony-chaudhary/fak/issues/5202)**;
runbook + witnessed tracking for issue
[#5220](https://github.com/anthony-chaudhary/fak/issues/5220). Nothing here changes
runtime. Theory, dominance proof, seam map, and the full protocol definitions live in
[the look-ahead reset concept note](CONCEPT-LOOKAHEAD-RESET-2026-07-17.md); this note is
the *operational* side — the order to run the protocols in, the exact done-condition each
must witness, and the ledger of what has actually been witnessed so far.

The tracking rule is the repo's standing one: a row moves from `not yet` only on
**witnessed evidence** (ledger rows, transcript spans, commit SHAs confirmable via
`dos verify` / commit-audit) — never on a session's self-report that it worked.

## Sequencing: why this order

Cheapest-first, and each stage de-risks the next:

1. **P0** ([#5207](https://github.com/anthony-chaudhary/fak/issues/5207)) is async and
   nearly free — fire-and-forget at a PreCompact boundary that is already quiescent, so it
   needs no paired checkpoint seam. It proves the rollout-in-fork runner + witnessed
   distillation + lesson injection end to end.
2. **P1** ([#5208](https://github.com/anthony-chaudhary/fak/issues/5208)) adds the
   fork-confirm discipline to a live classifier (doomloop OBSERVE), converting a
   heuristic regime call into a witnessed one — and measures the heuristic's
   false-positive rate as a side effect.
3. **P2** (full drift-triggered look-ahead reset; needs the paired atomic checkpoint
   seam) is the full operator: fork, roll, witness, distill, reset-with-lesson at
   mild/strong strengths. It only runs once P0/P1 have witnessed the shared machinery.

The 4-arm A/B ([#5210](https://github.com/anthony-chaudhary/fak/issues/5210)) is the
measurement harness across all three — it licenses the dominance claim on real sessions
rather than the proof's assumptions.

## Runbook

### P0 — PreCompact async mini-lookahead ([#5207](https://github.com/anthony-chaudhary/fak/issues/5207))

Run: a normal long `fak manage -- claude` working session; no operator action at the
boundary. On a PreCompact *allow* decision the hook (once #5207 lands) gates via
`AdmitSpeculation`, spawns the detached rollout (transcript fork + worktree prepare +
diff patch + `--max-turns 3` resume under a deny-push floor), and exits 0 immediately.
The runner witnesses at exit (build/affected-tests -> W3; ActivityDivergence -> W2),
distills a `Lesson`, and writes the per-session lesson file;
`runGuardSessionStart` `source=compact` injects a fresh same-base-SHA lesson as
additionalContext beside the compacted summary. Fail-open on both sides.

Evidence to capture per compaction boundary: the ledger boundary row, the rollout's
spawn/exit rows, the lesson file (rung + evidence pointers), and the SessionStart
transcript span showing the injected additionalContext.

**Witnessed done-condition:** one long real session with >=3 auto-compactions shows
>=1 rollout that produced a W2+ lesson which was injected at SessionStart(compact) —
verified from ledger rows + transcript, not self-report.

### P1 — doomloop OBSERVE fork-confirm ([#5208](https://github.com/anthony-chaudhary/fak/issues/5208))

Run: sessions long enough for the doomloop classifier to reach
`Correction == CorrectObserve` with `BurningFlatStreak == TripWindows-1`. At that edge
(once #5208 lands) the shell forks + rolls `k` turns and feeds the fork's
effort/progress counters back through the pure `doomloop.Classify` (classifier
untouched). Fork verdict DOOM -> the nudge is enqueued now, lesson-enriched; fork
verdict HEALTHY (W3 progress in the fork) -> a suppressed-nudge ledger row with closed
reason `SPEC_RECOVERED` (declared in dos.toml). `calibrate.go` scores
heuristic-vs-simulation disagreement.

Evidence to capture per confirm rollout: the OBSERVE-stage ledger row with fork verdict
+ rung, and (for suppressed nudges) the `SPEC_RECOVERED` row.

**Witnessed done-condition:** the ledger has >=K OBSERVE-stage confirm rollouts with
fork verdict + rung, and calibrate reports the measured false-positive rate of the
K-window heuristic.

### P2 — full drift-triggered look-ahead reset

Run: a real long-horizon session where trajctl fires `Signal == DRIFT|STALL` at a
Stop-hook turn end. The full cycle (once the paired-checkpoint seam and P2 wiring land):
paired atomic checkpoint -> k=5-8 headless rollout in the fork -> witness (CheapScorers
+ build/test + shadowgit) -> witness-gated distill -> render at mild (lesson via steer,
history stays) or strong (true reset: relaunch from the pre-drift fork with the lesson
spliced) strength. Every event ledgers as `KindLookahead`.

Evidence to capture: the DRIFT signal row, every `KindLookahead` ledger row for the
cycle, the pre-drift/post-reset turn indices, and the post-reset W3 landing commit SHA.

**Witnessed done-condition:** one real long-horizon session where a DRIFT fired, the
full cycle ran, and the post-reset curve reaches its next W3 landing (witnessed commit,
`dos_verify`-confirmable) in fewer turns than the pre-drift drift span.

### 4-arm A/B ([#5210](https://github.com/anthony-chaudhary/fak/issues/5210))

At a real branch point, serial re-runs from the identical paired checkpoint — arms
{reset+lesson, rollback-alone, let-drift (as a fork, never the live session),
nudge-alone}, each in its own detached worktree at the same base SHA with the identical
diff patch, k-turn budget, deny-push floor, reaped at end (no arm lands). Witnessed
scores: turns-to-first-W3, curve slope, binary goal-attained; SUTVA covariates (base
SHA, peer commits since) recorded per arm. Fold on the `FoldWorktreeAB` pattern
extended to a 4-arm `lookahead-ab` fold.

**Witnessed done-condition:** one reproducible 4-arm fold artifact from a real branch
point with all four arms scored on witnessed metrics and covariates recorded, evidence
captured pre-Reap.

## Witnessed tracking

Rows move only on witnessed evidence. `not yet` is the honest resting state — see the
workspace rule that an open follow-on is `not yet`, not a failure and not a ship.

| Stage | Issue | Implementation | Dogfood run | Witness recorded |
|---|---|---|---|---|
| P0 PreCompact mini-lookahead | [#5207](https://github.com/anthony-chaudhary/fak/issues/5207) | not yet | not yet | — |
| P1 OBSERVE fork-confirm | [#5208](https://github.com/anthony-chaudhary/fak/issues/5208) | not yet | not yet | — |
| P2 full drift reset | #5202 (needs paired-checkpoint seam) | not yet | not yet | — |
| 4-arm A/B fold | [#5210](https://github.com/anthony-chaudhary/fak/issues/5210) | not yet | not yet | — |
| One-long-real-session evidence capture | this note ([#5220](https://github.com/anthony-chaudhary/fak/issues/5220)) | runbook shipped (this note) | not yet | — |

To update a row: append the witness under "Witness log" below (ledger path + row ids,
transcript span, commit SHA), then flip the cell. A cell without a matching Witness-log
entry is a bug in this note.

## Witness log

*(empty — no protocol has produced witnessed evidence yet; P0/P1 implementations are
open issues as of 2026-07-17.)*

## Fences

- This tracker inherits every fence in the
  [concept note](CONCEPT-LOOKAHEAD-RESET-2026-07-17.md#fences-verify-before-building) —
  in particular: the transcript fork transport and `--max-turns` support are unverified;
  fork-worktree SHAs dangle after Reap, so witnesses must be persisted pre-Reap.
- Per-instance *benefit* remains unprovable; the per-episode claim is "took
  (acceptance predicate witnessed)". The population license is #5202's e-process/SPIBB
  machinery, with the 4-arm A/B as the evaluator.
