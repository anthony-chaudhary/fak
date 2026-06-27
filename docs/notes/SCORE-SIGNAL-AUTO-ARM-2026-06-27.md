# score-signal: arming autonomous live filing (#980) — posture decision (2026-06-27)

Operational/decision note for [#980](https://github.com/anthony-chaudhary/fak/issues/980).
Records the arming posture chosen for the `score-signal` feeder
(`tools/score_signal.py` + `.github/workflows/score-signal.yml`) and the safety
contract that makes it sound.

## The decision

`score-signal` shipped **report-by-default**: the daily scheduled run only planned
the issues to the step summary; filing required an explicit `workflow_dispatch` with
`live=true`. #980 is the operator decision to let CI/CD create the tickets without a
human.

**Chosen posture: full flip to `--live` on the schedule.** The daily scheduled run
now calls `score_signal.py … --live` directly, filing autonomously for every
actionable rise (blocking *and* advisory). A manual `workflow_dispatch` with
`live=true` is kept for an off-cadence arm.

This was a deliberate operator call to take the simplest fully-autonomous posture,
accepting the dedup + cap discipline as the storm bound rather than adding a
RED-only guard. The two posture alternatives considered and not taken:

- **Guarded auto-arm (RED only)** — file on the schedule only when the portfolio
  ratchet is RED. Safer, but leaves the advisory early-warning case (#712) still
  needing a human, which is most of the value of the feeder.
- **Dry-run + notify** — post the plan for a one-click manual arm. Least change, but
  does not deliver "CI files the ticket without a human" at all.

## Why a blind flip is safe here — the storm bound is structural, not a human

The anti-spam guarantee does **not** depend on a human reviewing each run. It is the
feeder's own discipline, proven by `tools/score_signal_test.py`:

1. **Label-scoped marker dedup** — at most one OPEN `score-signal` issue per
   scorecard key (`<!-- fak-score-signal: <key> -->`), and the dedup index is fetched
   scoped to the tool's own label, so the bound is exact regardless of total backlog
   size. A daily run on a *steady* regression files the issue **once**; subsequent
   days find the open marker and file nothing.
2. **Refresh-not-re-file on worsening (#981)** — when a tracked regression worsens
   past `--worsen-delta`, the open ticket is *refreshed in place* (a comment + a title
   bump), not duplicated. This is the load-bearing mechanism under autonomous filing:
   without it, a drifting metric would tempt a re-file every day. With it, the daily
   live run touches an open ticket at most once, only when it materially worsened.
3. **Worst-first hard cap** — `--max-issues` (default 5) bounds a multi-metric burst,
   filing the worst regressions first.

The net behavior under the flip: on any given day, the schedule files **at most
`max_issues` brand-new issues** (one per never-yet-tracked regressed metric) plus
**at most one refresh per already-tracked metric that worsened**, and **nothing** for
a steady or improving tree. That is a bounded, non-duplicating, self-quieting feed —
the property that makes report-by-default unnecessary.

## Soak evidence

The "soak" here is the period the feeder ran report-by-default (dry-run plans to the
step summary) before this flip, plus the hermetic proof that the plan/dedup core is
correct:

- The pure relevance-filter → dedup → cap → refresh logic is covered by 26 hermetic
  tests in `tools/score_signal_test.py` (run first in the workflow, so a filed issue
  is only as trustworthy as a green core).
- A local dry-run against a synthetic regressed pane plans exactly the expected scoped
  issue and mutates nothing; the dedup path is open-issue-scoped so a re-run with the
  same open marker plans **nothing** (no duplicate).

If the live feed ever misbehaves (a false or duplicate filing slips through), the
revert is a one-line change to the workflow's file step `if:` condition back to
`workflow_dispatch && live == 'true'` — no code change.

## What changed

- `.github/workflows/score-signal.yml`: the file step now fires on
  `github.event_name == 'schedule'` **or** an armed `workflow_dispatch`; the header
  and step-summary mode line document the autonomous posture and the storm bound.
- No change to `tools/score_signal.py` itself — the dedup + cap + refresh discipline
  that bounds the feed was already in place (the feeder + #981).
