---
title: "Triage: durable kanban board / auto-block-on-spin (#2928) — what already shipped, what remains"
description: "Classifies #2928's horizon from repo evidence: the auto-block-after-N-failures spin-loop core already shipped as internal/attemptbudget; the residual durable-SQLite-board + atomic-claim primitive is a cmd/dispatch-lane epic blocked on the issue's own operator-input acceptance gate."
---

# Triage: durable kanban board / auto-block-on-spin (#2928)

*2026-07-15 — docs-lane triage, generation stream unclassified (triage-only frame).*

[#2928](https://github.com/anthony-chaudhary/fak/issues/2928) asks fak to adopt
Hermes' durable SQLite kanban shape: a durable task board, an atomic claim
primitive, **auto-block a task after `failure_limit` (default 2) consecutive
non-success attempts** with a witnessed attempt count, and a **spin-loop
regression test**. Its parent is program #2908 (Hermes-parity gaps).

This note classifies the horizon from repo evidence *before* any implementation,
which is the proof bar for an unclassified generation issue: do not guess the
stream; read what the tree already carries and name what is genuinely missing.

## The load-bearing ask already shipped

The issue's own **primary** value — "auto-blocks a task after N witnessed
failures … no infinite re-pick storm" plus "a spin-loop regression test" — is
already in the tree as `internal/attemptbudget`, a pure fold over one issue's
attempt history, with a CLI front door and a full property-test suite:

- **Auto-block-after-N-failures.** `attemptbudget.Decide` returns `StatusHeld`
  once `AttemptCount >= Budget` (`Budget` is fak's `failure_limit`), so a
  repeatedly failing issue stops being re-offered instead of spinning workers
  forever. This is the #1777 spine, extended by #1778 (per-failure-class backoff
  windows), #2860 (a structured, queryable `BlockReason` + `Route` — genuinely
  stuck vs. flaky vs. precondition-unmet), and #2892 (a **witnessed** transient
  vs. structural adjudication: a rate-limit/network last-failure keeps retrying
  under a measured backoff up to an extended ceiling instead of a blunt hold).
- **Witnessed attempt count.** `Decision.AttemptCount`, `Decision.Verdict`
  (`transient_retry | transient_exhausted | structural_block`), and the
  `BlockReason`/`Route` fields are a pure fold over the recorded per-attempt
  facts — re-running `Decide` over the same history reproduces the verdict
  exactly, so the block is witnessed, not self-reported.
- **The spin-loop regression test.** `TestDecide_RepeatedAttemptsMoveDispatchableToHeld`
  in `internal/attemptbudget/attemptbudget_test.go` is precisely the regression
  the issue asks for: a fixture with repeated attempts moves dispatchable → held.
  `TestDecide_RateLimitedTaskRetriesWithBackoffInsteadOfHardBlock` even names
  "the exact history Hermes' `failure_limit` auto-blocks," and improves on it —
  a self-clearing transient wall is not blunt-blocked the way Hermes would.
- **Exposed.** `cmd/fak/attempt_budget.go` wires `fak dispatch attempt-budget`
  (`--budget`, `--json`), so the policy is reflected in a report, not just
  defined in a leaf.

Against the issue's four acceptance clauses, three are already met: *auto-block
after N failures*, *a witnessed attempt count*, and *a spin-loop regression
test*.

## What genuinely remains (and why it is not this lane / not triage-only)

The residual is the **durable SQLite board itself** plus an **atomic-claim
primitive** as a new persistence subsystem. That work is deliberately *not*
landed here, for three concrete reasons the repo evidence names:

1. **Blocked on the issue's own operator-input acceptance gate.** #2928's
   `## Acceptance gate` reads verbatim: *"unknown — needs operator input: what
   package/command will host the board (new `internal/kanban` or an extension of
   an existing dispatch package), and what is the regression test's name?"* That
   is a `HUMAN_RESIDUAL`, not a decidable witness — a worker cannot pick the
   board's home package or its schema without an operator choosing the shape.
2. **Wrong lane for this artifact.** #2928 dispatched to the **docs** lane, but
   its own `## Likely files` are `cmd/fak/dispatch_auto.go` /
   `cmd/fak/dispatch_knownbad.go` — the **cmd** lane. A board implementation
   committed under a `(fak docs)` ship stamp would not diff-witness (the same
   lane-vs-file-tree mismatch #2467 hit). The board is a `cmd`/dispatch leaf.
3. **Atomic claim is largely already served.** The issue itself scopes out "a
   rewrite of the existing lane/lease arbiter (`dos_arbitrate`)": lane leases
   already give an atomic, disjoint claim under a single-writer discipline. The
   net-new is *durable persisted claim history*, not a new claim mechanism.

The single-dispatcher WAL-safety posture the issue cites is, likewise, a
property of *whatever* durable store is chosen — it cannot be specified before
clause (1) is answered.

## Classification

- **Stream: keep `needs-triage` (do not guess).** The horizon is genuinely
  mixed. The delivered spin-loop core is `gen/now` trunk-hygiene value that has
  *already* landed with witnesses; the residual durable board is a `gen/next`
  near-term foundation that still needs an operator-chosen schema/home package
  and a WAL-safety gate before it is runnable. Per `docs/generation.md`'s intake
  rule, an unclear stream stays `needs-triage` rather than being laundered into
  one label — and `gen/future` is explicitly not a dumping ground.
- **Smallest honest next step:** an operator answers #2928's acceptance gate
  (board home package + regression-test name), after which the residual becomes
  a single `cmd`/dispatch leaf: persist `attemptbudget`'s existing per-issue
  attempt history + `Decision` to a durable store and re-derive dispatchability
  from it, reusing the already-shipped fold rather than reinventing the block.

## Generation close evidence

- **Promotion evidence:** the auto-block-after-N-failures core + its spin-loop
  regression test are shipped and CLI-exposed (`internal/attemptbudget`,
  `cmd/fak/attempt_budget.go`, #1777/#1778/#2860/#2892), retiring the "infinite
  re-pick storm" blocker the issue was filed against for the in-memory path.
- **Demotion / retirement evidence:** the *durable-board* clause is demoted to
  blocked-on-operator-input — its named assumption ("a SQLite-backed board is
  acceptable … rather than reusing an existing fak store") is untested against an
  operator decision, and the issue's acceptance gate is explicitly `unknown`.
- **Invalidating assumption:** this triage assumes the durable requirement is
  *persistence of claim/attempt history*, and that `attemptbudget`'s existing
  fold is the reusable decision layer. If an operator instead wants a
  board that owns the *scheduling* decision (not just records it) — a real
  cross-process WAL-contended writer with its own claim FSM — then the residual
  is a larger new subsystem than "persist the existing fold," and this note's
  "smallest next step" is wrong and should be re-scoped.
