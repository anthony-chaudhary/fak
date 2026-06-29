---
title: "The symptom witness — a fix is witnessed-fixed only when a test fails on the bug and passes on the fix"
description: "The verification standard for `fix(...)` commits. A green `dos commit-audit` is diff-witnessed — it proves the diff does the KIND of thing the message claims — but it is NOT a correctness check, so a fix can pass the audit while the symptom is fully live. This page pins what 'witnessed fixed' must additionally require: for a fix of a reproducible symptom, the diff ships a SYMPTOM WITNESS — a test that FAILS on the parent commit's code and PASSES on the fix, because it reproduces the actual failure condition rather than encoding the bug as the expected behavior. It names the red-then-green check (the test analog of the diff-witness), the distinguishing property, the anti-pattern that defeats it, and the guard-login class as the worked example (`28dd3b33` is the witness; `c9cd25b2` is the diff-witnessed-but-wrong counter-example). The correctness-rung sibling of the verification-ladder spec (which orders the rungs) and net-true-value (which grades a gain): this grades whether a fix carries proof the symptom is gone."
---

# The symptom witness

A bug-fix commit can pass `dos commit-audit` with verdict **OK / diff-witnessed** while the
bug is **still fully live**. The audit is honest about its own rung — it grades
*did-the-diff-do-the-KIND-of-thing-claimed*, never *is-the-code-correct*. Its own
interpretation says so plainly:

> WITNESSED — the diff does the KIND of thing the message claims (rung: diff-witnessed).
> This is NOT a correctness check (a wrong-but-real change still passes); it only confirms
> the claim isn't empty. Run the tests for correctness.

The gap this page closes is in the **workflow around** that rung, not the rung itself: a
green commit-audit was once read as "fixed," and there was no required artifact proving the
**symptom is gone**. This page names that artifact.

## The rule

> For a commit whose message claims `fix(...)` of a **reproducible symptom**, the diff MUST
> include a **symptom witness**: a test that **FAILS on the parent commit's code** and
> **PASSES on the fix** — not merely a test of the new helper in isolation.

The distinguishing property is not "the diff touched a `_test.go`." It is that the test
**reproduces the actual failure condition** — the exact shape that made the symptom occur.
A symptom witness is the *test* analog of `dos commit-audit`'s diff-witness: the diff-witness
proves the change is non-empty; the symptom witness proves the change *constrains the bug*.

## The anti-pattern: a test that encodes the bug

The failure mode a symptom witness is built to catch is a test that asserts the buggy
behavior as the expected one — it can never go red on the bug, so it witnesses nothing:

- The first guard unit test asserted `guardModeInteractive(os.ModeCharDevice) == true`. On
  the box where the bug lived, `os.ModeCharDevice` **was** the broken signal, so the
  assertion encoded the bug as the contract. It passed on the broken code and on the fixed
  code alike — green, and blind.
- A test driven from a shape the broken check *already handled* is the same blindness in
  disguise. A pipe-stdin test reports `os.ModeNamedPipe`, which even the broken gate
  classified correctly, so it passes on both versions and witnesses nothing.

A test that passes on the parent commit is not a witness for the fix — it is a witness for
something both versions already do.

## The check: red-then-green (the apply-test-to-old-code witness)

The mechanical form of the rule, and the test analog of the diff-witness. It is runnable
today by hand, with no new kernel verb — point it at the fix commit `F` and its parent `P`:

```bash
# 1. On the fix commit, keep its NEW test but restore the parent's PRODUCTION code:
git checkout F -- <the_new_test>.go          # the symptom witness from the fix
git checkout P -- <the_changed_production>.go # the pre-fix code the bug lived in

# 2. RED — the witness must FAIL against the pre-fix code (it reproduces the bug):
go test ./<pkg> -run <TheWitnessTest>         # MUST be a failure / non-zero exit

# 3. GREEN — restore the fix; the same witness must now PASS:
git checkout F -- <the_changed_production>.go
go test ./<pkg> -run <TheWitnessTest>         # MUST pass
```

A test that is red at step 2 and green at step 3 is a real witness: it failed *because* the
bug was present and passes *because* the fix removed it. A test that is green at step 2 is a
tautology — discard it and write one that reproduces the failure condition. (In this repo,
native `go test` is OS-blocked on the Windows dev box; run the witness under WSL via
`./test.ps1 ./<pkg>/` or in CI — the red-then-green property is the same wherever it runs.)

The weaker, advisory form is a heuristic: a `fix(...)` commit that changes production code
**should** also add or change a `_test.go` in the same diff. It is a cheap signal that a
witness *might* be present; only the red-then-green check above proves that it *is*.

## Where it sits on the verification ladder

The symptom witness is a rung **above** diff-witnessed on the
[verification ladder](verification-ladder-spec.md). Diff-witnessed is an in-process
structural read of the commit (does the diff do the KIND of thing claimed). The symptom
witness is a `suite`-cost rung: it builds and runs a test red-then-green, so it costs
seconds, not nanoseconds, and you climb to it only when the claim is a `fix(...)` of a
reproducible symptom — exactly the smallest-sufficient-rung discipline the ladder encodes. A
green diff-witness that never climbs to the symptom-witness rung for a reproducible-bug fix
is the cheap rung silently allowing a claim it cannot conclusively decide.

## The worked example: the guard "stuck on login" class

This standard exists because the gap fired concretely, and the repo now carries both halves
of the lesson as durable commits.

**The diff-witnessed-but-wrong counter-example — [`c9cd25b2`](https://github.com/anthony-chaudhary/fak/commit/c9cd25b2).**
`fix(guard): fail loud headless when no Claude token exists …` — `dos commit-audit` =
**OK / diff-witnessed**. The fix used `os.ModeCharDevice` to detect a headless stdin. On
Windows, `NUL` / `</dev/null` reports **as** a character device, so the gate treated the
exact headless-automation case as interactive and **never fired** — the symptom (a hang on a
login the wrapped agent cannot complete) was unchanged. The unit test passed (a synthetic
char-device mode *is* interactive); the shipped binary did not. The bug was found only by
hand-building the binary and noticing the gate stay silent.

**The symptom witness — [`28dd3b33`](https://github.com/anthony-chaudhary/fak/commit/28dd3b33)**
([`cmd/fak/guard_login_e2e_test.go`](../../cmd/fak/guard_login_e2e_test.go)). It re-execs the
real `fak guard -- claude` entry point headless with stdin driven from `os.DevNull` — the
Windows char-device shape — and asserts exit-2-with-guidance within a hard 20s deadline (a
hang = a timeout = a test failure). The `DevNull` stdin is load-bearing: it is the shape the
original `os.ModeCharDevice` gate mishandled, and it was verified standalone that
`term.IsTerminal(DevNull) == false` while `os.ModeCharDevice(DevNull) == true`. So this test
**fails on the pre-fix check** (which read `ModeCharDevice` and called `NUL` interactive) and
**passes on the fix** (which reads `term.IsTerminal`, the seam at
[`cmd/fak/guard.go`](../../cmd/fak/guard.go) `guardFdIsTerminal`). A real witness, not a
tautology. The symptom's absence now lives in a CI-resident test instead of a transcript of
someone eyeballing stderr once.

## Honest fences

- **This is the documented rule plus a runnable check, not yet an automated gate.** The
  red-then-green recipe above is a procedure an agent or operator runs by hand today; there
  is no `dos`/`fak` verb that auto-applies a fix's new test to its parent commit and refuses
  the commit on a missing-or-tautological witness. That automated gate (the strongest form of
  the issue's proposal) is the named follow-on, not shipped here.
- **It applies to a *reproducible* symptom, not every `fix(...)`.** A fix for a non-observable
  invariant, a build break, or a typo may have no reproducible runtime symptom to witness; the
  rule binds when there is a failure condition a test can reproduce. Honesty over a forced
  witness: a test contrived to pass-then-fail without reproducing the real condition is the
  anti-pattern, not the rule.
- **Diff-witnessed is still required, not replaced.** The symptom witness is a rung *above*
  the diff-witness, not a substitute. A commit still owes a gradeable, diff-witnessed subject
  (`fix(scope): <verb> … (fak <leaf>)`); the symptom witness is the additional artifact a
  reproducible-bug fix carries on top.
- **Distinct from the per-turn harness-quality gates.** This grades whether a *fix* carries
  proof its *symptom* is gone (#1326). It is not the per-turn harness-quality gate (#1323) or
  the loop-body grader (#1315), which grade the agent's loop, not a fix's witness.

## Cross-references

- [The verification-ladder spec](verification-ladder-spec.md) — the cost-ordered rung ladder this witness is a rung of (a `suite`-cost rung above the in-process diff-witness); same fail-closed, smallest-sufficient-rung discipline.
- [Net-true-value](net-true-value.md) · [The observer-effect contract](observer-effect.md) · [The support-maturity honesty fence](support-maturity-honesty-fence.md) — the sibling honesty standards in `docs/standards/`: net-true-value grades a *gain*, observer-effect grades a *cost number*, this grades a *fix's symptom proof*.
- [`cmd/fak/guard_login_e2e_test.go`](../../cmd/fak/guard_login_e2e_test.go) — the worked-example symptom witness (`28dd3b33`).
- [Claims ledger](../../CLAIMS.md) — shipped vs stub, claim by claim.
