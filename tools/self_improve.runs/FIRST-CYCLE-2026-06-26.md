# dos-self-improve — first end-to-end cycle on fak (#388)

Date: 2026-06-26. Harness: [`tools/self_improve.py`](https://github.com/anthony-chaudhary/fak/blob/c588c993e221c11ddba60df10fb5a3132ec8eb2a/tools/self_improve.py) (since ported off Python; link pinned to the run's base SHA). Base SHA
(pinned): `c588c993e221c11ddba60df10fb5a3132ec8eb2a` (`main` at run time).

This is the first time the recursive-self-improvement loop ran end-to-end on fak:
*propose a change → keep it only if a witness the change's author did not write
confirms an improvement → otherwise REVERT.* Until this ran once, "fak improves
fak" was a design (GROWTH §7). It is now a capability, witnessed below.

## The witness gate (GROWTH §5)

A candidate is KEPT iff **all** hold, else REVERTED — and the keep-bit is the AND
of three facts the loop did **not** author:

1. **suite green** on a clean isolated worktree — `go test ./internal/<leaf>/...`
2. **`internal/architest` green** — the layering/tier contract held
3. **`dos verify fak <leaf>`** confirms the leaf shipped from git evidence, and the
   verdict is **not** `source=none` / `subject-only` (witness honesty, #125)

The kernel then **ratifies** with its typed `dos improve` verdict over the same
env-authored facts. KEEP requires BOTH the AND-of-three AND `dos improve = KEEP`;
either says no → REVERT (fail-safe). The loop cannot keep a change by *narrating*
that it is better — `dos improve` parses the narration for nothing.

## Cycle 1 — KEEP (the good candidate)

A genuine additive improvement to `internal/benchids`: a new passing test guarding
the `LCG` id-bound invariant across a vocab sweep.

| witness | verdict |
|---|---|
| 1 · suite (`go test ./internal/benchids/...`) | **GREEN** (exit 0) |
| 2 · `internal/architest` | **GREEN** (exit 0) |
| 3 · `dos verify fak benchids` | **CONFIRMS** — shipped, `source=grep-subject`, `rung=trailer` |
| metric · leaf test count | 4 → **5** (a real coverage gain) |
| kernel · `dos improve` | **KEEP** (exit 0) |
| **DECISION** | **KEEP** |

- Candidate subject: `test(benchids): guard LCG id bound across vocab sweep (fak benchids)`
- **First kept SHA:** `0b46ae6e965687c3d663899037ca2951d2094695`
- Record: [`cycle-01-good-0b46ae6e.json`](cycle-01-good-0b46ae6e.json) (carries the full diff)

## Cycle 2 — REVERT (the bad candidate)

A deliberate regression: a failing test injected into the same leaf, to prove the
REVERT arm fires on a red suite — and that a metric gain **cannot rescue it**.

| witness | verdict |
|---|---|
| 1 · suite (`go test ./internal/benchids/...`) | **RED** (exit 1) — the injected `t.Fatal` |
| 2 · `internal/architest` | GREEN (exit 0) |
| 3 · `dos verify fak benchids` | CONFIRMS — shipped, `source=grep-subject` |
| metric · leaf test count | 4 → **5** (the failing test still counts — a gain!) |
| kernel · `dos improve` | **REVERT** (exit 3) — the red suite is the non-negotiable floor |
| **DECISION** | **REVERT** |

- Candidate subject: `test(benchids): inject a failing test to exercise REVERT (fak benchids)`
- **First reverted attempt SHA:** `933e29bb5e6f0d98fd43eeb8fb587f3866254da3` (discarded with its worktree)
- Record: [`cycle-02-bad-933e29bb.json`](cycle-02-bad-933e29bb.json)

**Why this cycle is the load-bearing one:** the bad candidate had a *strictly higher*
metric (5 > 4) — exactly the signal a loop grading its own homework would keep on.
The witness reverted it anyway, because witness 1 (the runner's exit code, not the
loop's word) was red. The keep-bit is non-forgeable: a change cannot buy a KEEP with
a number while the suite is red.

## Isolation (the second acceptance bullet)

Every candidate was applied in an **isolated `git worktree --detach` off the pinned
base**, never the live shared tree. This is what made the baseline green at all: the
live tree currently carries an untracked peer package (`internal/reportout`) that
makes `architest` RED, but the clean worktree off the committed base does not
contain it, so witness 2 was honestly GREEN. The loop never edited the live shared
worktree (avoiding the 9p edit/read race in GROWTH §7); both throwaway worktrees
were removed and pruned after each cycle.

**Promotion is deliberately NOT exercised here.** Honoring "the loop never edits the
live shared worktree", a KEEP records the kept SHA and preserves the candidate's full
diff as an artifact (in the cycle record) — it does **not** merge onto the live
trunk. Promoting a kept candidate to `main` is a separate, `dos verify`-gated
operator step, kept out of the unattended loop on purpose.

## Reproduce

```bash
python tools/self_improve.py --leaf benchids        # both arms (this seed)
python tools/self_improve.py --leaf benchids --only good   # just KEEP
python tools/self_improve.py --leaf benchids --only bad    # just REVERT
python -m pytest tools/self_improve_test.py -q             # the pure keep-logic gate
```
