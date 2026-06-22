---
title: "fak proof: preflight cheapest-first rung ladder"
description: "Structural proof for fak's preflight ladder: rungs evaluate cheapest-first by ascending index, and a cheap-rung Deny short-circuits the expensive rungs."
---

# A7 · preflight

The `preflight` ladder (`fak/internal/preflight`) is a **cheapest-first rung ladder**:
a sequence of well-formedness checks that adjudicate a `*abi.ToolCall` *before* it
fires, so a malformed or unsafe call is caught without spawning a process or burning a
model turn. v0.1 has two rungs — **rung 0** static JSON parse (`preflight.go:74-79`) and
**rung 1** JSON-Schema required-field + type validation (`preflight.go:82-100`). A catch
returns `VerdictDeny` and is recorded as a typed hard-negative row carrying the
`(RungPassed, RungFailed)` index pair; a well-formed call returns `VerdictDefer`
(the ladder has nothing to prove and yields to the authoritative monitor).

"Correct" for this module (regime **A — algebraic/structural**) is two structural
invariants of the ladder, *independent of the numerical content of any rung*:
(1) the rungs are evaluated **cheapest-first** — by ascending rung index, rung 0 strictly
before rung 1 — and (2) a Deny at a cheaper rung **short-circuits** every more-expensive
rung, so no evaluation is wasted on an already-rejected call. Both are observable through
the `(RungPassed, RungFailed)` indices stamped into the emitted hard-negative row, which
makes them deterministically witnessable rather than narrated.

> **Naming caveat (honesty).** The obligation text says "order == FoldRank." `abi.FoldRank`
> (`internal/abi/registry.go:740`) ranks **verdict restrictiveness** for the *kernel's*
> cross-adjudicator fold (most-restrictive-wins) — it is **not** the mechanism preflight
> uses to order its own rungs. Preflight orders by **sequential rung index** (source order,
> documented at `preflight.go:1-16,64`). The proofs below discharge the real, code-grounded
> cheapest-first-rung claim; the `FoldRank` wording is recorded as a loose borrow, not
> silently treated as the witnessed mechanism.

---

## Theorem A7.1 — rungs evaluate cheapest-first (ascending rung index)

**THEOREM.** For every `ToolCall`, `Adjudicate` evaluates rung 0 (static parse) strictly
before rung 1 (schema validation); the rung indices recorded in the hard-negative row are
monotone increasing (rung 0 → rung 1), i.e. the ladder is cheapest-rung-first.

**REGIME.** A — structural (total order over rung indices).

**PROOF.** `Adjudicate` runs the rungs as plain sequential Go statements: the rung-0 parse
probe at `preflight.go:74-79`, then the rung-1 schema loop at `preflight.go:82-100`. There
is no reordering — the evaluation order *is* the source order, which is cheapest-first by
construction (the package contract, `preflight.go:1-16,64`). Each catch stamps its rung
position into the negative row via `caughtAt(c, passed, failed, …)` (`preflight.go:106-126`):
a rung-0 catch records `RungPassed=-1, RungFailed=0` (`preflight.go:77`); a rung-1 catch
records `RungPassed=0, RungFailed=1` (`preflight.go:94,97`). `TestRung0FailureNeverReachesRung1`
installs a schema (so rung 1 *would* stamp `RungFailed==1` if reached) and feeds unparseable
args, asserting the lone row has `RungFailed==0, RungPassed==-1` (`preflight_test.go:87-92`)
— rung 0 first. `TestNegativesRowFields` asserts a rung-1 catch row has `RungPassed==0,
RungFailed==1` (`preflight_test.go:127-129`) — rung 0 was passed *before* rung 1 ran. The
two index facts together pin the ascending 0→1 order.

**WITNESS.** `go test ./internal/preflight/ -count=1 -timeout 120s -run 'TestRung0FailureNeverReachesRung1|TestNegativesRowFields' -v`

**VERDICT.** **PROVEN** — 2026-06-20. `--- PASS: TestRung0FailureNeverReachesRung1`,
`--- PASS: TestNegativesRowFields`, `ok github.com/anthony-chaudhary/fak/internal/preflight 0.241s`.

**DOS.** bound at ship (ship commit `c72ddf1` "fak v0.1.0"; `dos commit-audit` / `dos verify fak preflight` to confirm diff-witness).

---

## Theorem A7.2 — a Deny at a cheap rung short-circuits the expensive rungs

**THEOREM.** When a cheaper rung Denies, every more-expensive rung is skipped: a rung-0
Deny means rung 1 (schema validation) is never evaluated — no work is wasted on a call
already rejected.

**REGIME.** A — structural (early-exit / short-circuit).

**PROOF.** Every rung's catch path is an *immediate* `return l.caughtAt(...)`. Rung 0
returns inside its parse-failure branch at `preflight.go:77`, before the schema lookup at
`preflight.go:82` is reached. Since Go executes statements top-down and `return` exits the
function, a rung-0 Deny structurally cannot fall through into the rung-1 schema loop
(`preflight.go:91-99`). `TestRung0FailureNeverReachesRung1` makes the skip observable: it
installs a schema requiring `"origin"` (so rung 1, if reached on this same call, would
itself catch and stamp `RungFailed==1`), feeds `"{bad"`, and asserts (a) `VerdictDeny` is
returned (`preflight_test.go:78-80`) and (b) there is **exactly one** negative row with
`RungFailed==0` (`preflight_test.go:83-89`). Had rung 1 run, a `RungFailed==1` row (or a
second row) would exist; the single `RungFailed==0` row is the witness that the expensive
schema rung was skipped — the cheap Deny short-circuited it.

**WITNESS.** `go test ./internal/preflight/ -count=1 -timeout 120s -run 'TestRung0FailureNeverReachesRung1' -v`

**VERDICT.** **PROVEN** — 2026-06-20. `--- PASS: TestRung0FailureNeverReachesRung1`,
`ok github.com/anthony-chaudhary/fak/internal/preflight 0.241s` — exactly one negative row,
`RungFailed==0`, `RungPassed==-1`.

**DOS.** bound at ship (ship commit `c72ddf1` "fak v0.1.0"; `dos commit-audit` / `dos verify fak preflight` to confirm diff-witness).
