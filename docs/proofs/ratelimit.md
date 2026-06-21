# D9 · ratelimit

`internal/ratelimit` is fak's throughput/cost governor — the adjudicator that turns the
already-plumbed `RATE_LIMITED` reason into a real enforcer. It maintains a per-key
counter (`{calls, cost}`) bucketed by trace, tool, or global, and on each `ToolCall`
either **Defers** (abstains — under cap) or emits **`Deny(RATE_LIMITED)`** (over cap),
a reason the kernel maps to a `WAIT` disposition so a runaway loop backs off instead of
burning another model turn. It is **fail-open by default**: with no cap configured the
limiter Defers on every call, so registering the leaf changes no behavior until an
operator sets `FAK_RATELIMIT_MAX_CALLS` / `FAK_RATELIMIT_MAX_COST` (or calls `SetLimit`).

"Correct" here is **regime D — decision-procedure soundness**: the gate's verdict must be
*sound* (it must deny once a key has exhausted its quota/budget — it never admits past the
cap) and its accounting must be *conservative* (no call is charged twice, no credit leaks,
a refused call consumes nothing). Two falsifiable theorems capture this.

---

## Theorem 1 — the quota/budget bounds throughput

**THEOREM.** For a key configured with `MaxCalls = N` (resp. `MaxCost = B`), the first `N`
admitted calls (resp. calls whose cumulative cost stays `≤ B`) Defer/Allow, and the
`(N+1)`-th call (resp. the call that would push cumulative cost past `B`) emits
`Deny(RATE_LIMITED)`. Driven through a real kernel, that Deny carries the `WAIT`
disposition.

**REGIME.** D — decision-procedure soundness (fail-closed-at-the-cap; the over-cap call
is shed cheaply at rank 8 before the heavy trust rungs run).

**PROOF.** The cap is the strict pre-consume comparison in `Adjudicate`:
`if r.lim.MaxCalls > 0 && st.calls+1 > r.lim.MaxCalls` returns `denyVerdict` before any
counter mutation (`fak/internal/ratelimit/ratelimit.go:178`), and analogously
`st.cost+cost > r.lim.MaxCost` (`ratelimit.go:182`). Because the comparison uses
`st.calls+1` — the count *this* call would reach — the boundary is exact: the gate admits
while `st.calls < N` and denies at `st.calls == N`, i.e. precisely on the `(N+1)`-th call.
`denyVerdict` sets `Reason = abi.ReasonRateLimited` (`ratelimit.go:220`); the kernel folds
that reason to the `WAIT` disposition. All three key dimensions (per-trace / per-tool /
global) and the explicit-cost override (`Meta["fak.ratelimit.cost"]`, `costOf`,
`ratelimit.go:238`) route through the same check.

**WITNESS.**
```
(go test ./internal/ratelimit/ -count=1 -timeout 120s \
  -run 'TestQuotaDeniesOverCap|TestCostBudgetDeniesOverBudget|TestGlobalMode|TestPerToolMode|TestPerTraceIsolation|TestExplicitCostOverride|TestRateLimitedDenySurfacesWaitDisposition' -v)
```
`TestQuotaDeniesOverCap` (ratelimit_test.go:53) — 3 under-cap calls Defer, the 4th is
`Deny(RATE_LIMITED)`, `Stats` = `admits=3 denies=1`.
`TestCostBudgetDeniesOverBudget` (ratelimit_test.go:125) — costs 5 then 4 fit (total 9 ≤
10), the `+2` call denies (11 > 10), a later empty-arg call still fits.
`TestRateLimitedDenySurfacesWaitDisposition` (ratelimit_test.go:203) — through real
`kernel.Submit`/`Syscall` the 4th call denies `RATE_LIMITED`, `kernel.Disposition(reason)
== "WAIT"`, and `DenyResult.Meta["disposition"] == "WAIT"`; a fresh trace is unaffected.

**VERDICT.** **PROVEN** — 2026-06-20, macOS native go1.26 node. All 7 selected tests PASS;
`ok github.com/anthony-chaudhary/fak/internal/ratelimit 0.266s`.

**DOS.** bound at ship (`dos commit-audit` on the ship commit; `dos verify fak ratelimit`).

---

## Theorem 2 — the budget is conserved (no double-spend, no leak)

**THEOREM.** Accounting never double-spends or leaks credit: a call's cost is added to its
per-key counter exactly when — and only when — the call is admitted; a **denied call
consumes no budget** (the admit ledger never advances on a `Deny`), and `Reset`/`ResetAll`
restore exactly the cleared key's budget. An exhausted key probed repeatedly returns an
idempotent `WAIT` with the admit count pinned at the cap.

**REGIME.** D — decision-procedure soundness (conservation invariant of the gate's state).

**PROOF.** Conservation follows from **check-then-consume** ordering under a single mutex
(`r.mu` is held for the whole `Adjudicate`). The counter mutations
`st.calls++; st.cost += cost; r.admits++` (`fak/internal/ratelimit/ratelimit.go:187`)
execute only on the fall-through admit path, strictly *after* both deny checks have already
returned (`ratelimit.go:178`–`185`). Therefore (a) a denied call mutates nothing — no
double-spend and no leak past the cap; (b) an admitted call adds its cost exactly once. Each
decision increments exactly one of `admits` / `denies` / `dropped`, partitioning calls
disjointly. `Reset` (`ratelimit.go:125`) deletes one key's counter so a re-decision starts
from a fresh zero counter (full budget restored); `ResetAll` replaces the whole map, leaving
no partial-credit residue. The inert path returns `defer_` before touching any counter
(`ratelimit.go:151`), so an unconfigured limiter is the zero-spend identity.

**WITNESS.**
```
(go test ./internal/ratelimit/ -count=1 -timeout 120s \
  -run 'TestDeniedCallConsumesNoBudget|TestResetClearsBudget|TestBoundedKeysFailOpen|TestInertUntilConfigured' -v)
```
`TestDeniedCallConsumesNoBudget` (ratelimit_test.go:71) — after 2 admits and 5 over-cap
probes, `Stats` `admits == 2` (the probes consumed nothing; the WAIT is idempotent).
`TestInertUntilConfigured` (ratelimit_test.go:40) — 100 calls on an inert limiter leave
`admits=0 denies=0`.
`TestResetClearsBudget` (ratelimit_test.go:155) — after a deny, `Reset("trace:t")` restores
the full budget.
`TestBoundedKeysFailOpen` (ratelimit_test.go:169) — a key past the `maxKeys` ceiling
fails-open (`dropped++`) without evicting a live budget; an existing key still enforces.

**VERDICT.** **PROVEN** — 2026-06-20, macOS native go1.26 node. All 4 selected tests PASS;
`ok github.com/anthony-chaudhary/fak/internal/ratelimit 0.189s`. (Full package: `ok ...ratelimit
0.195s`.)

**DOS.** bound at ship (`dos commit-audit` on the ship commit; `dos verify fak ratelimit`).
