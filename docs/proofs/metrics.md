# D14 · metrics

`internal/metrics` is fak's KPI/observability layer and the A/B `report.json` shape. It computes
**latency-histogram percentiles** (`Hist.P50`/`P99`/`Mean` over a sorted ns sample slice, log-spaced
`Buckets`), and owns the **paired A/B `Report`**: two `Arm`s (vdso On/Off), the five-counter `KPIs`
set, the **identical-workload guard** (`Validate`), and the **on-vs-baseline gate** (`ComputeGate`).
The package is deliberately pure (no engine/kernel import) so the bench, the steward KPI-regression
check, and external `/metrics` consumers can reuse it. "Correct" here is regime **D** (decision /
aggregation soundness): a percentile read is a monotone index into a sorted array, the gate is
fail-closed, and a KPI fold must equal its arithmetic definition over the paired arm counters.

This box is a macOS fleet node (Darwin/arm64), so witnesses run with native `go test` from `fak/`
(the Windows/WSL machinery in the root `CLAUDE.md` does not apply here).

---

## THEOREM 1 — histogram percentiles are monotonic in q (P50 <= P99)

**THEOREM.** For `Hist.pct(p) = sorted[int(p/100·(n−1))]` over the ascending-sorted sample slice, the
percentile is monotone non-decreasing in `p`; concretely `P50() <= P99()`. (The prompt's literal
`p50 <= p90 <= p99` names a `p90` the type does not expose — there is no `P90` method.)

**REGIME.** D — decision/aggregation soundness (ordered read of a sorted reduction).

**PROOF.** `pct` at `internal/metrics/metrics.go:31-45`: line 35-36 copies the samples and sorts them
ascending via `sort.Slice`; line 37 maps the quantile to an index `idx = int(p/100·float64(len(s)−1))`,
clamped to `[0, len−1]` (lines 38-43). The map `p ↦ idx` is monotone non-decreasing, and `s` is
non-decreasing after the sort, so `p1 <= p2 ⇒ pct(p1) <= pct(p2)`; in particular
`P50()=pct(50) <= P99()=pct(99)` (`metrics.go:47-48`). The witness pins the exact arithmetic:
on the input `{100,200,…,1000}` (`metrics_test.go:12-14`), `idx(50)=int(0.5·9)=4 ⇒ 500` and
`idx(99)=int(0.99·9)=8 ⇒ 900`, and it asserts both the order `p50<=p99` and those exact values
(`metrics_test.go:36-47`), plus that an empty hist returns 0 without panic (lines 50-53).

**WITNESS.** `go test ./internal/metrics/ -run TestHistPercentilesMonotonic -count=1 -timeout 120s -v`

**VERDICT.** **PROVEN** (2026-06-20) — `--- PASS: TestHistPercentilesMonotonic (0.00s)`,
`ok github.com/anthony-chaudhary/fak/internal/metrics 0.173s`, for the witnessed single-input instance.
**Residual OPEN:** only one concrete sample set is witnessed; there is no `P90` method and no
`testing/quick` property asserting the ∀-q monotonicity (grep for `p90`/`quick`/`Property`/`Fuzz`
returns no match). The general form would be closed by a quick-style property over random sample
slices and a random `q ∈ [0,100]` asserting `pct` non-decreasing.

**DOS.** bound at ship.

---

## THEOREM 2 — the A/B KPI fold is correct (paired aggregation matches the definition)

**THEOREM.** The five KPI fields aggregate the paired On/Off `Arm` counters per their definitions:
`vdso_hit_rate = VDSOHits/Calls`, `context_pollution_rate = Quarantines/Calls`,
`tokens_per_task = (InTokens+OutTokens)/Calls`, `tool_call p50/p99 = On`-arm percentiles, and
`token_delta_pct = 100·(offTok−onTok)/offTok`.

**REGIME.** D — paired aggregation soundness.

**PROOF / SCOPE NOTE.** The fold that actually *computes* these KPIs lives **outside this scope**, in
`internal/bench/bench.go:237-256` (`hitRate = VDSOHits/Calls` line 239; `ContextPollutionRate =
rate(Quarantines, Calls)` line 245; `TokensPerTask = (In+Out)/max1(Calls)` line 246; `TokenDeltaPct =
100·(offTok−onTok)/offTok` line 253). Within `fak/internal/metrics` the only paired-aggregation code
is the **pairing precondition guard** `Validate` (`metrics.go:175-180`), which refuses to compare arms
with mismatched workload hashes — witnessed green by `TestValidateWorkloadHash` — and the
**on-vs-baseline gate** `ComputeGate` (`metrics.go:162-171`), fail-closed at a zero baseline, witnessed
by `TestComputeGate`. The KPI numbers themselves are only **round-tripped structurally** by
`TestReportJSONAndTokenDelta` (`metrics_test.go:150-212` sets and reads `InTokens/OutTokens/
TokenDeltaPct` through JSON but never derives a KPI from arm counters). So **no in-scope test asserts
KPI == definition**; the theorem is therefore not discharged within `internal/metrics`.

**WITNESS.** `go test ./internal/metrics/ -run 'TestReportJSONAndTokenDelta|TestValidateWorkloadHash|TestComputeGate' -count=1 -timeout 120s -v`

**VERDICT.** **OPEN** (2026-06-20) — all three named tests PASS green here, but none of them assert the
fold-equals-definition; they witness the report shape, the pairing guard, and the baseline gate, not the
KPI arithmetic. Closes by either (a) a metrics-scoped test that builds On/Off `Arm`s with known counters
and asserts each KPI equals its definition, or (b) relocating this obligation to the bench package and
witnessing `bench.go:237-256` there. Not promoted to PROVEN on argument alone.

**DOS.** bound at ship.

---

### Honesty ledger

| # | Theorem | Verdict | Why |
|---|---|---|---|
| 1 | percentile monotone (p50<=p99) | PROVEN (instance) / OPEN (∀q form) | `pct` is a monotone index into a sorted slice; `TestHistPercentilesMonotonic` green on one input; no `P90`, no property test for the general form. |
| 2 | A/B KPI fold == definition | OPEN | the computing fold is in `bench.go` (out of scope); in-scope tests only round-trip the KPI fields and witness the pairing guard + baseline gate, not KPI arithmetic. |
