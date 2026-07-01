// overheadbudget.go — the DECLARED per-rung/per-method overhead envelope for the
// self-tax plane (epic #1147, ticket T2 / issue #1150).
//
// The per-turn meter (the rest of this package) and the offline `rung_overhead`
// fold (T1, over internal/rungobs) can report what a lifecycle span COST — its
// elapsed-ns and the tokens it added. A cost number is only actionable against a
// declared EXPECTED cost: the budget a breach is defined AGAINST. This file is that
// missing baseline — the "expected" half of the breach predicate.
//
// Two design choices keep it honest:
//
//   - It reuses the closed-vocabulary discipline of the refusal spine
//     (internal/abi/reasons.go + the dos.toml [reasons] table): an over-budget span
//     does not return free-text prose, it names ONE token — OverheadBudgetExceeded
//     ("OVERHEAD_BUDGET_EXCEEDED") — so a breach is emittable, verifiable, and
//     refusable the same way a deny is. The token is declared in dos.toml so
//     `dos check-reason OVERHEAD_BUDGET_EXCEEDED` resolves it as a real member of the
//     vocabulary rather than UNCLASSIFIED prose-drift.
//
//   - A budget is an ENVELOPE WITH A STATED SCOPE, not a promise of zero cost (the
//     epic's explicit non-goal: a gate that costs 8% and saves 40% is a net win, and
//     the plane must say so rather than red on the 8% alone). The values below are a
//     v0.1 DECLARED calibration — a generous per-rung ceiling, not a measured
//     benchmark claim. They are deliberately scoped, not tight: the table exists so a
//     GROSS regression (a rung an order of magnitude over its envelope) reads back as
//     a structured breach, while normal jitter stays OK. Tightening a row toward a
//     measured p99 is a follow-on once T1 folds real cost into rungstats.
package turntaxmeter

// OverheadBudgetExceeded is the closed-vocabulary refusal token a budget breach
// names. It MUST stay byte-identical to the dos.toml [reasons.OVERHEAD_BUDGET_EXCEEDED]
// declaration so the same token that a producer stamps is the one `dos check-reason`
// verifies and the deny-loopback routes — never two spellings of one reason.
const OverheadBudgetExceeded = "OVERHEAD_BUDGET_EXCEEDED"

// Span is one observed lifecycle-rung cost sample: how long a single adjudication or
// admission rung took, and how many tokens it ADDED (a transform/quarantine re-emit
// adds tokens; a pure allow/deny decision adds none). It is the unit the budget
// judges — a synthetic over-budget Span is exactly the acceptance witness.
type Span struct {
	// Rung is the rung's canonical self-reported identity — the verdict `By` string
	// (abi.RungName), e.g. "adjudicator", "witness", "secretgate". It is the same key
	// the live fak_gateway_operation_duration_seconds{adjudicator-rung} histogram and
	// the offline rungobs distribution bucket by, so a budget row lines up 1:1 with an
	// observed row.
	Rung string
	// Method is the operation within the rung — the cost-attribution sub-key, e.g.
	// "decide" for the adjudicator's verdict, "confirm" for the witness gate. Empty
	// matches a rung-wide budget row declared with an empty Method.
	Method string
	// ElapsedNS is the measured wall-clock nanoseconds this span took.
	ElapsedNS int64
	// TokenDelta is the NET tokens this span added (positive) — transform/quarantine
	// re-emission. A pure decision rung adds zero; tokens SAVED (vDSO/radix) are not a
	// per-rung overhead cost and are accounted on the net line, not here.
	TokenDelta int
}

// Budget is one declared per-(Rung,Method) overhead ceiling: the expected upper bound
// on a single span's latency and added tokens. A span over EITHER bound breaches.
type Budget struct {
	Rung   string
	Method string
	// MaxNS is the declared latency envelope in nanoseconds — the ceiling a single
	// span of this rung/method may take before it reads as a regression. It is a
	// scope-stated calibration ceiling, NOT a measured p99 (see the file doc).
	MaxNS int64
	// MaxTokenDelta is the declared per-span token-add envelope. Zero means the rung
	// is expected to add no tokens, so any positive add breaches the token bound.
	MaxTokenDelta int
	// SubprocessBound marks a rung whose cost is dominated by spawning an external
	// process (the witness gate spawns `git`), not by kernel compute. Its envelope is
	// necessarily looser and a breach there should be read as "a slow disk/host", not
	// "a kernel regression" (epic §3). The flag is advisory metadata for a reader; it
	// does not change the breach predicate.
	SubprocessBound bool
}

// defaultBudget is the v0.1 declared envelope, keyed by (rung, method). The rungs are
// the lifecycle cost-attribution boundaries the epic names (§3): the Submit path
// (preflight → gitgate → egressfloor → ifc-sink → adjudicator → witness) and the Reap
// path (ctxmmu → normgate → secretgate → ifc-stamp → recall). Compute-bound rungs get
// a low-microsecond ceiling; the witness gate is subprocess-bound and gets a wide one
// so a slow `git` spawn does not read as a kernel regression. normgate carries a
// non-zero token envelope because normalization may legitimately re-emit a result.
var defaultBudget = budgetTable([]Budget{
	// Submit path — adjudication tax.
	{Rung: "preflight", Method: "check", MaxNS: 5_000, MaxTokenDelta: 0},
	{Rung: "gitgate", Method: "check", MaxNS: 5_000, MaxTokenDelta: 0},
	{Rung: "egressfloor", Method: "check", MaxNS: 5_000, MaxTokenDelta: 0},
	{Rung: "ifc-sink", Method: "check", MaxNS: 5_000, MaxTokenDelta: 0},
	{Rung: "adjudicator", Method: "decide", MaxNS: 5_000, MaxTokenDelta: 0},
	{Rung: "witness", Method: "confirm", MaxNS: 50_000_000, MaxTokenDelta: 0, SubprocessBound: true},
	// Reap path — result-admission tax.
	{Rung: "ctxmmu", Method: "admit", MaxNS: 10_000, MaxTokenDelta: 0},
	{Rung: "normgate", Method: "normalize", MaxNS: 10_000, MaxTokenDelta: 256},
	{Rung: "secretgate", Method: "scan", MaxNS: 10_000, MaxTokenDelta: 0},
	{Rung: "ifc-stamp", Method: "stamp", MaxNS: 5_000, MaxTokenDelta: 0},
	{Rung: "recall", Method: "fold", MaxNS: 20_000, MaxTokenDelta: 0},
	// Dispatch planner — dry-run fan-out/collision pricing before workers launch.
	{Rung: "dispatch", Method: "plan_fanout", MaxNS: 100_000_000, MaxTokenDelta: 0},
})

// budgetKey is the lookup key for the declared table.
type budgetKey struct {
	rung   string
	method string
}

// budgetTable indexes a declared envelope slice by (rung, method) for O(1) lookup. It
// is a package helper so the table stays a flat, readable literal above.
func budgetTable(rows []Budget) map[budgetKey]Budget {
	m := make(map[budgetKey]Budget, len(rows))
	for _, b := range rows {
		m[budgetKey{b.Rung, b.Method}] = b
	}
	return m
}

// DefaultBudget returns the declared overhead envelope for a rung/method, and whether
// one is declared. A caller checks `ok` first: an UNDECLARED rung is not a breach (the
// table is fail-open on coverage — a rung with no envelope yet cannot be "over" one),
// so callers must distinguish "within budget" from "no budget to be within".
func DefaultBudget(rung, method string) (Budget, bool) {
	b, ok := defaultBudget[budgetKey{rung, method}]
	return b, ok
}

// CheckSpan reads a span back against the declared budget. It returns the breach token
// OverheadBudgetExceeded when a budget IS declared for the span's (rung, method) and
// the span exceeds either bound (latency or added tokens); otherwise it returns
// breach=false with an empty reason.
//
// The fail-open contract is deliberate: an undeclared rung returns (false, "") — it is
// NOT a breach, because there is no declared "expected" to be over. This keeps a new,
// un-budgeted rung from spuriously reding the plane; the way to gate it is to ADD its
// row, not to default it to breaching.
func CheckSpan(span Span) (breach bool, reason string) {
	b, ok := DefaultBudget(span.Rung, span.Method)
	if !ok {
		return false, ""
	}
	if span.ElapsedNS > b.MaxNS || span.TokenDelta > b.MaxTokenDelta {
		return true, OverheadBudgetExceeded
	}
	return false, ""
}
