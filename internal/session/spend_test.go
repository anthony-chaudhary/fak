package session

import "testing"

// spend_test.go — the spend ceiling as a LIVE budget axis (#1573's spend axis,
// enforcement). The envelope's spend=$N was previously a parsed inspectable
// contract field only; these tests pin that it now projects onto
// Budget.SpendMicroCentsLeft, that DebitUsage debits priced turns toward it,
// that crossing drains the session with the closed BUDGET_SPEND_EXHAUSTED
// reason WITHOUT minting a fresh-window continuation, and that an unpriced
// (dollar-blind) turn never debits a guessed cost.

// TestBudgetEnvelopeSpendProjectsToSessionBudget: spend=$25 becomes a real
// runtime axis, in exact micro-cents, with the cap stamped as the denominator.
func TestBudgetEnvelopeSpendProjectsToSessionBudget(t *testing.T) {
	env, err := ParseBudgetEnvelope("spend=$25")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	b := env.SessionBudget()
	want := int64(2500) * MicroCentsPerCent // $25 = 2500 cents
	if b.SpendMicroCentsLeft != want {
		t.Fatalf("SpendMicroCentsLeft = %d, want %d", b.SpendMicroCentsLeft, want)
	}
	if b.SpendMicroCentsCap != want {
		t.Fatalf("SpendMicroCentsCap = %d, want %d (stamped denominator)", b.SpendMicroCentsCap, want)
	}
}

// TestBudgetEnvelopeWithoutSpendStaysUnconfigured: an envelope that never
// mentions spend must not grow a surprise spend budget.
func TestBudgetEnvelopeWithoutSpendStaysUnconfigured(t *testing.T) {
	env, err := ParseBudgetEnvelope("turns=5")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	b := env.SessionBudget()
	if b.SpendMicroCentsLeft != 0 || b.SpendMicroCentsCap != 0 {
		t.Fatalf("unstated spend axis = (%d,%d), want (0,0)", b.SpendMicroCentsLeft, b.SpendMicroCentsCap)
	}
}

// TestDebitUsageSpendDebitsAndDrains: priced turns debit the axis; the turn
// that crosses the ceiling drains the session immediately with the closed
// spend reason, no continuation id (a spent dollar cap is terminal, never a
// fresh-window continuation), and the next Decide takes the stop.
func TestDebitUsageSpendDebitsAndDrains(t *testing.T) {
	tbl := NewTable()
	const trace = "spend-drain"
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, SpendMicroCentsLeft: 1000})

	st := tbl.DebitUsage(trace, Usage{OutputTokens: 10, CostMicroCents: 400})
	if st.Run != Running {
		t.Fatalf("after first debit Run = %v, want Running", st.Run)
	}
	if st.Budget.SpendMicroCentsLeft != 600 {
		t.Fatalf("after first debit SpendMicroCentsLeft = %d, want 600", st.Budget.SpendMicroCentsLeft)
	}

	st = tbl.DebitUsage(trace, Usage{OutputTokens: 10, CostMicroCents: 700})
	if st.Run != Draining {
		t.Fatalf("after ceiling crossed Run = %v, want Draining", st.Run)
	}
	if st.Reason != ReasonBudgetSpend {
		t.Fatalf("Reason = %q, want %q", st.Reason, ReasonBudgetSpend)
	}
	if st.Budget.SpendMicroCentsLeft != 0 {
		t.Fatalf("drained SpendMicroCentsLeft = %d, want clamped 0", st.Budget.SpendMicroCentsLeft)
	}
	if st.ContinuationID != "" {
		t.Fatalf("spend drain minted continuation %q, want none", st.ContinuationID)
	}

	v := tbl.Decide(trace)
	if v.Proceed || !v.Stop || v.Reason != ReasonBudgetSpend {
		t.Fatalf("Decide after spend drain = {Proceed:%v Stop:%v Reason:%q}, want refuse/stop/%s",
			v.Proceed, v.Stop, v.Reason, ReasonBudgetSpend)
	}
}

// TestDebitUsageUnpricedTurnNeverDebitsSpend: a dollar-blind host reports
// CostMicroCents=0; the configured spend budget must stay untouched rather
// than be debited a guessed cost.
func TestDebitUsageUnpricedTurnNeverDebitsSpend(t *testing.T) {
	tbl := NewTable()
	const trace = "spend-blind"
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, SpendMicroCentsLeft: 1000})
	st := tbl.DebitUsage(trace, Usage{OutputTokens: 50, ContextTokens: 100})
	if st.Budget.SpendMicroCentsLeft != 1000 {
		t.Fatalf("unpriced turn moved spend axis to %d, want 1000 untouched", st.Budget.SpendMicroCentsLeft)
	}
	if st.Run != Running {
		t.Fatalf("Run = %v, want Running", st.Run)
	}
}

// TestDebitUsageNoSpendBudgetIgnoresCost: a priced turn against a session with
// no spend budget configured is a no-op on the spend axis (0 = not configured,
// matching the context axis convention).
func TestDebitUsageNoSpendBudgetIgnoresCost(t *testing.T) {
	tbl := NewTable()
	const trace = "spend-unconfigured"
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded})
	st := tbl.DebitUsage(trace, Usage{OutputTokens: 5, CostMicroCents: 999_999})
	if st.Run != Running || st.Reason != "" {
		t.Fatalf("cost against unconfigured spend axis = {Run:%v Reason:%q}, want Running/\"\"", st.Run, st.Reason)
	}
}

// TestSpendDrainWinsOverSameTurnContextDrain: when ONE turn crosses both the
// spend and the context ceiling, money wins — the session drains with the
// spend reason and NO continuation id, so the fresh-window reset path (which
// keys on the context drain's continuation) cannot continue past a spent cap.
func TestSpendDrainWinsOverSameTurnContextDrain(t *testing.T) {
	tbl := NewTable()
	const trace = "spend-vs-context"
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 100, SpendMicroCentsLeft: 100})
	st := tbl.DebitUsage(trace, Usage{OutputTokens: 10, ContextTokens: 200, CostMicroCents: 200})
	if st.Run != Draining || st.Reason != ReasonBudgetSpend {
		t.Fatalf("double drain = {Run:%v Reason:%q}, want Draining/%s", st.Run, st.Reason, ReasonBudgetSpend)
	}
	if st.ContinuationID != "" {
		t.Fatalf("double drain minted continuation %q, want none (spend wins)", st.ContinuationID)
	}
	if st.Budget.ContextTokensLeft != 0 {
		t.Fatalf("context accounting after double drain = %d, want 0 (still debited)", st.Budget.ContextTokensLeft)
	}
}

// TestRecontinueCarriesSpendForward: the spend ceiling is a lineage budget like
// the wall clock — a hidden context reset must not zero a dollar cap. A fresh
// budget with no spend opinion inherits the parent's remaining allotment and
// cap; an explicit fresh spend axis wins.
func TestRecontinueCarriesSpendForward(t *testing.T) {
	tbl := NewTable()
	tbl.SetBudget("parent", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, SpendMicroCentsLeft: 1000})
	tbl.DebitUsage("parent", Usage{OutputTokens: 1, CostMicroCents: 250})

	child := tbl.Recontinue("parent", "child", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 500})
	if child.Budget.SpendMicroCentsLeft != 750 {
		t.Fatalf("child SpendMicroCentsLeft = %d, want 750 carried forward", child.Budget.SpendMicroCentsLeft)
	}
	if child.Budget.SpendMicroCentsCap != 1000 {
		t.Fatalf("child SpendMicroCentsCap = %d, want 1000 carried forward", child.Budget.SpendMicroCentsCap)
	}

	rearmed := tbl.Recontinue("child", "grandchild", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, SpendMicroCentsLeft: 5000})
	if rearmed.Budget.SpendMicroCentsLeft != 5000 || rearmed.Budget.SpendMicroCentsCap != 5000 {
		t.Fatalf("explicit re-arm = (%d,%d), want (5000,5000) — caller wins",
			rearmed.Budget.SpendMicroCentsLeft, rearmed.Budget.SpendMicroCentsCap)
	}
}
