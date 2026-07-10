package resume

import "testing"

// TestFoldRecoveryCost pins the attribution boundary (mirrors NewTurnsAfter): only turns
// strictly after the launch are charged; no launch (sinceUnix <= 0) charges nothing; bad
// negative readings never lower the sum.
func TestFoldRecoveryCost(t *testing.T) {
	turns := []RecoveryTurnCost{
		{UnixSeconds: 90, Tokens: 1000, CostMicroUSD: 500},   // before launch — not charged
		{UnixSeconds: 100, Tokens: 2000, CostMicroUSD: 900},  // exactly at launch — not charged
		{UnixSeconds: 150, Tokens: 3000, CostMicroUSD: 1200}, // after — charged
		{UnixSeconds: 200, Tokens: 4000, CostMicroUSD: 1500}, // after — charged
		{UnixSeconds: 250, Tokens: -9, CostMicroUSD: -9},     // after but bad reading — ignored, but counts as a turn
	}
	got := FoldRecoveryCost(turns, 100)
	if got.Turns != 3 || got.Tokens != 7000 || got.CostMicroUSD != 2700 {
		t.Fatalf("FoldRecoveryCost = %+v, want Turns=3 Tokens=7000 Cost=2700", got)
	}
	if none := FoldRecoveryCost(turns, 0); none != (RecoveryCost{}) {
		t.Fatalf("no launch boundary = %+v, want zero", none)
	}
	if empty := FoldRecoveryCost(nil, 100); empty != (RecoveryCost{}) {
		t.Fatalf("nil turns = %+v, want zero", empty)
	}
}

// TestRecoveryCostCapExceeded: a zero cap gates nothing (fail-open default); a bounded cap
// breaches strictly above the ceiling and emits the closed RESUME_COST_EXCEEDED reason.
func TestRecoveryCostCapExceeded(t *testing.T) {
	rc := RecoveryCost{Turns: 5, Tokens: 12_000, CostMicroUSD: 8_000}

	if ok, _ := (RecoveryCostCap{}).Exceeded(rc); ok {
		t.Fatal("all-zero cap must gate nothing (fail-open default)")
	}
	// Exactly at the ceiling is allowed; strictly above breaches.
	if ok, _ := (RecoveryCostCap{MaxTokens: 12_000}).Exceeded(rc); ok {
		t.Fatal("spend exactly at the token ceiling must be allowed")
	}
	ok, reason := (RecoveryCostCap{MaxTokens: 10_000}).Exceeded(rc)
	if !ok || reason != ResumeCostExceeded {
		t.Fatalf("token breach = (%v,%q), want (true,%q)", ok, reason, ResumeCostExceeded)
	}
	okC, reasonC := (RecoveryCostCap{MaxCostMicroUSD: 5_000}).Exceeded(rc)
	if !okC || reasonC != ResumeCostExceeded {
		t.Fatalf("cost breach = (%v,%q), want (true,%q)", okC, reasonC, ResumeCostExceeded)
	}
}

// TestFoldNextActionRecoveryCostHold: a fire-eligible session over its recovery-cost cap
// tops out at a REVERSIBLE hold_admission carrying RESUME_COST_EXCEEDED — and an operator
// force bit pushes the same launch through. It never turns into a kill or a gave_up.
func TestFoldNextActionRecoveryCostHold(t *testing.T) {
	base := NextInput{
		State:                ResumePending,
		Outcome:              OutcomeUnknown,
		Retry:                RetryDecision{Blocked: false, Reason: "first resume"},
		Admitted:             true,
		RecoveryCostBreached: true,
	}
	hold := FoldNextAction(base)
	if hold.Action != ActHoldAdmission || hold.Fire {
		t.Fatalf("cost-breached = %q fire=%v, want hold_admission/false (%s)", hold.Action, hold.Fire, hold.Reason)
	}
	if !containsToken(hold.Reason, ResumeCostExceeded) {
		t.Fatalf("hold reason %q must carry the closed reason %q", hold.Reason, ResumeCostExceeded)
	}

	forced := base
	forced.ForceCostOverride = true
	if v := FoldNextAction(forced); v.Action != ActRun || !v.Fire {
		t.Fatalf("operator force = %q fire=%v, want run/true (%s)", v.Action, v.Fire, v.Reason)
	}
}

func containsToken(s, tok string) bool {
	for i := 0; i+len(tok) <= len(s); i++ {
		if s[i:i+len(tok)] == tok {
			return true
		}
	}
	return false
}
