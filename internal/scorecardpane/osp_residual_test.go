package scorecardpane

// osp_residual_test.go — the #5022 card's own witnesses.
//
// The card folds "how many forming units owe an operator attention" into the
// control pane so it TRENDS instead of living only in an ad-hoc `fak steer prs`
// run. Two properties have to hold for that to be honest:
//
//  1. the debt integer is the RESIDUAL count read from the live fak.steerpr.v1
//     payload (not a re-derivation, and not some neighbouring count), and
//  2. a payload that cannot be READ folds to no number at all — never a clean 0.
//
// (2) is the whole registry's declared-vs-measured property: an unread member
// must not fold as clean, and a 0 on a broken read is precisely the bug the
// fence exists to prevent.

import "testing"

// ospResidualCard returns the registered card, failing if it drifted out of Cards.
func ospResidualCard(t *testing.T) Card {
	t.Helper()
	for _, c := range Cards {
		if c.Key == "osp_residual" {
			return c
		}
	}
	t.Fatal("osp_residual is not registered in Cards — the control pane cannot baseline a card it does not fold")
	return Card{}
}

// steerPRsPayload is a fak.steerpr.v1-shaped view with the fields the card reads.
// The residual count sits at the TOP level of that schema, beside the other
// posted counts, which is the shape `fak steer prs --json` emits.
func steerPRsPayload(residual int) map[string]any {
	return map[string]any{
		"schema":          "fak.steerpr.v1",
		"range":           "origin/release..main",
		"commit_count":    float64(9),
		"unit_count":      float64(4),
		"residual_count":  float64(residual),
		"unstamped_count": float64(1),
	}
}

// TestOSPResidualCardIsWiredToTheSteerPRsPayload pins the card's binding: it reads
// the residual count from the `fak steer prs --json` verb, under a key the pane can
// pin per-metric.
func TestOSPResidualCardIsWiredToTheSteerPRsPayload(t *testing.T) {
	card := ospResidualCard(t)
	if card.Debt != "residual_count" {
		t.Errorf("osp_residual debt key = %q, want %q — debt IS the count of RESIDUAL units", card.Debt, "residual_count")
	}
	if card.Cmd != "go run ./cmd/fak steer prs --json" {
		t.Errorf("osp_residual cmd = %q, want the live steer-prs verb — the card must read the real payload, never a second oracle", card.Cmd)
	}
	if card.Script != "" {
		t.Errorf("osp_residual script = %q, want empty: this is a Go-backed card", card.Script)
	}
	if len(card.Corpus) != 0 {
		t.Errorf("osp_residual Corpus = %v, want empty — the residual pile is a function of git history and the "+
			"witness ledger, not of tracked tree files, so a --since fold must never CARRY a stale count", card.Corpus)
	}
	if !goBackedKey(card.Key) {
		t.Error("osp_residual must be classed Go-backed so a build break is triaged as a build break, not a card bug")
	}
}

// TestOSPResidualFoldsTheResidualCountAsDebt is the measured-value witness: a real
// payload yields the residual count as this card's debt, and it sums into the
// portfolio total like any other card.
func TestOSPResidualFoldsTheResidualCountAsDebt(t *testing.T) {
	card := ospResidualCard(t)
	m := MetricFromPayload(card, steerPRsPayload(3), "")
	if m.Debt == nil {
		t.Fatalf("debt is nil for a readable payload: %+v", m)
	}
	if *m.Debt != 3 {
		t.Errorf("debt = %d, want 3 — the count of RESIDUAL units", *m.Debt)
	}
	if m.Error != "" {
		t.Errorf("error = %q, want empty for a readable payload", m.Error)
	}
	p := Fold([]Metric{m}, nil, "ws", "abc123")
	if p.TotalDebt != 3 {
		t.Errorf("portfolio total = %d, want the card's 3 folded in", p.TotalDebt)
	}
	if p.Measured != 1 || p.Errored != 0 {
		t.Errorf("measured/errored = %d/%d, want 1/0", p.Measured, p.Errored)
	}
}

// TestOSPResidualCleanOverlayHoldsAtZero pins the debt semantics the ticket declares:
// a clean overlay (nothing the kernel could not witness) holds the card at 0, and that
// zero is a MEASURED zero — distinguishable from an unread card, which carries no
// number at all.
func TestOSPResidualCleanOverlayHoldsAtZero(t *testing.T) {
	card := ospResidualCard(t)
	m := MetricFromPayload(card, steerPRsPayload(0), "")
	if m.Debt == nil || *m.Debt != 0 {
		t.Fatalf("clean overlay debt = %v, want a measured 0", m.Debt)
	}
	if got := displayGrade(m); got != "A" {
		t.Errorf("clean overlay grade = %q, want A", got)
	}
	if b := BaselineDoc(Fold([]Metric{m}, nil, "ws", "c")); b.Metrics["osp_residual"] != 0 {
		t.Errorf("a measured 0 must PIN as 0, got %v", b.Metrics)
	}
}

// TestOSPResidualUnreadableIsUnmeasuredNeverZero is the honesty fence the acceptance
// gate names. Break the payload source — an unresolvable base/head ref makes `fak steer
// prs` exit non-zero with no JSON on stdout, which reaches the fold as an error row —
// and the card carries NO debt integer: it errors, it is EXCLUDED from the pinned
// baseline, and the pane reports it unmeasured. A 0 here would tell an operator the
// residual pile is clean when in truth nobody read it.
func TestOSPResidualUnreadableIsUnmeasuredNeverZero(t *testing.T) {
	card := ospResidualCard(t)
	m := MetricFromPayload(card, nil, "non-JSON output (exit 1): fak steer prs: resolve base: unknown revision")
	if m.Debt != nil {
		t.Fatalf("unreadable payload folded debt %d — an unread card must carry NO number, never a clean 0", *m.Debt)
	}
	if m.Verdict != "ERROR" {
		t.Errorf("verdict = %q, want ERROR", m.Verdict)
	}

	p := Fold([]Metric{m}, nil, "ws", "abc123")
	if p.Errored != 1 || p.Measured != 0 {
		t.Errorf("errored/measured = %d/%d, want 1/0", p.Errored, p.Measured)
	}
	if p.TotalDebt != 0 || p.OK {
		t.Errorf("an unmeasured card must block a clean fold: ok=%v total=%d", p.OK, p.TotalDebt)
	}
	if p.Finding != "scorecard_unmeasured" {
		t.Errorf("finding = %q, want scorecard_unmeasured", p.Finding)
	}

	// The load-bearing half: an errored card is NOT pinned. The superloop walk reads
	// its scorecard members out of this baseline, so an absent key is what makes the
	// walk say UNMEASURED instead of reporting a fabricated zero.
	base := BaselineDoc(p)
	if _, pinned := base.Metrics["osp_residual"]; pinned {
		t.Error("an errored osp_residual was pinned into the baseline — the walk would then read a fabricated number")
	}

	// And the ratchet reds rather than silently shrinking the portfolio.
	if code, _ := CheckGate(p); code != 1 {
		t.Errorf("CheckGate exit = %d, want 1 on an unmeasured card", code)
	}
}
