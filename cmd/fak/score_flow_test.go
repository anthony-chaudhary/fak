package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// score_flow_test.go pins the WIRING half of #6198: the fold reaches an operator
// through a `fak score` route and one control-pane row. The payload half — schema,
// eight KPIs, array-typed defects/soft — is witnessed where the hop now lives, in
// internal/flowmetrics/score_test.go, so it runs from the package's own test binary.

// TestScoreFlowRouteIsRegistered pins the route. The #2247 convention is that a
// scorecard lands under `fak score <name>` and never as another top-level verb, so a
// handler reachable only by calling the Go function is the defect this catches.
func TestScoreFlowRouteIsRegistered(t *testing.T) {
	if scoreRoutes["flow"] == nil {
		t.Fatal("`fak score flow` is not registered in scoreRoutes -- internal/flowmetrics " +
			"grades eight KPIs that no command emits, so none of them can change a decision")
	}
}

// TestScoreFlowControlPaneCardIsRegistered pins the second half of the readout: the
// control pane must COLLECT a flow_debt row rather than report the card missing. The
// Corpus stays empty on purpose -- the KPIs are a function of git history plus issue
// state, not of tracked tree files, so a diff-disjoint carry on a --since fold would
// replay a stale reading as if it were fresh.
func TestScoreFlowControlPaneCardIsRegistered(t *testing.T) {
	var got *scorecardpane.Card
	for i := range scorecardpane.Cards {
		if scorecardpane.Cards[i].Debt == "flow_debt" {
			got = &scorecardpane.Cards[i]
			break
		}
	}
	if got == nil {
		t.Fatal("no control-pane card carries debt key \"flow_debt\" -- the pane cannot " +
			"baseline a card it does not fold")
	}
	if got.Cmd != "go run ./cmd/fak score flow --json" {
		t.Fatalf("flow card Cmd = %q, want the score-flow route", got.Cmd)
	}
	if len(got.Corpus) != 0 {
		t.Fatalf("flow card Corpus = %v, want empty (always measure): the KPIs read git "+
			"history and issue state, not tracked tree files, so a carry would go stale", got.Corpus)
	}
}
