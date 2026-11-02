package policy

import "testing"

// TestControlDecisionTokensAreStable pins the closed control-decision vocabulary
// to its wire-stable token strings and turn/session semantics. The companion
// session package keeps a parallel vocabulary (session.SessionControlDecision);
// the two are aligned by convention at the harness seam, and this test anchors
// the policy side to absolute values so a rename on either side fails loudly
// here instead of drifting silently through the string seam.
func TestControlDecisionTokensAreStable(t *testing.T) {
	cases := []struct {
		d            ControlDecision
		token        string
		endsTurn     bool
		stopsSession bool
	}{
		{ControlContinue, "CONTINUE", false, false},
		{ControlEndTurn, "END_TURN", true, false},
		{ControlPauseSession, "PAUSE_SESSION", true, false},
		{ControlStopSession, "STOP_SESSION", true, true},
	}
	for _, tc := range cases {
		if got := tc.d.String(); got != tc.token {
			t.Fatalf("%v.String() = %q, want %q", tc.d, got, tc.token)
		}
		if got := tc.d.EndsTurn(); got != tc.endsTurn {
			t.Fatalf("%v.EndsTurn() = %v, want %v", tc.d, got, tc.endsTurn)
		}
		if got := tc.d.StopsSession(); got != tc.stopsSession {
			t.Fatalf("%v.StopsSession() = %v, want %v", tc.d, got, tc.stopsSession)
		}
	}
}
