package policy

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestControlDecisionTokensMatchSessionControl(t *testing.T) {
	cases := []struct {
		policy  ControlDecision
		session session.SessionControlDecision
	}{
		{ControlContinue, session.SessionControlContinue},
		{ControlEndTurn, session.SessionControlEndTurn},
		{ControlPauseSession, session.SessionControlPause},
		{ControlStopSession, session.SessionControlStop},
	}
	for _, tc := range cases {
		if tc.policy.String() != tc.session.String() {
			t.Fatalf("policy token %s != session token %s", tc.policy.String(), tc.session.String())
		}
		if tc.policy.EndsTurn() != tc.session.EndsTurn() {
			t.Fatalf("%s EndsTurn mismatch: policy=%v session=%v", tc.policy, tc.policy.EndsTurn(), tc.session.EndsTurn())
		}
		if tc.policy.StopsSession() != tc.session.StopsSession() {
			t.Fatalf("%s StopsSession mismatch: policy=%v session=%v", tc.policy, tc.policy.StopsSession(), tc.session.StopsSession())
		}
	}
}
