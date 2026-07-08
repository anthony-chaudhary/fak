package relay

import "testing"

// TestNextActionBoundaryYieldsActionAmbiguousFails is rung E4's witness (issue #1883): the
// one-line-next-action extractor derives the SafePoint NextActionExpressible axis from leg
// state with NO model call. The Done condition names two poles — a nameable boundary yields
// a next_action, and an ambiguous state fails — so the table drives both, plus the closed
// causes and the SafePoint hand-off the axis feeds.
func TestNextActionBoundaryYieldsActionAmbiguousFails(t *testing.T) {
	cases := []struct {
		name           string
		state          LegState
		wantExpress    bool
		wantNextAction string // required only when wantExpress
		wantReason     string // required only when !wantExpress
	}{
		{
			name:           "nameable boundary yields the single next action",
			state:          LegState{NextSteps: []string{"Run the E4 witnesses and close #1883."}},
			wantExpress:    true,
			wantNextAction: "Run the E4 witnesses and close #1883.",
		},
		{
			name:           "surrounding whitespace is trimmed, still a boundary",
			state:          LegState{NextSteps: []string{"  Wire ExtractNextAction into the driver.\t"}},
			wantExpress:    true,
			wantNextAction: "Wire ExtractNextAction into the driver.",
		},
		{
			name:           "the same step named twice is one step, not ambiguous",
			state:          LegState{NextSteps: []string{"Push the E4 commit.", "Push the E4 commit."}},
			wantExpress:    true,
			wantNextAction: "Push the E4 commit.",
		},
		{
			name:        "ambiguous: two competing next steps cannot name a single one",
			state:       LegState{NextSteps: []string{"Ship E4 now.", "Or first land E3."}},
			wantExpress: false,
			wantReason:  ReasonAmbiguousNextAction,
		},
		{
			name:        "no next step named at all is mid-thought",
			state:       LegState{},
			wantExpress: false,
			wantReason:  ReasonNoNextAction,
		},
		{
			name:        "only blank candidates name nothing baton-expressible",
			state:       LegState{NextSteps: []string{"", "   ", "\t"}},
			wantExpress: false,
			wantReason:  ReasonNoNextAction,
		},
		{
			name:        "a multi-line recap is a summary, not a one-line next action",
			state:       LegState{NextSteps: []string{"Did X.\nDid Y.\nNow do Z."}},
			wantExpress: false,
			wantReason:  ReasonNoNextAction,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractNextAction(tc.state)
			if got.Expressible != tc.wantExpress {
				t.Fatalf("ExtractNextAction(%+v).Expressible = %v, want %v (reason %q)", tc.state, got.Expressible, tc.wantExpress, got.Reason)
			}
			if tc.wantExpress {
				if got.NextAction != tc.wantNextAction {
					t.Fatalf("NextAction = %q, want %q", got.NextAction, tc.wantNextAction)
				}
				if got.Reason != "" {
					t.Fatalf("expressible verdict carried a reason %q; want empty", got.Reason)
				}
			} else {
				if got.Reason != tc.wantReason {
					t.Fatalf("failed-axis reason = %q, want %q", got.Reason, tc.wantReason)
				}
				if got.NextAction != "" {
					t.Fatalf("non-expressible verdict carried NextAction %q; want empty", got.NextAction)
				}
			}
		})
	}
}

// TestNextActionFeedsSafePointAxis proves the extractor's Expressible verdict is exactly the
// third SafePoint axis (issue #1880): a nameable boundary lifts an otherwise-safe point to
// IsSafe, and an ambiguous state holds it unsafe on the NextActionExpressible axis alone.
func TestNextActionFeedsSafePointAxis(t *testing.T) {
	// Otherwise-safe: no in-flight turn and a green/parked tree; only the next-action axis
	// is left to the extractor to derive.
	base := SafePoint{NoInFlightTurn: true, TreeGreenOrParked: true}

	boundary := ExtractNextAction(LegState{NextSteps: []string{"Close #1883."}})
	base.NextActionExpressible = boundary.Expressible
	if !base.IsSafe() {
		t.Fatalf("boundary next action did not lift the point to safe: %+v", base)
	}

	ambiguous := ExtractNextAction(LegState{NextSteps: []string{"Do A.", "Do B."}})
	base.NextActionExpressible = ambiguous.Expressible
	if base.IsSafe() {
		t.Fatalf("ambiguous next action left the point safe: %+v (reason %q)", base, ambiguous.Reason)
	}
}
