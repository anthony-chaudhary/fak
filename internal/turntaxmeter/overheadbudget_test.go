package turntaxmeter

import "testing"

// TestOverBudgetSpanReadsBackAsBreach is the acceptance witness for issue #1150: a
// synthetic over-budget span must read back as a breach naming the closed-vocabulary
// token OVERHEAD_BUDGET_EXCEEDED. It exercises both bounds (latency and added tokens)
// and proves the fail-open contract (within budget, and an undeclared rung, are NOT
// breaches) so a passing test cannot be a tautology that flags everything.
func TestOverBudgetSpanReadsBackAsBreach(t *testing.T) {
	// The declared adjudicator/decide envelope is 5_000 ns, 0 added tokens.
	b, ok := DefaultBudget("adjudicator", "decide")
	if !ok {
		t.Fatalf("expected a declared budget for adjudicator/decide")
	}

	cases := []struct {
		name       string
		span       Span
		wantBreach bool
	}{
		{
			name:       "over the latency envelope breaches",
			span:       Span{Rung: "adjudicator", Method: "decide", ElapsedNS: b.MaxNS + 1},
			wantBreach: true,
		},
		{
			name:       "over the token envelope breaches",
			span:       Span{Rung: "adjudicator", Method: "decide", ElapsedNS: 100, TokenDelta: b.MaxTokenDelta + 1},
			wantBreach: true,
		},
		{
			name:       "within both bounds is OK",
			span:       Span{Rung: "adjudicator", Method: "decide", ElapsedNS: b.MaxNS, TokenDelta: b.MaxTokenDelta},
			wantBreach: false,
		},
		{
			name:       "exactly at the latency ceiling is OK (breach is strictly over)",
			span:       Span{Rung: "adjudicator", Method: "decide", ElapsedNS: b.MaxNS},
			wantBreach: false,
		},
		{
			name:       "an undeclared rung is fail-open, not a breach",
			span:       Span{Rung: "no-such-rung", Method: "nope", ElapsedNS: 1 << 40, TokenDelta: 1 << 20},
			wantBreach: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			breach, reason := CheckSpan(tc.span)
			if breach != tc.wantBreach {
				t.Fatalf("CheckSpan(%+v): breach=%v, want %v (reason=%q)", tc.span, breach, tc.wantBreach, reason)
			}
			if breach {
				if reason != OverheadBudgetExceeded {
					t.Fatalf("a breach must name the closed-vocabulary token %q, got %q", OverheadBudgetExceeded, reason)
				}
			} else if reason != "" {
				t.Fatalf("a non-breach must carry no reason, got %q", reason)
			}
		})
	}
}

// TestBudgetTokenIsStable pins the breach token's spelling: it is the contract the
// dos.toml [reasons.OVERHEAD_BUDGET_EXCEEDED] declaration and `dos check-reason` rely
// on, so a rename here that drifts from the vocabulary must fail the build.
func TestBudgetTokenIsStable(t *testing.T) {
	if OverheadBudgetExceeded != "OVERHEAD_BUDGET_EXCEEDED" {
		t.Fatalf("breach token drifted from the closed vocabulary: %q", OverheadBudgetExceeded)
	}
}

// TestWitnessRungIsSubprocessBound guards the one envelope that must stay loose: the
// witness gate spawns `git`, so a normal multi-millisecond spawn must NOT read as a
// kernel regression, and the row must be flagged subprocess-bound for a reader.
func TestWitnessRungIsSubprocessBound(t *testing.T) {
	b, ok := DefaultBudget("witness", "confirm")
	if !ok {
		t.Fatalf("expected a declared budget for the witness rung")
	}
	if !b.SubprocessBound {
		t.Fatalf("the witness rung must be flagged subprocess-bound (it spawns git)")
	}
	// A 5 ms git spawn is normal, not a breach, under the wide subprocess envelope.
	if breach, _ := CheckSpan(Span{Rung: "witness", Method: "confirm", ElapsedNS: 5_000_000}); breach {
		t.Fatalf("a normal git-spawn latency must not breach the subprocess-bound witness envelope")
	}
}
