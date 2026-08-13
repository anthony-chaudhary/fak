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

// TestVDSOServeBudgetDeclaredAndBreaches is the acceptance witness for issue #2219
// (epic #2218, gap G1): the hottest local-serve rung — the vDSO hit on s.syscall — must
// carry a DECLARED envelope so a pure-Go regression on it reads back as a structured
// breach instead of the fail-open silence an undeclared rung gets. It proves the row
// exists, that an over-budget vdso.serve span reaches OVERHEAD_BUDGET_EXCEEDED (the
// "reachable from a vdso span" acceptance), and that a serve at the 3.4 us serve anchor
// stays comfortably OK — so the gate catches a gross regression without flagging normal
// serves.
func TestVDSOServeBudgetDeclaredAndBreaches(t *testing.T) {
	b, ok := DefaultBudget("vdso", "serve")
	if !ok {
		t.Fatalf("issue #2219: no declared budget for the vdso/serve rung — the hottest " +
			"serve path is still fail-open (an undeclared rung can never breach)")
	}
	// A generous gross-regression ceiling, not a tight p99: it must sit well above the
	// ~3.4 us serve anchor so normal serve jitter never reds, while still catching an
	// order-of-magnitude regression.
	if b.MaxNS < 3_400 {
		t.Fatalf("vdso.serve envelope %d ns is below the ~3.4 us serve anchor — it would "+
			"red on a normal serve", b.MaxNS)
	}
	// A serve adds no tokens (it SAVES them), so any positive add is a breach.
	if b.MaxTokenDelta != 0 {
		t.Fatalf("vdso.serve token envelope = %d, want 0 (a serve that ADDS tokens is a bug)", b.MaxTokenDelta)
	}

	cases := []struct {
		name       string
		span       Span
		wantBreach bool
	}{
		{
			name:       "an order-of-magnitude serve regression breaches from a vdso span",
			span:       Span{Rung: "vdso", Method: "serve", ElapsedNS: b.MaxNS * 10},
			wantBreach: true,
		},
		{
			name:       "a serve that adds tokens breaches the zero token envelope",
			span:       Span{Rung: "vdso", Method: "serve", ElapsedNS: 3_400, TokenDelta: 1},
			wantBreach: true,
		},
		{
			name:       "a serve at the ~3.4 us anchor is well within budget",
			span:       Span{Rung: "vdso", Method: "serve", ElapsedNS: 3_459},
			wantBreach: false,
		},
		{
			name:       "exactly at the ceiling is OK (breach is strictly over)",
			span:       Span{Rung: "vdso", Method: "serve", ElapsedNS: b.MaxNS},
			wantBreach: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			breach, reason := CheckSpan(tc.span)
			if breach != tc.wantBreach {
				t.Fatalf("CheckSpan(%+v): breach=%v, want %v (reason=%q)", tc.span, breach, tc.wantBreach, reason)
			}
			if breach && reason != OverheadBudgetExceeded {
				t.Fatalf("a vdso.serve breach must name %q, got %q", OverheadBudgetExceeded, reason)
			}
		})
	}
}

// TestBudgetTokenIsStable pins the breach token's spelling: it is the contract the
// dos.toml [reasons.OVERHEAD_BUDGET_EXCEEDED] declaration and `dos man wedge <TOKEN> --explain` rely
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
