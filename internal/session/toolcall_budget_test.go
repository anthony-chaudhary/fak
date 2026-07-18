package session

import "testing"

// TestDebitToolCallExhaustsAndDrains proves the runaway floor (#2887): a bounded
// tool-call budget permits exactly N dispatched calls, then the (N+1)th is refused
// with the closed BUDGET_TOOLCALLS_EXHAUSTED reason and the session is driven to
// Stopped so the exhaustion is observable in the drive state, not re-derived.
func TestDebitToolCallExhaustsAndDrains(t *testing.T) {
	tbl := NewTable()
	const trace = "tc-1"
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ToolCallsLeft: 3})

	for i := 1; i <= 3; i++ {
		if v := tbl.DebitToolCall(trace); !v.Proceed {
			t.Fatalf("call %d refused early: reason=%q", i, v.Reason)
		}
	}
	v := tbl.DebitToolCall(trace)
	if v.Proceed || !v.Stop {
		t.Fatalf("4th call verdict = %+v, want proceed=false stop=true", v)
	}
	if v.Reason != ReasonBudgetToolCalls {
		t.Fatalf("reason = %q, want %s", v.Reason, ReasonBudgetToolCalls)
	}
	if st := tbl.Get(trace); st.Run != Stopped {
		t.Fatalf("session Run=%v after exhaustion, want Stopped", st.Run)
	}
}

// TestDebitToolCallUnboundedIsPermissive guards the zero-value default: a Budget that
// never configures the tool-call axis proceeds forever (the historical loop).
func TestDebitToolCallUnboundedIsPermissive(t *testing.T) {
	tbl := NewTable()
	const trace = "tc-off"
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded})
	for i := 0; i < 100; i++ {
		if v := tbl.DebitToolCall(trace); !v.Proceed {
			t.Fatalf("unbounded tool-call axis refused call %d: %q", i, v.Reason)
		}
	}
}

// TestDebitToolCallNilTablePermissive confirms the nil-receiver no-op so a loop with
// no session table behaves byte-identically to the pre-axis path.
func TestDebitToolCallNilTablePermissive(t *testing.T) {
	var tbl *Table
	if v := tbl.DebitToolCall("x"); !v.Proceed {
		t.Fatalf("nil table refused: %+v", v)
	}
}

// TestParseBudgetEnvelopeToolCalls proves the CLI budget spec carries the tool-call
// axis alongside wall-clock, so a scheduled/dispatched run is configured with one
// `--budget "wall=3m,calls=50"` string (#2887's "wall-clock + tool-call budget").
func TestParseBudgetEnvelopeToolCalls(t *testing.T) {
	env, err := ParseBudgetEnvelope("calls=50,wall=3m")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	if env.Budget.ToolCallsLeft != 50 {
		t.Fatalf("ToolCallsLeft = %d, want 50", env.Budget.ToolCallsLeft)
	}
	// SessionBudget stamps the cap so "0 left, positive cap = exhausted" survives.
	if b := env.SessionBudget(); b.ToolCallsCap != 50 {
		t.Fatalf("ToolCallsCap = %d, want 50 stamped from remaining", b.ToolCallsCap)
	}
}
