package demo_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/demo"

	// Blank-import the full leaf registry so the real adjudicator + result-admitter
	// chain is wired before Run folds it — mandatory, exactly as every on-box demo
	// main carries (see the doc comment atop internal/agentdemo). Without it the
	// chains are empty and every verdict fails closed to DEFAULT_DENY.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// TestRunProducesTheThreeRealVerdicts pins the canonical proof: the offline scenario,
// folded through the REAL kernel, yields exactly ALLOW, DENY, QUARANTINE — in that
// order, from the real chains, never a hardcoded string.
func TestRunProducesTheThreeRealVerdicts(t *testing.T) {
	res, err := demo.Run(context.Background())
	if err != nil {
		t.Fatalf("demo.Run: %v", err)
	}
	if len(res.Lines) != 3 {
		t.Fatalf("expected 3 verdict lines, got %d: %+v", len(res.Lines), res.Lines)
	}

	want := []struct {
		step    string
		verdict string
	}{
		{"safe read", demo.VerdictAllow},
		{"destructive call", demo.VerdictDeny},
		{"poisoned result", demo.VerdictQuarantine},
	}
	for i, w := range want {
		got := res.Lines[i]
		if got.Verdict != w.verdict {
			t.Errorf("line %d (%s): verdict = %q, want %q (reason %q, by %q)",
				i, got.Step, got.Verdict, w.verdict, got.Reason, got.By)
		}
		if got.Step != w.step {
			t.Errorf("line %d: step = %q, want %q", i, got.Step, w.step)
		}
	}

	// The refusals must cite a real reason, not an empty/allow token — proof the
	// verdict came from a chain that actually fired.
	if r := res.Lines[1].Reason; r == "" || r == "NONE" {
		t.Errorf("DENY line has no refusal reason: %q", r)
	}
	if r := res.Lines[2].Reason; r == "" || r == "NONE" {
		t.Errorf("QUARANTINE line has no refusal reason: %q", r)
	}
}
