package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestTurnUseBaselineGolden pins the agent loop's turn-use projection against the
// checked-in golden — the #2001 M1 acceptance witness ("the bench still produces
// identical turn-use numbers"). If the internal/microagent extraction changes how
// the loop threads context, sequences tool calls, or accounts turns, this fails.
// Regenerate with UPDATE_GOLDEN=1 only when a turn-use change is intended.
func TestTurnUseBaselineGolden(t *testing.T) {
	b, m, err := MeasureTurnUse(context.Background(), 8)
	if err != nil {
		t.Fatalf("MeasureTurnUse: %v", err)
	}

	// Loop-scoped invariants that must hold regardless of the golden's exact values.
	if b.HitTurnCap || !b.FinalAnswerReached {
		t.Fatalf("session did not finish cleanly: hit_turn_cap=%v final_answer_reached=%v",
			b.HitTurnCap, b.FinalAnswerReached)
	}
	if b.Turns != len(b.TurnLog) {
		t.Fatalf("metrics turns=%d != recorded turns=%d (loop accounting and the planner's view must agree)",
			b.Turns, len(b.TurnLog))
	}
	if n := len(b.TurnLog); n == 0 || !b.TurnLog[n-1].Final {
		t.Fatalf("last recorded turn must be the final answer, got %+v", b.TurnLog)
	}
	if b.ToolErrors != 0 {
		t.Errorf("tool_errors=%d, want 0 (the fak arm repairs in-syscall)", b.ToolErrors)
	}
	// Non-pinned sanity on the raw metrics: the duplicate get_user re-verify must
	// dedup on the fak arm. The exact count belongs to the vdso lane, so it is
	// asserted >0 here, never pinned in the golden.
	if m.VDSOHits <= 0 {
		t.Errorf("vdso_hits=%d, want >0 (intra-session dedup on the duplicate read)", m.VDSOHits)
	}

	got, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		t.Fatalf("marshal baseline: %v", err)
	}

	// Golden: the committed turn-use artifact. Regenerate with UPDATE_GOLDEN=1.
	golden := filepath.Join("testdata", "turnuse_baseline.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("turn-use projection drifted from golden %s — the loop's turn accounting, "+
			"tool sequence, or context threading changed; re-run with UPDATE_GOLDEN=1 ONLY if intended:\n got: %s\nwant: %s",
			golden, got, want)
	}
}

// TestTurnUseBaselineDeterministic re-measures the projection twice and asserts
// bit-identity — the property that makes the golden pin meaningful (a flaky pin
// would be noise, not a witness).
func TestTurnUseBaselineDeterministic(t *testing.T) {
	a, _, err := MeasureTurnUse(context.Background(), 8)
	if err != nil {
		t.Fatalf("MeasureTurnUse (first): %v", err)
	}
	b, _, err := MeasureTurnUse(context.Background(), 8)
	if err != nil {
		t.Fatalf("MeasureTurnUse (second): %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Errorf("turn-use projection not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}
