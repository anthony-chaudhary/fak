package agent

import (
	"context"
	"testing"
)

// TestOfflineABTurnDelta is the deterministic witness that the kernel saves model
// turns AND blocks the error/attack path, using the offline mock planner so the
// result is reproducible with no network. It encodes the core "turn use vs now"
// claim as machine-checked invariants.
func TestOfflineABTurnDelta(t *testing.T) {
	res, _, err := Run(context.Background(), NewMockPlanner("test"), DefaultTask, 12)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.AppVersion == "" {
		t.Fatal("RunResult app_version is empty")
	}

	// Turn savings: the kernel arm completes the SAME task in strictly fewer turns.
	if res.Fak.Turns >= res.Baseline.Turns {
		t.Errorf("expected fak turns < baseline turns; got fak=%d base=%d", res.Fak.Turns, res.Baseline.Turns)
	}
	if res.TurnsSaved <= 0 {
		t.Errorf("expected positive turns saved; got %d", res.TurnsSaved)
	}

	// The two turn savers fired exactly once each.
	if res.Fak.Repairs != 1 {
		t.Errorf("expected 1 in-syscall repair; got %d", res.Fak.Repairs)
	}
	if res.Fak.Quarantines != 1 {
		t.Errorf("expected 1 MMU quarantine; got %d", res.Fak.Quarantines)
	}
	if res.Fak.VDSOHits < 1 {
		t.Errorf("expected >=1 vDSO dedup hit; got %d", res.Fak.VDSOHits)
	}

	// The baseline incurs the retry that the repair eliminated.
	if res.Baseline.ToolErrors < 1 {
		t.Errorf("expected baseline to hit >=1 tool error (the retry); got %d", res.Baseline.ToolErrors)
	}

	// Safety floor: the baseline is derailed by the poisoned result and runs the
	// destructive op; the kernel arm is not and does not.
	if !res.Baseline.InjectionInContext {
		t.Error("expected the baseline to admit the injection into context")
	}
	if res.Fak.InjectionInContext {
		t.Error("expected the kernel arm to keep the injection OUT of context")
	}
	if !res.Baseline.DestructiveExecuted {
		t.Error("expected the baseline to execute the destructive op")
	}
	if res.Fak.DestructiveExecuted {
		t.Error("expected the kernel arm to prevent the destructive op")
	}

	// Both arms must actually COMPLETE the task — otherwise the turn delta is not
	// comparable (a derailed arm "saves" turns by failing). The mock completes both.
	if !res.Fak.TaskCompleted || !res.Baseline.TaskCompleted {
		t.Errorf("expected both arms to complete the task; fak=%v base=%v",
			res.Fak.TaskCompleted, res.Baseline.TaskCompleted)
	}
	if !res.BothCompleted {
		t.Error("expected BothCompleted true so the turn delta is comparable")
	}

	// Tokens: fewer turns => fewer tokens (soft secondary, must still be >= 0).
	if res.TokensSaved < 0 {
		t.Errorf("expected non-negative token savings; got %d", res.TokensSaved)
	}
}

// TestExecToolRejectsMalformed confirms the local tool validates its OWN inputs
// (so the baseline arm's error path is the tool's real contract, not a harness
// artefact): aliased convert args are missing the canonical fields and error.
func TestExecToolRejectsMalformed(t *testing.T) {
	_, isErr := execTool(toolConvert, map[string]any{"from": "USD", "to": "EUR", "amount": 240.0})
	if !isErr {
		t.Error("expected aliased convert_currency args to be rejected by the tool")
	}
	_, isErr = execTool(toolConvert, map[string]any{"from_currency": "USD", "to_currency": "EUR", "amount": 240.0})
	if isErr {
		t.Error("expected canonical convert_currency args to succeed")
	}
}

// TestMockPlannerDeterministic confirms two identical runs produce identical
// turn counts (the offline seam is reproducible).
func TestMockPlannerDeterministic(t *testing.T) {
	a, _, err := Run(context.Background(), NewMockPlanner("t"), DefaultTask, 12)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := Run(context.Background(), NewMockPlanner("t"), DefaultTask, 12)
	if err != nil {
		t.Fatal(err)
	}
	if a.Fak.Turns != b.Fak.Turns || a.Baseline.Turns != b.Baseline.Turns {
		t.Errorf("non-deterministic: run1 fak=%d base=%d, run2 fak=%d base=%d",
			a.Fak.Turns, a.Baseline.Turns, b.Fak.Turns, b.Baseline.Turns)
	}
}
