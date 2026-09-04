package agent

import (
	"context"
	"testing"
	"time"
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

	// Wall-clock (#3113): the offline mock lane has NO real model latency, so it must
	// omit/zero the seconds rather than fabricate them — turns_saved only, the same
	// observed-only, silent-when-untimed rule the guard exit line uses.
	if res.TimeSavedSeconds != 0 {
		t.Errorf("offline lane must not price seconds; got time_saved_seconds=%v", res.TimeSavedSeconds)
	}
	if res.MeanTurnLatencyMs != 0 {
		t.Errorf("offline lane must not report a mean per-turn latency; got %v", res.MeanTurnLatencyMs)
	}
	if res.Fak.ElapsedMs != 0 || res.Baseline.ElapsedMs != 0 {
		t.Errorf("offline lane must leave per-arm wall-clock unset; got fak=%d base=%d",
			res.Fak.ElapsedMs, res.Baseline.ElapsedMs)
	}
	if res.Live {
		t.Error("offline lane must report live=false")
	}
}

// TestPriceTimeSaved is the deterministic witness for the wall-clock pricing (#3113):
// the live lane converts spared round-trips into observed seconds at the fak arm's
// observed mean per-turn latency, while the offline lane and non-comparable runs
// return zero — no fabricated latency.
func TestPriceTimeSaved(t *testing.T) {
	// Live, both arms completed, 2 turns saved: 7 fak turns over 7s => 1000ms/turn,
	// so 2 spared turns price to exactly 2.0 observed seconds.
	mean, secs := priceTimeSaved(true, true, 2, 7, 7*time.Second)
	if mean != 1000 {
		t.Errorf("mean per-turn latency: got %v, want 1000ms", mean)
	}
	if secs != 2.0 {
		t.Errorf("time saved: got %v, want 2.0s", secs)
	}

	// Offline lane (live=false): observed-only rule zeroes BOTH figures.
	if mean, secs := priceTimeSaved(false, true, 2, 7, 7*time.Second); mean != 0 || secs != 0 {
		t.Errorf("offline lane must not fabricate latency; got mean=%v secs=%v", mean, secs)
	}

	// Not comparable (a derailed baseline "saved" turns by failing): a mean latency is
	// still observed, but no seconds are booked.
	if mean, secs := priceTimeSaved(true, false, 3, 5, 5*time.Second); mean != 1000 || secs != 0 {
		t.Errorf("non-comparable run must not book seconds; got mean=%v secs=%v", mean, secs)
	}

	// No turns saved => no seconds, even live.
	if _, secs := priceTimeSaved(true, true, 0, 5, 5*time.Second); secs != 0 {
		t.Errorf("zero spared turns must price to zero seconds; got %v", secs)
	}

	// Guard against divide-by-zero when a live arm somehow logged no turns.
	if mean, secs := priceTimeSaved(true, true, 1, 0, time.Second); mean != 0 || secs != 0 {
		t.Errorf("zero fak turns must return zero; got mean=%v secs=%v", mean, secs)
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
