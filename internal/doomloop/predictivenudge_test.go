package doomloop

import (
	"fmt"
	"strings"
	"testing"
)

func TestPredictiveNudge(t *testing.T) {
	t.Run("BelowThreshold", func(t *testing.T) {
		// With threshold K=4, K/2 is 2. At 1 flat turn, nudge should not trigger.
		nudge := EvaluatePredictiveNudge("refactor-auth", 1, 4, "compiler error")
		if nudge.Triggered {
			t.Fatalf("Triggered = true, want false at 1 flat turn with K=4")
		}
		if nudge.NudgeMessage != "" {
			t.Fatalf("NudgeMessage = %q, want empty string when not triggered", nudge.NudgeMessage)
		}
		if nudge.ActiveObjective != "refactor-auth" {
			t.Fatalf("ActiveObjective = %q, want refactor-auth", nudge.ActiveObjective)
		}
		if nudge.ConsecutiveFlatTurns != 1 {
			t.Fatalf("ConsecutiveFlatTurns = %d, want 1", nudge.ConsecutiveFlatTurns)
		}
		if nudge.ThresholdK != 4 {
			t.Fatalf("ThresholdK = %d, want 4", nudge.ThresholdK)
		}
		if nudge.LastObservedFailure != "compiler error" {
			t.Fatalf("LastObservedFailure = %q, want compiler error", nudge.LastObservedFailure)
		}
		if nudge.FullHaltReached() {
			t.Fatalf("FullHaltReached() = true, want false at 1 turn with K=4")
		}
	})

	t.Run("TriggeredAtKDiv2", func(t *testing.T) {
		// With threshold K=4, K/2 is 2. At 2 flat turns, nudge should trigger.
		nudge := EvaluatePredictiveNudge("fix-tests", 2, 4, "test timeout")
		if !nudge.Triggered {
			t.Fatalf("Triggered = false, want true at 2 flat turns with K=4")
		}
		wantMsg := "Your last 2 attempts showed no forward progress. Step back and inspect the file or error diagnostics before retrying."
		if nudge.NudgeMessage != wantMsg {
			t.Fatalf("NudgeMessage = %q, want %q", nudge.NudgeMessage, wantMsg)
		}
		if nudge.FullHaltReached() {
			t.Fatalf("FullHaltReached() = true, want false before K=4")
		}
	})

	t.Run("TriggeredBeforeFullHalt", func(t *testing.T) {
		// With threshold K=4, at 3 flat turns (between K/2 and K), nudge is still active.
		nudge := EvaluatePredictiveNudge("fix-tests", 3, 4, "syntax error")
		if !nudge.Triggered {
			t.Fatalf("Triggered = false, want true at 3 flat turns with K=4")
		}
		wantMsg := "Your last 3 attempts showed no forward progress. Step back and inspect the file or error diagnostics before retrying."
		if nudge.NudgeMessage != wantMsg {
			t.Fatalf("NudgeMessage = %q, want %q", nudge.NudgeMessage, wantMsg)
		}
		if nudge.FullHaltReached() {
			t.Fatalf("FullHaltReached() = true, want false at 3 turns with K=4")
		}
	})

	t.Run("FullHaltBoundary", func(t *testing.T) {
		// With threshold K=4, at 4 flat turns full doom-loop halt is reached;
		// predictive nudge is no longer active (before reaching full halt).
		nudge := EvaluatePredictiveNudge("fix-tests", 4, 4, "repeated crash")
		if nudge.Triggered {
			t.Fatalf("Triggered = true, want false once full doom-loop halt is reached")
		}
		if !nudge.FullHaltReached() {
			t.Fatalf("FullHaltReached() = false, want true at K=4 flat turns")
		}
	})

	t.Run("OddThresholdCeiling", func(t *testing.T) {
		// K=3 (default TripWindows): ceiling(3/2) = 2.
		// 1 flat turn -> not triggered
		n1 := EvaluatePredictiveNudge("obj", 1, 3, "")
		if n1.Triggered {
			t.Fatalf("K=3, flat=1: Triggered = true, want false")
		}
		// 2 flat turns -> triggered
		n2 := EvaluatePredictiveNudge("obj", 2, 3, "")
		if !n2.Triggered {
			t.Fatalf("K=3, flat=2: Triggered = false, want true (ceiling of 3/2 is 2)")
		}

		// K=5: ceiling(5/2) = 3.
		// 2 flat turns -> not triggered
		n3 := EvaluatePredictiveNudge("obj", 2, 5, "")
		if n3.Triggered {
			t.Fatalf("K=5, flat=2: Triggered = true, want false (ceiling is 3)")
		}
		// 3 flat turns -> triggered
		n4 := EvaluatePredictiveNudge("obj", 3, 5, "")
		if !n4.Triggered {
			t.Fatalf("K=5, flat=3: Triggered = false, want true (ceiling is 3)")
		}
	})

	t.Run("MinimumTwoFloor", func(t *testing.T) {
		// K=2: ceiling(2/2) = 1, but minimum is 2.
		// Threshold should be 2, so 1 flat turn does not trigger.
		if got := PredictiveNudgeThreshold(2); got != 2 {
			t.Fatalf("PredictiveNudgeThreshold(2) = %d, want 2 (minimum floor)", got)
		}
		n1 := EvaluatePredictiveNudge("obj", 1, 2, "")
		if n1.Triggered {
			t.Fatalf("K=2, flat=1: Triggered = true, want false (minimum floor is 2)")
		}

		// K=1: minimum floor is 2.
		if got := PredictiveNudgeThreshold(1); got != 2 {
			t.Fatalf("PredictiveNudgeThreshold(1) = %d, want 2", got)
		}

		// K <= 0: defaults to DefaultConfig().TripWindows (3)
		if got := PredictiveNudgeThreshold(0); got != 2 {
			t.Fatalf("PredictiveNudgeThreshold(0) = %d, want 2", got)
		}
		nDefault := EvaluatePredictiveNudge("obj", 2, 0, "")
		if !nDefault.Triggered {
			t.Fatalf("K=0 (default 3), flat=2: Triggered = false, want true")
		}
		if nDefault.ThresholdK != DefaultConfig().TripWindows {
			t.Fatalf("ThresholdK = %d, want default TripWindows %d", nDefault.ThresholdK, DefaultConfig().TripWindows)
		}
	})

	t.Run("TrackerTurnLifecycle", func(t *testing.T) {
		tracker := NewPredictiveNudgeTracker("resolve-issue-11061", 4)

		// Initial state
		initial := tracker.Evaluate()
		if initial.Triggered || initial.ConsecutiveFlatTurns != 0 {
			t.Fatalf("initial state not clean: %+v", initial)
		}

		// Turn 1: flat turn with compiler error
		n1 := tracker.RecordFlatTurn("undefined: Foo")
		if n1.Triggered || n1.ConsecutiveFlatTurns != 1 {
			t.Fatalf("turn 1: Triggered = %v, flat = %d (want false, 1)", n1.Triggered, n1.ConsecutiveFlatTurns)
		}
		if n1.LastObservedFailure != "undefined: Foo" {
			t.Fatalf("turn 1: LastObservedFailure = %q, want undefined: Foo", n1.LastObservedFailure)
		}

		// Turn 2: flat turn reaches K/2 (2) -> predictive nudge triggered
		n2 := tracker.RecordFlatTurn("cannot use Bar as Baz")
		if !n2.Triggered || n2.ConsecutiveFlatTurns != 2 {
			t.Fatalf("turn 2: Triggered = %v, flat = %d (want true, 2)", n2.Triggered, n2.ConsecutiveFlatTurns)
		}
		if !strings.Contains(n2.NudgeMessage, "Your last 2 attempts showed no forward progress") {
			t.Fatalf("turn 2 message unexpected: %q", n2.NudgeMessage)
		}

		// Turn 3: progress made -> resets flat turns
		n3 := tracker.RecordProgress()
		if n3.Triggered || n3.ConsecutiveFlatTurns != 0 {
			t.Fatalf("turn 3 after progress: Triggered = %v, flat = %d (want false, 0)", n3.Triggered, n3.ConsecutiveFlatTurns)
		}
		if tracker.ConsecutiveFlatTurns() != 0 {
			t.Fatalf("ConsecutiveFlatTurns = %d, want 0 after progress", tracker.ConsecutiveFlatTurns())
		}

		// Turn 4: flat turn again
		n4 := tracker.RecordTurn(false, "segfault")
		if n4.Triggered || n4.ConsecutiveFlatTurns != 1 {
			t.Fatalf("turn 4: Triggered = %v, flat = %d (want false, 1)", n4.Triggered, n4.ConsecutiveFlatTurns)
		}

		// Changing objective resets tracker state
		tracker.SetObjective("new-task")
		if tracker.ConsecutiveFlatTurns() != 0 {
			t.Fatalf("ConsecutiveFlatTurns = %d after SetObjective, want 0", tracker.ConsecutiveFlatTurns())
		}

		// Reset clears tracker
		tracker.RecordFlatTurn("some error")
		tracker.Reset()
		if tracker.ConsecutiveFlatTurns() != 0 {
			t.Fatalf("ConsecutiveFlatTurns = %d after Reset, want 0", tracker.ConsecutiveFlatTurns())
		}
	})

	t.Run("ResultIntegration", func(t *testing.T) {
		cfg := Config{
			MinSamples:  3,
			TripWindows: 4,
		}

		// 3 samples => 2 windows. Both burning and flat.
		samples := mkSamples(
			[]int64{10, 20, 30},
			[]int64{1, 1, 1},
			true,
		)
		res := Classify(samples, cfg)
		// At 2 windows, doomloop.Classify says HEALTHY/OBSERVE (sub-threshold streak)
		if res.Verdict != VerdictHealthy || res.Correction != CorrectObserve {
			t.Fatalf("verdict=%q correction=%q, want HEALTHY/OBSERVE", res.Verdict, res.Correction)
		}

		// But predictive nudge at K/2 (4/2 = 2) triggers!
		pn := res.PredictiveNudge("investigate-bug", cfg, "test failure")
		if !pn.Triggered {
			t.Fatalf("predictive nudge was not triggered on result with 2 burning flat windows and K=4")
		}
		if pn.ConsecutiveFlatTurns != 2 {
			t.Fatalf("ConsecutiveFlatTurns = %d, want 2", pn.ConsecutiveFlatTurns)
		}
		if pn.ThresholdK != 4 {
			t.Fatalf("ThresholdK = %d, want 4", pn.ThresholdK)
		}

		// Also test EvaluateSamples convenience wrapper
		pnSamples := EvaluateSamples("investigate-bug", samples, cfg, "test failure")
		if !pnSamples.Triggered {
			t.Fatalf("EvaluateSamples: Triggered = false, want true")
		}
	})

	t.Run("MessageFormatExact", func(t *testing.T) {
		for _, attempts := range []int{2, 3, 5, 10} {
			got := FormatNudgeMessage(attempts)
			want := fmt.Sprintf("Your last %d attempts showed no forward progress. Step back and inspect the file or error diagnostics before retrying.", attempts)
			if got != want {
				t.Fatalf("FormatNudgeMessage(%d) = %q, want %q", attempts, got, want)
			}
		}
	})
}
