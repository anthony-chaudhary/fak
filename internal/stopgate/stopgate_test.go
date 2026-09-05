package stopgate

import (
	"fmt"
	"strings"
	"testing"
)

func TestLadderThresholdNormalization(t *testing.T) {
	// Test NormalizeDenyAllThresholds
	for _, tc := range []struct {
		w, f, m             int
		wantW, wantF, wantM int
	}{
		{3, 7, 9, 3, 7, 9},
		{0, 0, 0, 1, 1, DefaultMax},
		{-1, 15, 10, 1, 10, 10},
		{5, 2, 4, 4, 4, 4},
	} {
		gotW, gotF, gotM := NormalizeDenyAllThresholds(tc.w, tc.f, tc.m)
		if gotW != tc.wantW || gotF != tc.wantF || gotM != tc.wantM {
			t.Fatalf("NormalizeDenyAllThresholds(%d, %d, %d) = (%d, %d, %d); want (%d, %d, %d)",
				tc.w, tc.f, tc.m, gotW, gotF, gotM, tc.wantW, tc.wantF, tc.wantM)
		}
	}

	// Test NormalizeSameStop
	for _, tc := range []struct {
		stop                int
		wantW, wantF, wantS int
	}{
		{6, 3, 5, 6},
		{9, 6, 8, 9},
		{2, 1, 1, 2},
		{3, 1, 2, 3},
		{1, 3, 5, 6}, // < 2 falls back to default 6
		{0, 3, 5, 6},
		{-5, 3, 5, 6},
	} {
		gotW, gotF, gotS := NormalizeSameStop(tc.stop)
		if gotW != tc.wantW || gotF != tc.wantF || gotS != tc.wantS {
			t.Fatalf("NormalizeSameStop(%d) = (%d, %d, %d); want (%d, %d, %d)",
				tc.stop, gotW, gotF, gotS, tc.wantW, tc.wantF, tc.wantS)
		}
	}
}

func TestEvaluateDenyAllBlindLadder(t *testing.T) {
	cfg := DefaultLadderConfig()
	// Default: warn=3, final=7, max=9

	for _, tc := range []struct {
		consecutive int
		mode        Mode
		wantAction  Action
		wantStage   Stage
		wantDisp    Disposition
		wantExit    int
		wantBlocked bool
	}{
		{0, ModeEnforce, ActionAllow, StageAllow, DispCleanCompletion, 0, false},
		{1, ModeEnforce, ActionContinue, StageNudge, DispDenyAllContinue, 2, true},
		{2, ModeEnforce, ActionContinue, StageNudge, DispDenyAllContinue, 2, true},
		{3, ModeEnforce, ActionContinue, StageWarn, DispDenyAllContinue, 2, true},
		{6, ModeEnforce, ActionContinue, StageWarn, DispDenyAllContinue, 2, true},
		{7, ModeEnforce, ActionContinue, StageFinal, DispDenyAllContinue, 2, true},
		{9, ModeEnforce, ActionContinue, StageFinal, DispDenyAllContinue, 2, true},
		{10, ModeEnforce, ActionAllow, StageGiveUp, DispBlindGiveUp, 0, false},

		// Shadow mode
		{0, ModeShadow, ActionAllow, StageAllow, DispCleanCompletion, 0, false},
		{1, ModeShadow, ActionAllow, StageNudge, DispShadow, 0, true},
		{3, ModeShadow, ActionAllow, StageWarn, DispShadow, 0, true},
		{7, ModeShadow, ActionAllow, StageFinal, DispShadow, 0, true},
		{10, ModeShadow, ActionAllow, StageGiveUp, DispBlindGiveUp, 0, false},

		// Off mode
		{5, ModeOff, ActionAllow, StageWarn, DispModeOff, 0, false},
	} {
		cfg.Mode = tc.mode
		dec := EvaluateDenyAll(cfg, tc.consecutive, 0, false)
		if dec.Action != tc.wantAction || dec.Stage != tc.wantStage || dec.Disposition != tc.wantDisp || dec.ExitCode != tc.wantExit || dec.Blocked != tc.wantBlocked {
			t.Fatalf("EvaluateDenyAll(c=%d, mode=%s) = %+v; want action=%s stage=%s disp=%s exit=%d blocked=%v",
				tc.consecutive, tc.mode, dec, tc.wantAction, tc.wantStage, tc.wantDisp, tc.wantExit, tc.wantBlocked)
		}
	}
}

func TestEvaluateDenyAllSameIssueLadder(t *testing.T) {
	cfg := DefaultLadderConfig()
	// Default sameStop=6 -> warn=3, final=5, giveUp=6

	for _, tc := range []struct {
		sameConsecutive int
		mode            Mode
		wantAction      Action
		wantStage       Stage
		wantDisp        Disposition
		wantExit        int
		wantBlocked     bool
	}{
		{0, ModeEnforce, ActionAllow, StageAllow, DispCleanCompletion, 0, false},
		{1, ModeEnforce, ActionContinue, StageNudge, DispSameIssueContinue, 2, true},
		{2, ModeEnforce, ActionContinue, StageNudge, DispSameIssueContinue, 2, true},
		{3, ModeEnforce, ActionContinue, StageWarn, DispSameIssueContinue, 2, true},
		{4, ModeEnforce, ActionContinue, StageWarn, DispSameIssueContinue, 2, true},
		{5, ModeEnforce, ActionContinue, StageFinal, DispSameIssueContinue, 2, true},
		{6, ModeEnforce, ActionAllow, StageGiveUp, DispSameIssueGiveUp, 0, false},
		{10, ModeEnforce, ActionAllow, StageGiveUp, DispSameIssueGiveUp, 0, false},

		// Shadow
		{3, ModeShadow, ActionAllow, StageWarn, DispShadow, 0, true},
		{6, ModeShadow, ActionAllow, StageGiveUp, DispSameIssueGiveUp, 0, false},

		// Off
		{3, ModeOff, ActionAllow, StageWarn, DispModeOff, 0, false},
	} {
		cfg.Mode = tc.mode
		dec := EvaluateDenyAll(cfg, 0, tc.sameConsecutive, true)
		if dec.Action != tc.wantAction || dec.Stage != tc.wantStage || dec.Disposition != tc.wantDisp || dec.ExitCode != tc.wantExit || dec.Blocked != tc.wantBlocked {
			t.Fatalf("EvaluateDenyAll(same=%d, mode=%s) = %+v; want action=%s stage=%s disp=%s exit=%d blocked=%v",
				tc.sameConsecutive, tc.mode, dec, tc.wantAction, tc.wantStage, tc.wantDisp, tc.wantExit, tc.wantBlocked)
		}
	}
}

func TestEvaluateToolFeedback(t *testing.T) {
	cfg := DefaultLadderConfig()
	// ToolFeedbackMax = 25

	for _, tc := range []struct {
		consecutive int
		mode        Mode
		wantAction  Action
		wantStage   Stage
		wantDisp    Disposition
		wantExit    int
		wantBlocked bool
	}{
		{1, ModeEnforce, ActionContinue, StageNudge, DispToolFeedbackContinue, 2, true},
		{25, ModeEnforce, ActionContinue, StageNudge, DispToolFeedbackContinue, 2, true},
		{26, ModeEnforce, ActionAllow, StageGiveUp, DispToolFeedbackGiveUp, 0, false},

		// Shadow
		{5, ModeShadow, ActionAllow, StageNudge, DispShadow, 0, true},
		{30, ModeShadow, ActionAllow, StageGiveUp, DispShadow, 0, false},

		// Off
		{5, ModeOff, ActionAllow, StageAllow, DispModeOff, 0, false},
	} {
		cfg.Mode = tc.mode
		dec := EvaluateToolFeedback(cfg, tc.consecutive)
		if dec.Action != tc.wantAction || dec.Stage != tc.wantStage || dec.Disposition != tc.wantDisp || dec.ExitCode != tc.wantExit || dec.Blocked != tc.wantBlocked {
			t.Fatalf("EvaluateToolFeedback(%d, mode=%s) = %+v; want action=%s stage=%s disp=%s exit=%d blocked=%v",
				tc.consecutive, tc.mode, dec, tc.wantAction, tc.wantStage, tc.wantDisp, tc.wantExit, tc.wantBlocked)
		}
	}
}

func TestEvaluateWitness(t *testing.T) {
	cfg := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

	// 1. Not claimed
	notClaimed := WitnessClaim{Claimed: false}
	dec := EvaluateWitness(cfg, notClaimed, 0)
	if dec.Action != ActionAllow || dec.Disposition != DispCleanCompletion {
		t.Fatalf("not claimed want allow/clean, got %+v", dec)
	}

	// 2. Claimed and witnessed
	witnessed := WitnessClaim{Claimed: true, Witnessed: true, Commit: "abc1234", Detail: "stamped commit"}
	dec = EvaluateWitness(cfg, witnessed, 0)
	if dec.Action != ActionAllow || dec.Disposition != DispClaimWitnessed || !strings.Contains(dec.Note, "abc1234") {
		t.Fatalf("witnessed want allow/witnessed, got %+v", dec)
	}

	// 3. Claimed and unwitnessed, enforce mode, seq <= max
	unwitnessed := WitnessClaim{Claimed: true, Witnessed: false, Reason: "CLAIM_UNWITNESSED", Detail: "no commit"}
	dec = EvaluateWitness(cfg, unwitnessed, 0) // seq=1
	if dec.Action != ActionContinue || dec.Disposition != DispClaimUnwitnessedContinue || dec.ExitCode != 2 || !dec.Blocked {
		t.Fatalf("unwitnessed seq 1 want continue exit 2, got %+v", dec)
	}
	if !strings.Contains(dec.Guidance, "CLAIM_UNWITNESSED (1/3)") {
		t.Fatalf("guidance missing seq info: %s", dec.Guidance)
	}

	// 4. Claimed and unwitnessed, enforce mode, seq > max -> stand-down
	dec = EvaluateWitness(cfg, unwitnessed, 3) // seq=4 > max=3
	if dec.Action != ActionAllow || dec.Disposition != DispClaimUnwitnessedGiveUp || dec.ExitCode != 0 || dec.Blocked {
		t.Fatalf("unwitnessed seq 4 want allow stand-down, got %+v", dec)
	}

	// 5. Shadow mode
	cfg.Mode = ModeShadow
	dec = EvaluateWitness(cfg, unwitnessed, 0)
	if dec.Action != ActionAllow || dec.Disposition != DispClaimWitnessShadow || dec.ExitCode != 0 {
		t.Fatalf("shadow mode want allow/shadow, got %+v", dec)
	}

	// 6. STOP_UNWITNESSED detail format
	stopUnwitnessed := WitnessClaim{Claimed: true, Witnessed: false, Reason: "STOP_UNWITNESSED", Detail: "file:test.log"}
	cfg.Mode = ModeEnforce
	dec = EvaluateWitness(cfg, stopUnwitnessed, 0)
	if !strings.HasPrefix(dec.Guidance, "STOP_UNWITNESSED: missing declared witness: file:test.log") {
		t.Fatalf("STOP_UNWITNESSED guidance format mismatch: %s", dec.Guidance)
	}
}

func TestEvaluateBoundaryIntegration(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := DefaultWitnessGateConfig()

	// 1. Tool feedback prioritizes when denyAll == 0
	in := BoundaryInput{
		ConsecutiveDenyAll:      0,
		ConsecutiveToolFeedback: 3,
	}
	dec := EvaluateBoundary(ladder, witness, in)
	if dec.Disposition != DispToolFeedbackContinue || dec.Action != ActionContinue {
		t.Fatalf("want tool feedback continue, got %+v", dec)
	}

	// 2. DenyAll takes precedence when > 0
	in = BoundaryInput{
		ConsecutiveDenyAll:      2,
		ConsecutiveToolFeedback: 3,
	}
	dec = EvaluateBoundary(ladder, witness, in)
	if dec.Disposition != DispDenyAllContinue || dec.Action != ActionContinue {
		t.Fatalf("want deny all continue, got %+v", dec)
	}

	// 3. Noted no allowed path on clean stop -> clean wrapup
	in = BoundaryInput{
		NotedNoAllowedPath: true,
	}
	dec = EvaluateBoundary(ladder, witness, in)
	if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
		t.Fatalf("want clean wrapup, got %+v", dec)
	}

	// 4. DenyAll give-up + Noted no allowed path -> clean wrapup
	in = BoundaryInput{
		ConsecutiveDenyAll: 10,
		NotedNoAllowedPath: true,
	}
	dec = EvaluateBoundary(ladder, witness, in)
	if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
		t.Fatalf("want clean wrapup on giveup, got %+v", dec)
	}

	// 5. FinalGate unsatisfied
	in = BoundaryInput{
		FinalGate: func() (bool, string) { return false, "missing:stamp" },
	}
	dec = EvaluateBoundary(ladder, witness, in)
	if dec.Action != ActionContinue || !strings.Contains(dec.Guidance, "missing:stamp") {
		t.Fatalf("want finalGate continue, got %+v", dec)
	}

	// 6. Clean completion
	in = BoundaryInput{}
	dec = EvaluateBoundary(ladder, witness, in)
	if dec.Action != ActionAllow || dec.Disposition != DispCleanCompletion {
		t.Fatalf("want clean completion, got %+v", dec)
	}
}

func TestStopgateNotedNoAllowedPathRequiresWitness(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{
		Mode: ModeEnforce,
		Max:  2,
	}

	t.Run("final_gate_unsatisfied_blocks_clean_wrapup", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
			FinalGate:          func() (bool, string) { return false, "missing:stamp" },
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition == DispCleanWrapup || dec.ExitCode == 0 {
			t.Fatalf("expected blocked or unwitnessed continue, got clean wrapup: %+v", dec)
		}
		if dec.Action != ActionContinue || dec.ExitCode != 2 {
			t.Fatalf("want ActionContinue with exit 2, got %+v", dec)
		}
	})

	t.Run("final_gate_stand_down_after_max_blocks", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
			WitnessBlockCount:  2, // already at max
			FinalGate:          func() (bool, string) { return false, "missing:stamp" },
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("must not grant clean wrapup: %+v", dec)
		}
		if dec.Stage != StageGiveUp || dec.Disposition != DispClaimUnwitnessedGiveUp {
			t.Fatalf("want StageGiveUp with DispClaimUnwitnessedGiveUp, got %+v", dec)
		}
	})

	t.Run("missing_witness_claim_in_enforce_mode_blocks_clean_wrapup", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition == DispCleanWrapup || dec.ExitCode == 0 {
			t.Fatalf("expected blocked or unwitnessed continue, got clean wrapup: %+v", dec)
		}
		if dec.Action != ActionContinue || dec.ExitCode != 2 {
			t.Fatalf("want ActionContinue with exit 2, got %+v", dec)
		}
	})

	t.Run("witness_satisfied_allows_clean_wrapup", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
			FinalGate:          func() (bool, string) { return true, "" },
			WitnessClaim: &WitnessClaim{
				Claimed:   true,
				Witnessed: true,
				Commit:    "abc1234",
				Detail:    "verified commit",
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition != DispCleanWrapup || dec.ExitCode != 0 || dec.Action != ActionAllow {
			t.Fatalf("want clean wrapup when witness is satisfied, got %+v", dec)
		}
	})
}

// TestHarnessParityTrajectory proves that guard and native agent harnesses make
// IDENTICAL stop/continue decisions and synthesize IDENTICAL continuation prompts
// given equivalent consecutive denial trajectories.
func TestHarnessParityTrajectory(t *testing.T) {
	ladderCfg := DefaultLadderConfig()
	witnessCfg := DefaultWitnessGateConfig()

	t.Run("blind_consecutive_denial_trajectory", func(t *testing.T) {
		// Test trajectory steps from 0 to 11 consecutive deny-all turns
		for count := 0; count <= 11; count++ {
			// Guard harness evaluation
			guardDec := EvaluateDenyAll(ladderCfg, count, 0, false)

			// Agent harness boundary evaluation
			agentIn := BoundaryInput{
				SessionID:          "sess-test",
				Turn:               count,
				ConsecutiveDenyAll: count,
				UseSameIssue:       false,
			}
			agentDec := EvaluateBoundary(ladderCfg, witnessCfg, agentIn)

			if guardDec.Action != agentDec.Action {
				t.Fatalf("count %d: Action mismatch guard=%s agent=%s", count, guardDec.Action, agentDec.Action)
			}
			if guardDec.Stage != agentDec.Stage {
				t.Fatalf("count %d: Stage mismatch guard=%s agent=%s", count, guardDec.Stage, agentDec.Stage)
			}
			if guardDec.Disposition != agentDec.Disposition {
				t.Fatalf("count %d: Disposition mismatch guard=%s agent=%s", count, guardDec.Disposition, agentDec.Disposition)
			}
			if guardDec.Blocked != agentDec.Blocked {
				t.Fatalf("count %d: Blocked mismatch guard=%v agent=%v", count, guardDec.Blocked, agentDec.Blocked)
			}
			if guardDec.ExitCode != agentDec.ExitCode {
				t.Fatalf("count %d: ExitCode mismatch guard=%d agent=%d", count, guardDec.ExitCode, agentDec.ExitCode)
			}
			if guardDec.Guidance != agentDec.Guidance {
				t.Fatalf("count %d: Guidance prompt mismatch!\nGuard:\n%s\nAgent:\n%s", count, guardDec.Guidance, agentDec.Guidance)
			}

			// Validate ladder stages along the trajectory
			switch {
			case count == 0:
				if agentDec.Action != ActionAllow || agentDec.Stage != StageAllow {
					t.Fatalf("count 0 must allow clean completion, got %+v", agentDec)
				}
			case count >= 1 && count <= 2:
				if agentDec.Action != ActionContinue || agentDec.Stage != StageNudge {
					t.Fatalf("count %d must nudge continue, got %+v", count, agentDec)
				}
				if agentDec.Guidance != ContinueReason {
					t.Fatalf("count %d nudge prompt mismatch", count)
				}
			case count >= 3 && count <= 6:
				if agentDec.Action != ActionContinue || agentDec.Stage != StageWarn {
					t.Fatalf("count %d must warn continue, got %+v", count, agentDec)
				}
				if !strings.Contains(agentDec.Guidance, fmt.Sprintf("last %d turns", count)) {
					t.Fatalf("count %d warn prompt missing turn count: %s", count, agentDec.Guidance)
				}
			case count >= 7 && count <= 9:
				if agentDec.Action != ActionContinue || agentDec.Stage != StageFinal {
					t.Fatalf("count %d must final continue, got %+v", count, agentDec)
				}
				if !strings.Contains(agentDec.Guidance, "last auto-continue") {
					t.Fatalf("count %d final prompt missing 'last auto-continue': %s", count, agentDec.Guidance)
				}
			case count >= 10:
				if agentDec.Action != ActionAllow || agentDec.Stage != StageGiveUp {
					t.Fatalf("count %d must stand down give-up, got %+v", count, agentDec)
				}
			}
		}
	})

	t.Run("same_issue_denial_trajectory", func(t *testing.T) {
		// Test trajectory steps from 0 to 8 consecutive same-issue turns
		for count := 0; count <= 8; count++ {
			guardDec := EvaluateDenyAll(ladderCfg, 0, count, true)
			agentIn := BoundaryInput{
				SessionID:            "sess-test",
				Turn:                 count,
				ConsecutiveDenyAll:   count,
				ConsecutiveSameIssue: count,
				UseSameIssue:         true,
			}
			agentDec := EvaluateBoundary(ladderCfg, witnessCfg, agentIn)

			if guardDec.Action != agentDec.Action {
				t.Fatalf("same %d: Action mismatch guard=%s agent=%s", count, guardDec.Action, agentDec.Action)
			}
			if guardDec.Stage != agentDec.Stage {
				t.Fatalf("same %d: Stage mismatch guard=%s agent=%s", count, guardDec.Stage, agentDec.Stage)
			}
			if guardDec.Disposition != agentDec.Disposition {
				t.Fatalf("same %d: Disposition mismatch guard=%s agent=%s", count, guardDec.Disposition, agentDec.Disposition)
			}
			if guardDec.Blocked != agentDec.Blocked {
				t.Fatalf("same %d: Blocked mismatch guard=%v agent=%v", count, guardDec.Blocked, agentDec.Blocked)
			}
			if guardDec.ExitCode != agentDec.ExitCode {
				t.Fatalf("same %d: ExitCode mismatch guard=%d agent=%d", count, guardDec.ExitCode, agentDec.ExitCode)
			}
			if guardDec.Guidance != agentDec.Guidance {
				t.Fatalf("same %d: Guidance prompt mismatch!\nGuard:\n%s\nAgent:\n%s", count, guardDec.Guidance, agentDec.Guidance)
			}

			// Validate same-issue stages
			switch {
			case count == 0:
				if agentDec.Action != ActionAllow || agentDec.Stage != StageAllow {
					t.Fatalf("same 0 must allow clean, got %+v", agentDec)
				}
			case count >= 1 && count <= 2:
				if agentDec.Action != ActionContinue || agentDec.Stage != StageNudge {
					t.Fatalf("same %d must nudge continue, got %+v", count, agentDec)
				}
			case count >= 3 && count <= 4:
				if agentDec.Action != ActionContinue || agentDec.Stage != StageWarn {
					t.Fatalf("same %d must warn continue, got %+v", count, agentDec)
				}
				if !strings.Contains(agentDec.Guidance, "IDENTICAL refused action") {
					t.Fatalf("same %d warn prompt missing 'IDENTICAL refused action'", count)
				}
			case count == 5:
				if agentDec.Action != ActionContinue || agentDec.Stage != StageFinal {
					t.Fatalf("same 5 must final continue, got %+v", agentDec)
				}
				if !strings.Contains(agentDec.Guidance, "last auto-continue") {
					t.Fatalf("same %d final prompt missing 'last auto-continue'", count)
				}
			case count >= 6:
				if agentDec.Action != ActionAllow || agentDec.Stage != StageGiveUp {
					t.Fatalf("same %d must stand down give-up, got %+v", count, agentDec)
				}
			}
		}
	})

	t.Run("tool_feedback_trajectory", func(t *testing.T) {
		for _, count := range []int{1, 10, 25, 26, 30} {
			guardDec := EvaluateToolFeedback(ladderCfg, count)
			agentIn := BoundaryInput{
				ConsecutiveDenyAll:      0,
				ConsecutiveToolFeedback: count,
			}
			agentDec := EvaluateBoundary(ladderCfg, witnessCfg, agentIn)

			if guardDec.Action != agentDec.Action || guardDec.Stage != agentDec.Stage || guardDec.Guidance != agentDec.Guidance {
				t.Fatalf("tool feedback %d mismatch: guard=%+v agent=%+v", count, guardDec, agentDec)
			}
		}
	})
}
