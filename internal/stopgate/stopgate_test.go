package stopgate

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
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
		BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
			Reason:   "POLICY_BLOCK",
			Verified: true,
		},
	}
	dec = EvaluateBoundary(ladder, witness, in)
	if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
		t.Fatalf("want clean wrapup, got %+v", dec)
	}

	// 4. DenyAll give-up + Noted no allowed path -> clean wrapup
	in = BoundaryInput{
		ConsecutiveDenyAll: 10,
		NotedNoAllowedPath: true,
		BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
			Reason:   "POLICY_BLOCK",
			Verified: true,
		},
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
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "POLICY_BLOCK",
				Verified: true,
			},
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

func TestStopgateNoAllowedPathWitnessedBoundaryRefusal(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := DefaultWitnessGateConfig()

	// a) in.NotedNoAllowedPath = true with RefusalReceipt: &BoundaryRefusalReceipt{Reason: "LOCK_BUSY", Verified: true} -> refuses DispCleanWrapup, returns ActionContinue exit code 2
	t.Run("lock_busy_refuses_clean_wrapup", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
			RefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "LOCK_BUSY",
				Verified: true,
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup on LOCK_BUSY, got: %+v", dec)
		}
		if dec.Action != ActionContinue || dec.ExitCode != 2 {
			t.Fatalf("want ActionContinue with exit 2, got action=%s exit=%d", dec.Action, dec.ExitCode)
		}
		if dec.Signal != "STOP_UNWITNESSED" {
			t.Fatalf("want Signal STOP_UNWITNESSED, got %q", dec.Signal)
		}
	})

	// b) in.NotedNoAllowedPath = true with RefusalReceipt: nil -> refuses DispCleanWrapup, returns ActionContinue exit code 2
	t.Run("nil_receipt_refuses_clean_wrapup", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
			RefusalReceipt:     nil,
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup on nil receipt, got: %+v", dec)
		}
		if dec.Action != ActionContinue || dec.ExitCode != 2 {
			t.Fatalf("want ActionContinue with exit 2, got action=%s exit=%d", dec.Action, dec.ExitCode)
		}
		if dec.Signal != "STOP_UNWITNESSED" {
			t.Fatalf("want Signal STOP_UNWITNESSED, got %q", dec.Signal)
		}
	})

	// c) in.NotedNoAllowedPath = true with RefusalReceipt: &BoundaryRefusalReceipt{Reason: "POLICY_BLOCK", Verified: true, Disposition: "TERMINAL"} -> admits DispCleanWrapup exit code 0
	t.Run("policy_block_admits_clean_wrapup", func(t *testing.T) {
		in := BoundaryInput{
			NotedNoAllowedPath: true,
			RefusalReceipt: &BoundaryRefusalReceipt{
				Reason:      "POLICY_BLOCK",
				Verified:    true,
				Disposition: "TERMINAL",
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow || dec.ExitCode != 0 {
			t.Fatalf("want DispCleanWrapup with ActionAllow and exit 0, got: %+v", dec)
		}
	})

	// d) IsTransientHurdle and IsTerminalBoundary table test
	t.Run("transient_and_terminal_table_test", func(t *testing.T) {
		transientCases := []struct {
			reason      string
			disposition string
			want        bool
		}{
			{"LOCK_BUSY", "", true},
			{"LOCK_BUSY", "RETRYABLE", true},
			{"RATE_LIMITED", "", true},
			{"COLLISION_RISK", "WAIT", true},
			{"LEASE_HELD", "", true},
			{"MISROUTE", "", true},
			{"MALFORMED", "", true},
			{"POLICY_BLOCK", "", false},
			{"POLICY_BLOCK", "TERMINAL", false},
			{"LOCK_BUSY", "TERMINAL", false}, // explicit TERMINAL overrides reason
			{"TRUST_VIOLATION", "TERMINAL", false},
			{"UNKNOWN_REASON", "", false},
			{"UNKNOWN_REASON", "RETRYABLE", true},
		}
		for _, tc := range transientCases {
			got := IsTransientHurdle(tc.reason, tc.disposition)
			if got != tc.want {
				t.Errorf("IsTransientHurdle(%q, %q) = %v; want %v", tc.reason, tc.disposition, got, tc.want)
			}
		}

		terminalCases := []struct {
			receipt *BoundaryRefusalReceipt
			want    bool
		}{
			{nil, false},
			{&BoundaryRefusalReceipt{Reason: "POLICY_BLOCK", Verified: false}, false},
			{&BoundaryRefusalReceipt{Reason: "POLICY_BLOCK", Verified: true, Disposition: "TERMINAL"}, true},
			{&BoundaryRefusalReceipt{Reason: "POLICY_BLOCK", Verified: true}, true},
			{&BoundaryRefusalReceipt{Reason: "TRUST_VIOLATION", Verified: true}, true},
			{&BoundaryRefusalReceipt{Reason: "SELF_MODIFY", Verified: true}, true},
			{&BoundaryRefusalReceipt{Reason: "LOCK_BUSY", Verified: true}, false},
			{&BoundaryRefusalReceipt{Reason: "LOCK_BUSY", Verified: true, Disposition: "RETRYABLE"}, false},
			{&BoundaryRefusalReceipt{Reason: "CUSTOM", Verified: true, Terminal: true}, true},
			{&BoundaryRefusalReceipt{Reason: "CUSTOM", Verified: true, Disposition: "TERMINAL"}, true},
			{&BoundaryRefusalReceipt{Reason: "CUSTOM", Verified: true, Transient: true}, false},
			{&BoundaryRefusalReceipt{ReasonCode: abi.ReasonPolicyBlock, Verified: true}, true},
			{&BoundaryRefusalReceipt{ReasonCode: abi.ReasonRateLimited, Verified: true}, false},
		}
		for i, tc := range terminalCases {
			got := IsTerminalBoundary(tc.receipt)
			if got != tc.want {
				t.Errorf("case %d: IsTerminalBoundary(%+v) = %v; want %v", i, tc.receipt, got, tc.want)
			}
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

// TestUnverifiedSurrenderStringsTriggerRefusal verifies that active goals reject unverified
// textual surrender phrases (such as "no allowed path", "cannot proceed", "giving up", "no allowed path: <reason>"),
// triggering ActionContinue / refusal unless accompanied by a valid kernel refusal token or witness (#11771).
func TestUnverifiedSurrenderStringsTriggerRefusal(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

	surrenderPhrases := []struct {
		name   string
		phrase string
	}{
		{"no_allowed_path", "no allowed path"},
		{"no_allowed_path_with_reason", "no allowed path: permission denied on write target"},
		{"no_allowed_path_colon", "no allowed path: resource constrained"},
		{"cannot_proceed", "cannot proceed with this approach"},
		{"giving_up", "giving up on task due to compilation errors"},
		{"unable_to_proceed", "unable to proceed without external dependency"},
		{"cannot_fix", "cannot fix the issue after multiple attempts"},
	}

	for _, sp := range surrenderPhrases {
		t.Run("active_goal_unverified_"+sp.name, func(t *testing.T) {
			in := BoundaryInput{
				GoalActive:    true,
				GoalObjective: "adversarial anti-surrender enforcement",
				SurrenderNote: sp.phrase,
			}
			dec := EvaluateBoundary(ladder, witness, in)
			if dec.Action != ActionContinue {
				t.Fatalf("phrase %q: want ActionContinue, got %s", sp.phrase, dec.Action)
			}
			if dec.ExitCode != 2 {
				t.Fatalf("phrase %q: want ExitCode 2, got %d", sp.phrase, dec.ExitCode)
			}
			if !dec.Blocked {
				t.Fatalf("phrase %q: want Blocked true, got false", sp.phrase)
			}
			if dec.Disposition != DispClaimUnwitnessedContinue {
				t.Fatalf("phrase %q: want DispClaimUnwitnessedContinue, got %s", sp.phrase, dec.Disposition)
			}
			if dec.Kind != KindContinue {
				t.Fatalf("phrase %q: want KindContinue, got %s", sp.phrase, dec.Kind)
			}
		})
	}

	t.Run("unverified_refusal_receipt_triggers_refusal", func(t *testing.T) {
		in := BoundaryInput{
			GoalActive:    true,
			GoalObjective: "adversarial anti-surrender enforcement",
			SurrenderNote: "no allowed path: policy restriction",
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:    "POLICY_BLOCK",
				Verified:  false,
				Signature: "", // unverified, unsigned
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionContinue || dec.ExitCode != 2 || !dec.Blocked {
			t.Fatalf("unverified receipt must trigger ActionContinue/refusal, got: %+v", dec)
		}
		if dec.Disposition != DispClaimUnwitnessedContinue {
			t.Fatalf("want DispClaimUnwitnessedContinue, got %s", dec.Disposition)
		}
	})

	t.Run("witness_satisfied_allows_clean_wrapup_even_with_surrender_note", func(t *testing.T) {
		in := BoundaryInput{
			GoalActive:    true,
			GoalObjective: "adversarial anti-surrender enforcement",
			SurrenderNote: "cannot proceed further, completed all steps",
			WitnessClaim: &WitnessClaim{
				Claimed:   true,
				Witnessed: true,
				Commit:    "c0ffee1",
				Detail:    "witnessed proof commit",
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionAllow || dec.ExitCode != 0 || dec.Disposition != DispCleanWrapup {
			t.Fatalf("witnessed claim must admit DispCleanWrapup, got: %+v", dec)
		}
	})
}

// TestValidSignedRefusalTokensAllowCleanWrapupAndTripCircuitBreaker verifies that valid signed
// refusal tokens admit clean wrapup in EvaluateBoundary and trip the circuit breaker (#11771).
func TestValidSignedRefusalTokensAllowCleanWrapupAndTripCircuitBreaker(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

	signedReceipt := &BoundaryRefusalReceipt{
		Tool:        "bash",
		Reason:      "POLICY_BLOCK",
		Disposition: "TERMINAL",
		Signature:   "ed25519:sig_kernel_verified_7781",
		Verified:    true,
	}

	// 1. In EvaluateBoundary: valid signed refusal token admits DispCleanWrapup with ExitCode 0
	in := BoundaryInput{
		GoalActive:             true,
		GoalObjective:          "testing signed refusal tokens",
		NotedNoAllowedPath:     true,
		SurrenderNote:          "no allowed path: policy boundary reached",
		BoundaryRefusalReceipt: signedReceipt,
	}
	dec := EvaluateBoundary(ladder, witness, in)
	if dec.Action != ActionAllow {
		t.Fatalf("signed refusal token: want ActionAllow, got %s", dec.Action)
	}
	if dec.ExitCode != 0 {
		t.Fatalf("signed refusal token: want ExitCode 0, got %d", dec.ExitCode)
	}
	if dec.Disposition != DispCleanWrapup {
		t.Fatalf("signed refusal token: want DispCleanWrapup, got %s", dec.Disposition)
	}
	if dec.Kind != KindClean {
		t.Fatalf("signed refusal token: want KindClean, got %s", dec.Kind)
	}

	// 2. In CircuitBreaker: valid signed refusal token trips the circuit breaker
	cb := NewCircuitBreaker()
	if cb.IsTripped() {
		t.Fatal("new circuit breaker must not be tripped")
	}
	tripped := cb.RecordSignedRefusal(signedReceipt)
	if !tripped {
		t.Fatal("RecordSignedRefusal must return true for valid signed receipt")
	}
	if !cb.IsTripped() {
		t.Fatal("circuit breaker must be tripped after RecordSignedRefusal")
	}
	if !strings.Contains(cb.TripReason(), "POLICY_BLOCK") {
		t.Fatalf("trip reason must mention POLICY_BLOCK, got %q", cb.TripReason())
	}

	// 3. Tripped circuit breaker in EvaluateBoundary allows session to stand down cleanly
	inWithCB := BoundaryInput{
		GoalActive:     true,
		SurrenderNote:  "cannot proceed",
		CircuitBreaker: cb,
	}
	decCB := EvaluateBoundary(ladder, witness, inWithCB)
	if decCB.Action != ActionAllow || decCB.ExitCode != 0 || decCB.Kind != KindStandDown {
		t.Fatalf("tripped circuit breaker must allow stand-down, got: %+v", decCB)
	}
}

// TestCircuitBreakerMD5ToolSignatureTracking tests MD5 tool invocation signature tracking
// MD5(tool_name || canonical_json_args) and circuit breaker tripping invariants (#11771).
func TestCircuitBreakerMD5ToolSignatureTracking(t *testing.T) {
	// 1. Signature canonicalization invariance
	t.Run("md5_signature_canonical_json_invariance", func(t *testing.T) {
		sig1 := ToolInvocationSignature("bash", `{"command":"git status","timeout":10}`)
		sig2 := ToolInvocationSignature("bash", `{"timeout":10,"command":"git status"}`)
		sig3 := ToolInvocationSignature("bash", `{  "command" : "git status" , "timeout" : 10  }`)

		if sig1 != sig2 {
			t.Fatalf("key reordering produced different signatures: sig1=%s, sig2=%s", sig1, sig2)
		}
		if sig1 != sig3 {
			t.Fatalf("whitespace produced different signatures: sig1=%s, sig3=%s", sig1, sig3)
		}
		if len(sig1) != 32 {
			t.Fatalf("MD5 signature must be 32 hex chars, got len %d (%s)", len(sig1), sig1)
		}

		// Different tools produce different signatures
		sigOtherTool := ToolInvocationSignature("sh", `{"command":"git status","timeout":10}`)
		if sig1 == sigOtherTool {
			t.Fatalf("different tools must have different signatures: %s == %s", sig1, sigOtherTool)
		}

		// Different args produce different signatures
		sigOtherArgs := ToolInvocationSignature("bash", `{"command":"git log","timeout":10}`)
		if sig1 == sigOtherArgs {
			t.Fatalf("different args must have different signatures: %s == %s", sig1, sigOtherArgs)
		}
	})

	// 2. Trips circuit breaker only when identical failing calls repeat >= 3 consecutive times without progress
	t.Run("trips_at_three_consecutive_identical_failures", func(t *testing.T) {
		cb := NewCircuitBreaker()
		tool := "run_test"
		args := `{"test":"TestFeature","timeout":30}`

		// Call 1 fails
		tripped1 := cb.RecordFailure(tool, args)
		if tripped1 || cb.IsTripped() || cb.ConsecutiveFailures() != 1 {
			t.Fatalf("turn 1: unexpected trip or count: tripped=%v, count=%d", tripped1, cb.ConsecutiveFailures())
		}

		// Call 2 fails with identical args (reordered keys)
		argsReordered := `{"timeout":30,"test":"TestFeature"}`
		tripped2 := cb.RecordFailure(tool, argsReordered)
		if tripped2 || cb.IsTripped() || cb.ConsecutiveFailures() != 2 {
			t.Fatalf("turn 2: unexpected trip or count: tripped=%v, count=%d", tripped2, cb.ConsecutiveFailures())
		}

		// Call 3 fails with identical args (whitespace variant) -> trips circuit breaker!
		argsWhitespace := `{  "test" : "TestFeature" , "timeout" : 30  }`
		tripped3 := cb.RecordFailure(tool, argsWhitespace)
		if !tripped3 || !cb.IsTripped() || cb.ConsecutiveFailures() != 3 {
			t.Fatalf("turn 3: expected circuit breaker trip: tripped=%v, isTripped=%v, count=%d", tripped3, cb.IsTripped(), cb.ConsecutiveFailures())
		}
		if !strings.Contains(cb.TripReason(), "3 consecutive occurrences without progress") {
			t.Fatalf("unexpected trip reason: %s", cb.TripReason())
		}
	})

	// 3. Exploratory retries (different tools or args) do NOT trip the circuit breaker
	t.Run("exploratory_retries_do_not_trip_circuit_breaker", func(t *testing.T) {
		cb := NewCircuitBreaker()

		// Attempt 1: tool A fails
		cb.RecordFailure("fetch_page", `{"url":"https://example.com/api"}`)
		if cb.ConsecutiveFailures() != 1 || cb.IsTripped() {
			t.Fatal("unexpected state after attempt 1")
		}

		// Attempt 2: tool A with different URL fails (exploratory)
		cb.RecordFailure("fetch_page", `{"url":"https://example.com/health"}`)
		if cb.ConsecutiveFailures() != 1 || cb.IsTripped() {
			t.Fatalf("exploratory retry must reset count to 1, got %d", cb.ConsecutiveFailures())
		}

		// Attempt 3: tool B fails (exploratory)
		cb.RecordFailure("curl_cmd", `{"url":"https://example.com/api"}`)
		if cb.ConsecutiveFailures() != 1 || cb.IsTripped() {
			t.Fatalf("different tool must reset count to 1, got %d", cb.ConsecutiveFailures())
		}

		// Attempt 4: tool C fails (exploratory)
		cb.RecordFailure("ping_host", `{"host":"example.com"}`)
		if cb.ConsecutiveFailures() != 1 || cb.IsTripped() {
			t.Fatalf("different tool must reset count to 1, got %d", cb.ConsecutiveFailures())
		}
	})

	// 4. Progress or success resets failure tracking
	t.Run("progress_or_success_resets_circuit_breaker", func(t *testing.T) {
		cb := NewCircuitBreaker()
		tool := "build_target"
		args := `{"target":"//cmd/fak"}`

		// 2 consecutive failures
		cb.RecordFailure(tool, args)
		cb.RecordFailure(tool, args)
		if cb.ConsecutiveFailures() != 2 {
			t.Fatalf("want 2 failures, got %d", cb.ConsecutiveFailures())
		}

		// Progress occurs
		cb.RecordSuccess(tool, args)
		if cb.ConsecutiveFailures() != 0 || cb.IsTripped() {
			t.Fatalf("success must reset counter to 0, got %d", cb.ConsecutiveFailures())
		}

		// Subsequent failure starts from 1
		cb.RecordFailure(tool, args)
		if cb.ConsecutiveFailures() != 1 || cb.IsTripped() {
			t.Fatalf("post-success failure must be at count 1, got %d", cb.ConsecutiveFailures())
		}
	})
}
