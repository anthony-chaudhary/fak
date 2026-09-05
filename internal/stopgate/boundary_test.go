package stopgate

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestBoundaryNoAllowedPathRequiresVerifiedRefusal(t *testing.T) {
	ladder := DefaultLadderConfig()

	t.Run("refused_disp_clean_wrapup_when_unwitnessed", func(t *testing.T) {
		// 1. "no allowed path" without verified terminal boundary refusal receipt
		// is refused DispCleanWrapup (treated as STOP_UNWITNESSED).

		// Case A: Missing receipt (nil) in ModeEnforce -> blocks with DispClaimUnwitnessedContinue
		witnessEnforce := WitnessGateConfig{Mode: ModeEnforce, Max: 3}
		inMissingEnforce := BoundaryInput{
			NotedNoAllowedPath: true,
		}
		dec := EvaluateBoundary(ladder, witnessEnforce, inMissingEnforce)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup for missing receipt in enforce mode, got: %+v", dec)
		}
		if dec.Action != ActionContinue || dec.ExitCode != 2 || dec.Reason != "STOP_UNWITNESSED" {
			t.Fatalf("want ActionContinue with exit 2 and reason STOP_UNWITNESSED, got: %+v", dec)
		}
		if dec.Disposition != DispClaimUnwitnessedContinue {
			t.Fatalf("want DispClaimUnwitnessedContinue, got: %s", dec.Disposition)
		}

		// Case B: Missing receipt (nil) in ModeShadow -> refuses DispCleanWrapup
		witnessShadow := WitnessGateConfig{Mode: ModeShadow, Max: 3}
		inMissingShadow := BoundaryInput{
			NotedNoAllowedPath: true,
		}
		dec = EvaluateBoundary(ladder, witnessShadow, inMissingShadow)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup for missing receipt in shadow mode, got: %+v", dec)
		}
		if dec.Reason != "STOP_UNWITNESSED" || (dec.Disposition != DispClaimWitnessShadow && dec.Disposition != DispClaimUnwitnessedContinue) {
			t.Fatalf("want STOP_UNWITNESSED continuation, got: %+v", dec)
		}

		// Case C: Unverified receipt (Verified: false) -> treated as STOP_UNWITNESSED
		inUnverified := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "POLICY_BLOCK",
				Verified: false,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inUnverified)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup for unverified receipt, got: %+v", dec)
		}
		if dec.Reason != "STOP_UNWITNESSED" || dec.Disposition != DispClaimUnwitnessedContinue {
			t.Fatalf("want STOP_UNWITNESSED continue for unverified receipt, got: %+v", dec)
		}

		// Case D: Transient hurdle (LOCK_BUSY) without terminal receipt -> treated as STOP_UNWITNESSED
		inLockBusy := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "LOCK_BUSY",
				Verified: true,
				Terminal: false,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inLockBusy)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup on transient hurdle LOCK_BUSY, got: %+v", dec)
		}
		if dec.Reason != "STOP_UNWITNESSED" || dec.Disposition != DispClaimUnwitnessedContinue {
			t.Fatalf("want STOP_UNWITNESSED for transient LOCK_BUSY, got: %+v", dec)
		}

		// Case E: Transient hurdle with ReasonCode (abi.ReasonRateLimited) -> treated as STOP_UNWITNESSED
		inRateLimited := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				ReasonCode: abi.ReasonRateLimited,
				Verified:   true,
				Terminal:   false,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inRateLimited)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup on transient ReasonRateLimited, got: %+v", dec)
		}
		if dec.Reason != "STOP_UNWITNESSED" {
			t.Fatalf("want STOP_UNWITNESSED for transient reason code, got: %+v", dec)
		}

		// Case F: Explicit transient flag (Transient: true) -> treated as STOP_UNWITNESSED
		inExplicitTransient := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:    "POLICY_BLOCK",
				Verified:  true,
				Transient: true,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inExplicitTransient)
		if dec.Disposition == DispCleanWrapup {
			t.Fatalf("expected refusal of DispCleanWrapup on Transient:true, got: %+v", dec)
		}
		if dec.Reason != "STOP_UNWITNESSED" {
			t.Fatalf("want STOP_UNWITNESSED for Transient:true, got: %+v", dec)
		}
	})

	t.Run("admitted_disp_clean_wrapup_when_terminal_witnessed", func(t *testing.T) {
		// 2. "no allowed path" accompanied by a verified terminal boundary refusal receipt
		// (e.g. POLICY_BLOCK) is admitted with DispCleanWrapup.
		witnessEnforce := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

		// Case A: Verified POLICY_BLOCK receipt
		inPolicyBlock := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "POLICY_BLOCK",
				Verified: true,
			},
		}
		dec := EvaluateBoundary(ladder, witnessEnforce, inPolicyBlock)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow || dec.ExitCode != 0 {
			t.Fatalf("want DispCleanWrapup allow for verified POLICY_BLOCK, got: %+v", dec)
		}
		if dec.Kind != KindClean || dec.Stage != StageAllow {
			t.Fatalf("want KindClean and StageAllow, got kind=%s stage=%s", dec.Kind, dec.Stage)
		}

		// Case B: Verified ReasonCode (abi.ReasonPolicyBlock)
		inReasonCode := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				ReasonCode: abi.ReasonPolicyBlock,
				Verified:   true,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inReasonCode)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow || dec.ExitCode != 0 {
			t.Fatalf("want DispCleanWrapup for verified ReasonPolicyBlock, got: %+v", dec)
		}

		// Case C: Verified SELF_MODIFY receipt
		inSelfModify := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "SELF_MODIFY",
				Verified: true,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inSelfModify)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
			t.Fatalf("want DispCleanWrapup for verified SELF_MODIFY, got: %+v", dec)
		}

		// Case D: Explicit Terminal: true on receipt even with custom reason
		inCustomTerminal := BoundaryInput{
			NotedNoAllowedPath: true,
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "CUSTOM_TERMINAL_BARRIER",
				Verified: true,
				Terminal: true,
			},
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inCustomTerminal)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
			t.Fatalf("want DispCleanWrapup for explicit Terminal:true, got: %+v", dec)
		}

		// Case E: RefusalToken on BoundaryInput
		inToken := BoundaryInput{
			NotedNoAllowedPath: true,
			RefusalToken:       "POLICY_BLOCK",
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inToken)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
			t.Fatalf("want DispCleanWrapup for RefusalToken POLICY_BLOCK, got: %+v", dec)
		}

		// Case F: ReasonCode directly on BoundaryInput
		inDirectCode := BoundaryInput{
			NotedNoAllowedPath: true,
			ReasonCode:         abi.ReasonPolicyBlock,
		}
		dec = EvaluateBoundary(ladder, witnessEnforce, inDirectCode)
		if dec.Disposition != DispCleanWrapup || dec.Action != ActionAllow {
			t.Fatalf("want DispCleanWrapup for ReasonCode abi.ReasonPolicyBlock, got: %+v", dec)
		}
	})
}

func TestIsSurrenderNote(t *testing.T) {
	phrases := []string{
		"giving up",
		"give up",
		"cannot complete",
		"unable to proceed",
		"cannot proceed",
		"stopping here",
		"i surrender",
		"failed to complete",
		"unable to solve",
		"cannot solve",
		"cannot fix",
		"stop now",
	}
	for _, p := range phrases {
		if !IsSurrenderNote("I am " + p + " because of errors") {
			t.Errorf("IsSurrenderNote should match phrase %q", p)
		}
		if !IsSurrenderNote(strings.ToUpper(p)) {
			t.Errorf("IsSurrenderNote should match uppercase phrase %q", p)
		}
	}
	negatives := []string{
		"",
		"all tests pass",
		"completed successfully",
		"shipping the fix",
	}
	for _, neg := range negatives {
		if IsSurrenderNote(neg) {
			t.Errorf("IsSurrenderNote should not match %q", neg)
		}
	}
}

func TestStopgatePrematureSurrender(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

	t.Run("surrender_note_without_terminal_refusal_returns_continue", func(t *testing.T) {
		cases := []struct {
			name  string
			input BoundaryInput
		}{
			{
				name: "noted_surrender_flag",
				input: BoundaryInput{
					NotedSurrender: true,
				},
			},
			{
				name: "surrender_note_giving_up",
				input: BoundaryInput{
					SurrenderNote: "I am giving up on this task",
				},
			},
			{
				name: "surrender_note_cannot_proceed",
				input: BoundaryInput{
					SurrenderNote: "Unable to proceed further",
				},
			},
			{
				name: "surrender_note_cannot_fix",
				input: BoundaryInput{
					SurrenderNote: "I cannot fix this issue",
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				dec := EvaluateBoundary(ladder, witness, tc.input)
				if dec.Action != ActionContinue {
					t.Fatalf("want ActionContinue, got %s", dec.Action)
				}
				if dec.ExitCode != 2 {
					t.Fatalf("want ExitCode 2, got %d", dec.ExitCode)
				}
				if dec.Signal != "PREMATURE_SURRENDER" {
					t.Fatalf("want Signal PREMATURE_SURRENDER, got %s", dec.Signal)
				}
				if dec.Reason != "PREMATURE_SURRENDER" {
					t.Fatalf("want Reason PREMATURE_SURRENDER, got %s", dec.Reason)
				}
				if dec.Disposition != DispClaimUnwitnessedContinue {
					t.Fatalf("want DispClaimUnwitnessedContinue, got %s", dec.Disposition)
				}
				if !dec.Blocked {
					t.Fatalf("want Blocked true, got false")
				}
				if dec.Stage != StageWarn {
					t.Fatalf("want StageWarn, got %s", dec.Stage)
				}
				if !strings.Contains(dec.Guidance, "premature surrender refused") {
					t.Fatalf("guidance missing 'premature surrender refused': %s", dec.Guidance)
				}
			})
		}
	})

	t.Run("surrender_note_with_verified_terminal_refusal_returns_allow", func(t *testing.T) {
		in := BoundaryInput{
			NotedSurrender: true,
			SurrenderNote:  "giving up",
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "POLICY_BLOCK",
				Verified: true,
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionAllow {
			t.Fatalf("want ActionAllow, got %s", dec.Action)
		}
		if dec.ExitCode != 0 {
			t.Fatalf("want ExitCode 0, got %d", dec.ExitCode)
		}
		if dec.Disposition != DispCleanWrapup {
			t.Fatalf("want DispCleanWrapup, got %s", dec.Disposition)
		}
		if dec.Stage != StageAllow {
			t.Fatalf("want StageAllow, got %s", dec.Stage)
		}
		if dec.Kind != KindClean {
			t.Fatalf("want KindClean, got %s", dec.Kind)
		}
	})

	t.Run("surrender_note_stands_down_after_max_blocks", func(t *testing.T) {
		in := BoundaryInput{
			NotedSurrender:    true,
			SurrenderNote:     "cannot fix",
			WitnessBlockCount: 3, // equals max=3
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionAllow {
			t.Fatalf("want ActionAllow, got %s", dec.Action)
		}
		if dec.ExitCode != 0 {
			t.Fatalf("want ExitCode 0, got %d", dec.ExitCode)
		}
		if dec.Stage != StageGiveUp {
			t.Fatalf("want StageGiveUp, got %s", dec.Stage)
		}
		if dec.Disposition != DispClaimUnwitnessedGiveUp {
			t.Fatalf("want DispClaimUnwitnessedGiveUp, got %s", dec.Disposition)
		}
		if dec.Kind != KindStandDown {
			t.Fatalf("want KindStandDown, got %s", dec.Kind)
		}
		if dec.Signal != "PREMATURE_SURRENDER" {
			t.Fatalf("want Signal PREMATURE_SURRENDER, got %s", dec.Signal)
		}
		if dec.Reason != "PREMATURE_SURRENDER" {
			t.Fatalf("want Reason PREMATURE_SURRENDER, got %s", dec.Reason)
		}
		if !strings.Contains(dec.OperatorMsg, "PREMATURE_SURRENDER stood down after 3 blocks") {
			t.Fatalf("operator message mismatch: %s", dec.OperatorMsg)
		}
	})
}

func TestStopgateGoalPersistence(t *testing.T) {
	ladder := DefaultLadderConfig()
	witness := WitnessGateConfig{Mode: ModeEnforce, Max: 3}

	t.Run("goal_active_without_completion_returns_continue", func(t *testing.T) {
		in := BoundaryInput{
			GoalActive:    true,
			GoalObjective: "implement premature surrender gate",
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionContinue {
			t.Fatalf("want ActionContinue, got %s", dec.Action)
		}
		if dec.ExitCode != 2 {
			t.Fatalf("want ExitCode 2, got %d", dec.ExitCode)
		}
		if dec.Signal != "PREMATURE_SURRENDER" {
			t.Fatalf("want Signal PREMATURE_SURRENDER, got %s", dec.Signal)
		}
		if dec.Reason != "PREMATURE_SURRENDER" {
			t.Fatalf("want Reason PREMATURE_SURRENDER, got %s", dec.Reason)
		}
		if dec.Disposition != DispClaimUnwitnessedContinue {
			t.Fatalf("want DispClaimUnwitnessedContinue, got %s", dec.Disposition)
		}
		if !dec.Blocked {
			t.Fatalf("want Blocked true, got false")
		}
		if dec.Stage != StageWarn {
			t.Fatalf("want StageWarn, got %s", dec.Stage)
		}
		if !strings.Contains(dec.Guidance, "implement premature surrender gate") {
			t.Fatalf("guidance must include goal objective: %s", dec.Guidance)
		}
	})

	t.Run("goal_active_with_terminal_refusal_returns_allow", func(t *testing.T) {
		in := BoundaryInput{
			GoalActive:    true,
			GoalObjective: "implement premature surrender gate",
			BoundaryRefusalReceipt: &BoundaryRefusalReceipt{
				Reason:   "POLICY_BLOCK",
				Verified: true,
			},
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionAllow {
			t.Fatalf("want ActionAllow, got %s", dec.Action)
		}
		if dec.ExitCode != 0 {
			t.Fatalf("want ExitCode 0, got %d", dec.ExitCode)
		}
		if dec.Disposition != DispCleanWrapup {
			t.Fatalf("want DispCleanWrapup, got %s", dec.Disposition)
		}
		if dec.Stage != StageAllow {
			t.Fatalf("want StageAllow, got %s", dec.Stage)
		}
		if dec.Kind != KindClean {
			t.Fatalf("want KindClean, got %s", dec.Kind)
		}
	})

	t.Run("goal_active_stands_down_after_max_blocks", func(t *testing.T) {
		in := BoundaryInput{
			GoalActive:        true,
			GoalObjective:     "implement premature surrender gate",
			WitnessBlockCount: 3, // equals max=3
		}
		dec := EvaluateBoundary(ladder, witness, in)
		if dec.Action != ActionAllow {
			t.Fatalf("want ActionAllow, got %s", dec.Action)
		}
		if dec.ExitCode != 0 {
			t.Fatalf("want ExitCode 0, got %d", dec.ExitCode)
		}
		if dec.Stage != StageGiveUp {
			t.Fatalf("want StageGiveUp, got %s", dec.Stage)
		}
		if dec.Disposition != DispClaimUnwitnessedGiveUp {
			t.Fatalf("want DispClaimUnwitnessedGiveUp, got %s", dec.Disposition)
		}
		if dec.Kind != KindStandDown {
			t.Fatalf("want KindStandDown, got %s", dec.Kind)
		}
		if dec.Signal != "PREMATURE_SURRENDER" {
			t.Fatalf("want Signal PREMATURE_SURRENDER, got %s", dec.Signal)
		}
		if dec.Reason != "PREMATURE_SURRENDER" {
			t.Fatalf("want Reason PREMATURE_SURRENDER, got %s", dec.Reason)
		}
		if !strings.Contains(dec.OperatorMsg, "PREMATURE_SURRENDER stood down after 3 blocks") {
			t.Fatalf("operator message mismatch: %s", dec.OperatorMsg)
		}
	})
}
