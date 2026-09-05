package stopgate

import (
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
