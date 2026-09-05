package stopgate

import (
	"fmt"
	"strings"
)

// WitnessGateConfig configures the witnessed completion gate.
type WitnessGateConfig struct {
	Mode Mode
	Max  int
}

// DefaultWitnessGateConfig returns the default witness gate configuration.
func DefaultWitnessGateConfig() WitnessGateConfig {
	return WitnessGateConfig{
		Mode: ModeShadow,
		Max:  3,
	}
}

// EvaluateWitness evaluates an asserted completion claim against declared witness rules.
func EvaluateWitness(cfg WitnessGateConfig, claim WitnessClaim, blockCount int) Decision {
	mode := cfg.Mode
	if mode == "" {
		mode = ModeShadow
	}
	max := cfg.Max
	if max < 1 {
		max = 3
	}

	if !claim.Claimed {
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispCleanCompletion,
			Kind:        KindClean,
			ExitCode:    0,
			Signal:      "witness",
		}
	}

	if claim.Witnessed {
		note := "CLAIM_WITNESSED"
		if claim.Commit != "" {
			note = fmt.Sprintf("fak guard Stop: CLAIM_WITNESSED commit=%s — %s", claim.Commit, claim.Detail)
		}
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispClaimWitnessed,
			Kind:        KindClean,
			ExitCode:    0,
			Signal:      "witness",
			Reason:      claim.Reason,
			Note:        note,
		}
	}

	if mode == ModeOff {
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispModeOff,
			Kind:        KindOff,
			ExitCode:    0,
			Signal:      "witness",
			Reason:      claim.Reason,
		}
	}

	if mode != ModeEnforce {
		guidance := fmt.Sprintf("fak guard Stop: shadow %s — %s; would require a witnessed stamped commit before stopping", claim.Reason, claim.Detail)
		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispClaimWitnessShadow,
			Kind:        KindShadow,
			Blocked:     false,
			ExitCode:    0,
			Signal:      claim.Reason,
			Reason:      claim.Reason,
			Guidance:    guidance,
			Note:        guidance,
		}
	}

	// Enforce mode
	seq := blockCount + 1
	if seq > max {
		opMsg := fmt.Sprintf("fak guard Stop: %s stood down after %d blocks (bounded max=%d); allowing stop", claim.Reason, seq-1, max)
		return Decision{
			Action:      ActionAllow,
			Stage:       StageGiveUp,
			Disposition: DispClaimUnwitnessedGiveUp,
			Kind:        KindStandDown,
			Blocked:     false,
			ExitCode:    0,
			Signal:      claim.Reason,
			Reason:      claim.Reason,
			Depth:       seq - 1,
			Bound:       max,
			OperatorMsg: opMsg,
			Note:        opMsg,
		}
	}

	var guidance string
	if claim.Reason == "STOP_UNWITNESSED" {
		if strings.HasPrefix(claim.Detail, "missing declared witness: ") {
			guidance = fmt.Sprintf("STOP_UNWITNESSED: %s. Continue working until that witness exists.", claim.Detail)
		} else {
			guidance = fmt.Sprintf("STOP_UNWITNESSED: missing declared witness: %s. Continue working until that witness exists.", claim.Detail)
		}
	} else {
		guidance = fmt.Sprintf("fak guard Stop: %s (%d/%d) — %s. Do not stop on self-narrated completion: commit the coherent change with a bindable (fak <leaf>) stamp, run its witness, and report that commit.", claim.Reason, seq, max, claim.Detail)
	}

	return Decision{
		Action:      ActionContinue,
		Stage:       StageWarn,
		Disposition: DispClaimUnwitnessedContinue,
		Kind:        KindContinue,
		Blocked:     true,
		ExitCode:    2,
		Signal:      claim.Reason,
		Reason:      claim.Reason,
		Depth:       seq,
		Bound:       max,
		Guidance:    guidance,
		Note:        guidance,
	}
}
