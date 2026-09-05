package stopgate

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// IsTransientHurdle returns true if the refusal reason or disposition represents
// a transient, retryable hurdle (such as concurrency lock contention, rate limiting, or malformed input)
// rather than an immovable policy boundary.
func IsTransientHurdle(reason, disposition string) bool {
	disp := strings.ToUpper(strings.TrimSpace(disposition))
	if disp == "TERMINAL" {
		return false
	}
	if disp == "RETRYABLE" || disp == "WAIT" {
		return true
	}
	r := strings.ToUpper(strings.TrimSpace(reason))
	switch r {
	case "LOCK_BUSY", "COLLISION_RISK", "MERGE_IN_PROGRESS", "RATE_LIMITED", "LEASE_HELD",
		"MISROUTE", "MALFORMED", "SHELL_DIALECT", "MESSAGE_RACE", "BUILD_CACHE_CLEAN_RACE",
		"INTERACTIVE_HANG", "DISAMBIGUATION_TIMEOUT", "HOST_CHURN_BACKOFF", "CRASH_RESTART_EXHAUSTED":
		return true
	}
	if code, ok := abi.ReasonByName(r); ok {
		switch code {
		case abi.ReasonLeaseHeld, abi.ReasonRateLimited, abi.ReasonMisroute, abi.ReasonMalformed, abi.ReasonShellDialect:
			return true
		}
	}
	return false
}

// IsTerminalBoundary returns true if the receipt records a verified terminal boundary refusal
// (e.g. POLICY_BLOCK, TRUST_VIOLATION, or explicit TERMINAL disposition).
func IsTerminalBoundary(receipt *BoundaryRefusalReceipt) bool {
	if receipt == nil || !receipt.Verified {
		return false
	}
	if receipt.Transient {
		return false
	}
	if IsTransientHurdle(receipt.Reason, receipt.Disposition) {
		return false
	}
	disp := strings.ToUpper(strings.TrimSpace(receipt.Disposition))
	if receipt.Terminal || disp == "TERMINAL" {
		return true
	}
	r := strings.ToUpper(strings.TrimSpace(receipt.Reason))
	switch r {
	case "POLICY_BLOCK", "DEFAULT_DENY", "SELF_MODIFY", "CORE_SELF_MODIFY",
		"TRUST_VIOLATION", "SECRET_EXFIL", "PII_EXFIL", "TAINT_EGRESS",
		"SCOPE_CROSSING", "PROMPT_INJECTION", "INTEGRITY_REFUTED", "EGRESS_BLOCK",
		"OUT_OF_TREE_WRITE", "FILE_ADMISSION", "PUBLIC_LEAK", "ARCH_LAYER_VIOLATION",
		"NEVER_AMEND_SHARED", "OFF_TRUNK":
		return true
	}
	if receipt.ReasonCode != abi.ReasonNone {
		switch receipt.ReasonCode {
		case abi.ReasonPolicyBlock, abi.ReasonSelfModify, abi.ReasonTrustViolation,
			abi.ReasonSecretExfil, abi.ReasonDefaultDeny, abi.ReasonPIIExfil,
			abi.ReasonTaintEgress, abi.ReasonScopeCrossing, abi.ReasonPromptInjection,
			abi.ReasonIntegrityRefuted:
			return true
		}
	}
	if code, ok := abi.ReasonByName(r); ok {
		switch code {
		case abi.ReasonPolicyBlock, abi.ReasonSelfModify, abi.ReasonTrustViolation,
			abi.ReasonSecretExfil, abi.ReasonDefaultDeny, abi.ReasonPIIExfil,
			abi.ReasonTaintEgress, abi.ReasonScopeCrossing, abi.ReasonPromptInjection,
			abi.ReasonIntegrityRefuted:
			return true
		}
	}
	return false
}

func getReceipt(in BoundaryInput) *BoundaryRefusalReceipt {
	if in.RefusalReceipt != nil {
		return in.RefusalReceipt
	}
	if in.BoundaryRefusalReceipt != nil {
		return in.BoundaryRefusalReceipt
	}
	if in.RefusalToken != "" || in.ReasonCode != abi.ReasonNone {
		r := in.RefusalToken
		if r == "" && in.ReasonCode != abi.ReasonNone {
			r = abi.ReasonName(in.ReasonCode)
		}
		return &BoundaryRefusalReceipt{
			Reason:     r,
			ReasonCode: in.ReasonCode,
			Verified:   true,
		}
	}
	return nil
}

// HasVerifiedTerminalBoundaryRefusal checks whether BoundaryInput presents verified
// non-transient boundary refusal evidence (such as POLICY_BLOCK or an explicitly
// terminal verified receipt). If unverified or transient, returns false with detail.
func (in BoundaryInput) HasVerifiedTerminalBoundaryRefusal() (bool, string) {
	rc := getReceipt(in)
	if rc == nil {
		return false, "missing verified boundary refusal receipt"
	}
	if !rc.Verified && rc.Token == "" {
		return false, "unverified boundary refusal receipt"
	}
	if IsTerminalBoundary(rc) {
		return true, ""
	}
	reason := rc.Reason
	if reason == "" && rc.ReasonCode != abi.ReasonNone {
		reason = abi.ReasonName(rc.ReasonCode)
	}
	if IsTransientHurdle(reason, rc.Disposition) {
		return false, "transient refusal (" + reason + ") cannot witness no allowed path without terminal receipt"
	}
	return false, "boundary refusal reason (" + reason + ") not recognized as terminal boundary"
}

func isTransientReason(reason string, code abi.ReasonCode) bool {
	switch code {
	case abi.ReasonLeaseHeld, abi.ReasonRateLimited, abi.ReasonMisroute, abi.ReasonMalformed, abi.ReasonShellDialect:
		return true
	}
	return IsTransientHurdle(reason, "")
}

func isTerminalBoundaryReason(reason string, code abi.ReasonCode) bool {
	rc := &BoundaryRefusalReceipt{
		Reason:     reason,
		ReasonCode: code,
		Verified:   true,
	}
	return IsTerminalBoundary(rc)
}

// EvaluateBoundary unifies turn-boundary lifecycle adjudication across harness architectures.
func EvaluateBoundary(ladder LadderConfig, witnessCfg WitnessGateConfig, in BoundaryInput) Decision {
	// 1. Witness check gate: if FinalGate is unsatisfied or WitnessClaim is unwitnessed,
	// NotedNoAllowedPath must NOT bypass the witness gate.
	if in.FinalGate != nil {
		satisfied, missing := in.FinalGate()
		if !satisfied {
			claim := WitnessClaim{
				Claimed:   true,
				Witnessed: false,
				Reason:    "STOP_UNWITNESSED",
				Detail:    missing,
			}
			fgWitnessCfg := witnessCfg
			if fgWitnessCfg.Mode == "" || fgWitnessCfg.Mode == ModeShadow {
				fgWitnessCfg.Mode = ModeEnforce
			}
			dec := EvaluateWitness(fgWitnessCfg, claim, in.WitnessBlockCount)
			if dec.ShouldContinue() {
				return dec
			}
			if dec.Disposition != DispCleanCompletion && dec.Disposition != DispClaimWitnessed {
				return dec
			}
		}
	} else if in.WitnessClaim != nil && in.WitnessClaim.Claimed {
		dec := EvaluateWitness(witnessCfg, *in.WitnessClaim, in.WitnessBlockCount)
		if dec.ShouldContinue() {
			return dec
		}
		if dec.Disposition != DispCleanCompletion && dec.Disposition != DispClaimWitnessed {
			return dec
		}
	}

	// 2. Clean wrap-up: if agent explicitly noted "no allowed path", require verified terminal
	// boundary evidence (either IsTerminalBoundary(in.RefusalReceipt) or (in.WitnessClaim != nil && in.WitnessClaim.Witnessed)).
	// If neither is present, treat as STOP_UNWITNESSED returning ActionContinue, ExitCode: 2, Signal: "STOP_UNWITNESSED".
	// If verified terminal boundary receipt is present (e.g. POLICY_BLOCK), admit DispCleanWrapup with ExitCode: 0.
	if in.NotedNoAllowedPath {
		rc := getReceipt(in)
		isTerminal := IsTerminalBoundary(rc)
		isWitnessed := in.WitnessClaim != nil && in.WitnessClaim.Witnessed

		if !isTerminal && !isWitnessed {
			reason := ""
			disp := ""
			if rc != nil {
				reason = rc.Reason
				if reason == "" && rc.ReasonCode != abi.ReasonNone {
					reason = abi.ReasonName(rc.ReasonCode)
				}
				disp = rc.Disposition
			}
			isTransient := IsTransientHurdle(reason, disp)
			guidance := NoAllowedPathContinuationMessage(reason, isTransient)

			max := witnessCfg.Max
			if max < 1 {
				max = 3
			}
			if in.WitnessBlockCount >= max {
				opMsg := fmt.Sprintf("fak guard Stop: STOP_UNWITNESSED stood down after %d blocks (bounded max=%d); allowing stop", in.WitnessBlockCount, max)
				return Decision{
					Action:      ActionAllow,
					Stage:       StageGiveUp,
					Disposition: DispClaimUnwitnessedGiveUp,
					Kind:        KindStandDown,
					Blocked:     false,
					ExitCode:    0,
					Signal:      "STOP_UNWITNESSED",
					Reason:      "STOP_UNWITNESSED",
					Depth:       in.WitnessBlockCount,
					Bound:       max,
					OperatorMsg: opMsg,
					Note:        opMsg,
				}
			}

			return Decision{
				Action:      ActionContinue,
				Stage:       StageWarn,
				Disposition: DispClaimUnwitnessedContinue,
				Kind:        KindContinue,
				Blocked:     true,
				ExitCode:    2,
				Signal:      "STOP_UNWITNESSED",
				Reason:      "STOP_UNWITNESSED",
				Depth:       in.WitnessBlockCount + 1,
				Bound:       max,
				Guidance:    guidance,
				Note:        guidance,
			}
		}

		return Decision{
			Action:      ActionAllow,
			Stage:       StageAllow,
			Disposition: DispCleanWrapup,
			Kind:        KindClean,
			ExitCode:    0,
			Signal:      "clean",
		}
	}

	// 3. Tool feedback check: if consecutive deny-all is 0 but tool feedback consecutive > 0.
	if in.ConsecutiveDenyAll <= 0 && in.ConsecutiveToolFeedback > 0 {
		return EvaluateToolFeedback(ladder, in.ConsecutiveToolFeedback)
	}

	// 4. Deny-all check: evaluate graduated back-off ladder.
	if in.ConsecutiveDenyAll > 0 {
		return EvaluateDenyAll(ladder, in.ConsecutiveDenyAll, in.ConsecutiveSameIssue, in.UseSameIssue)
	}

	// 5. Default clean completion.
	return Decision{
		Action:      ActionAllow,
		Stage:       StageAllow,
		Disposition: DispCleanCompletion,
		Kind:        KindClean,
		ExitCode:    0,
		Signal:      "clean",
	}
}
