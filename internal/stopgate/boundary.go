package stopgate

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// HasVerifiedTerminalBoundaryRefusal checks whether BoundaryInput presents verified
// non-transient boundary refusal evidence (such as POLICY_BLOCK or an explicitly
// terminal verified receipt). If unverified or transient, returns false with detail.
func (in BoundaryInput) HasVerifiedTerminalBoundaryRefusal() (bool, string) {
	if in.BoundaryRefusalReceipt != nil {
		rc := in.BoundaryRefusalReceipt
		if rc.Transient {
			r := rc.Reason
			if r == "" && rc.ReasonCode != abi.ReasonNone {
				r = abi.ReasonName(rc.ReasonCode)
			}
			if r == "" {
				r = "transient_hurdle"
			}
			return false, "transient refusal (" + r + ") cannot witness no allowed path without terminal receipt"
		}
		if !rc.Verified && rc.Token == "" {
			return false, "unverified boundary refusal receipt"
		}
		if rc.Terminal {
			return true, ""
		}
		reason := rc.Reason
		code := rc.ReasonCode
		if code == abi.ReasonNone && reason != "" {
			code, _ = abi.ReasonByName(reason)
		}
		if isTransientReason(reason, code) {
			r := reason
			if r == "" && code != abi.ReasonNone {
				r = abi.ReasonName(code)
			}
			return false, "transient refusal (" + r + ") cannot witness no allowed path without terminal receipt"
		}
		if isTerminalBoundaryReason(reason, code) {
			return true, ""
		}
		if reason == "" && code == abi.ReasonNone {
			return false, "boundary refusal receipt missing reason code"
		}
		return false, "boundary refusal reason (" + reason + ") not recognized as terminal boundary"
	}

	if in.ReasonCode != abi.ReasonNone {
		if isTransientReason("", in.ReasonCode) {
			return false, "transient refusal (" + abi.ReasonName(in.ReasonCode) + ") cannot witness no allowed path without terminal receipt"
		}
		if isTerminalBoundaryReason("", in.ReasonCode) {
			return true, ""
		}
		return false, "refusal code (" + abi.ReasonName(in.ReasonCode) + ") not recognized as terminal boundary"
	}

	if in.RefusalToken != "" {
		code, _ := abi.ReasonByName(in.RefusalToken)
		if isTransientReason(in.RefusalToken, code) {
			return false, "transient refusal token (" + in.RefusalToken + ") cannot witness no allowed path without terminal receipt"
		}
		if isTerminalBoundaryReason(in.RefusalToken, code) {
			return true, ""
		}
		return false, "refusal token (" + in.RefusalToken + ") not recognized as terminal boundary"
	}

	return false, "missing verified boundary refusal receipt"
}

func isTransientReason(reason string, code abi.ReasonCode) bool {
	switch code {
	case abi.ReasonLeaseHeld, abi.ReasonRateLimited, abi.ReasonMisroute, abi.ReasonMalformed, abi.ReasonShellDialect:
		return true
	}
	r := strings.ToUpper(strings.TrimSpace(reason))
	switch r {
	case "LOCK_BUSY", "COLLISION_RISK", "MERGE_IN_PROGRESS", "RATE_LIMITED", "LEASE_HELD",
		"MISROUTE", "MALFORMED", "SHELL_DIALECT", "MESSAGE_RACE", "BUILD_CACHE_CLEAN_RACE",
		"INTERACTIVE_HANG", "DISAMBIGUATION_TIMEOUT", "HOST_CHURN_BACKOFF", "CRASH_RESTART_EXHAUSTED":
		return true
	}
	return false
}

func isTerminalBoundaryReason(reason string, code abi.ReasonCode) bool {
	switch code {
	case abi.ReasonPolicyBlock, abi.ReasonSelfModify, abi.ReasonTrustViolation,
		abi.ReasonSecretExfil, abi.ReasonDefaultDeny, abi.ReasonPIIExfil,
		abi.ReasonTaintEgress, abi.ReasonScopeCrossing, abi.ReasonPromptInjection,
		abi.ReasonIntegrityRefuted:
		return true
	}
	r := strings.ToUpper(strings.TrimSpace(reason))
	switch r {
	case "POLICY_BLOCK", "DEFAULT_DENY", "SELF_MODIFY", "CORE_SELF_MODIFY",
		"TRUST_VIOLATION", "SECRET_EXFIL", "PII_EXFIL", "TAINT_EGRESS",
		"SCOPE_CROSSING", "PROMPT_INJECTION", "INTEGRITY_REFUTED", "EGRESS_BLOCK",
		"OUT_OF_TREE_WRITE", "FILE_ADMISSION", "PUBLIC_LEAK", "ARCH_LAYER_VIOLATION",
		"NEVER_AMEND_SHARED", "OFF_TRUNK":
		return true
	}
	return false
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

	// 2. Clean wrap-up: if agent explicitly noted "no allowed path", that is a sanctioned clean stop
	// only when accompanied by verified terminal boundary refusal evidence (e.g. POLICY_BLOCK or
	// a verified terminal receipt). If claimed without verified boundary evidence or on transient
	// hurdles like LOCK_BUSY without a terminal refusal receipt, treat as STOP_UNWITNESSED
	// (Disposition != DispCleanWrapup).
	if in.NotedNoAllowedPath {
		hasBoundary, detail := in.HasVerifiedTerminalBoundaryRefusal()
		if !hasBoundary {
			claim := WitnessClaim{
				Claimed:   true,
				Witnessed: false,
				Reason:    "STOP_UNWITNESSED",
				Detail:    detail,
			}
			if in.WitnessClaim != nil && in.WitnessClaim.Claimed {
				claim = *in.WitnessClaim
				claim.Witnessed = false
				if claim.Reason == "" {
					claim.Reason = "STOP_UNWITNESSED"
				}
				if claim.Detail == "" {
					claim.Detail = detail
				}
			}
			dec := EvaluateWitness(witnessCfg, claim, in.WitnessBlockCount)
			if dec.ShouldContinue() {
				return dec
			}
			if dec.Disposition != DispCleanCompletion && dec.Disposition != DispClaimWitnessed {
				return dec
			}
			// Fail-closed fallback: ensure Disposition is never DispCleanWrapup when unverified.
			if dec.Disposition == DispCleanCompletion || dec.Disposition == DispCleanWrapup {
				dec.Disposition = DispClaimUnwitnessedContinue
				dec.Action = ActionContinue
				dec.ExitCode = 2
				dec.Blocked = true
				return dec
			}
			return dec
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
