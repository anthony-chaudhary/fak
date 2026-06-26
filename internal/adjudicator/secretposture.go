package adjudicator

import (
	"regexp"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// secretposture.go is the secret-detection POSTURE (epic #880 pillar [B], issue
// #885): the policy-author knob that decides what the on-discovery secret rung
// (internal/secretgate, #884) does when a tool result bears a credential. It reuses
// the tool-admission posture grammar (fail_closed / admit_and_log) and adds the
// secret-specific default `quarantine` (= today's behavior), so a manifest can say
// "in this unattended batch, quarantine discovered secrets and keep going" vs "fail
// closed" vs "admit and log" — and declare EXTRA secret shapes unioned with the
// canon floor. The gate reads it at admit time via Adjudicator.SecretPolicy.

// SecretPosture selects the on-discovery secret verdict. The zero value is
// SecretQuarantine — today's behavior exactly — so the posture is additive.
type SecretPosture uint8

const (
	// SecretQuarantine holds the secret-bearing result out of context and continues
	// (the default, = the pre-#885 behavior).
	SecretQuarantine SecretPosture = iota
	// SecretFailClosed hard-denies on discovery.
	SecretFailClosed
	// SecretAdmitAndLog admits a read-shaped result while recording the would-deny.
	SecretAdmitAndLog
)

// String renders the posture's stable manifest token.
func (p SecretPosture) String() string {
	switch p {
	case SecretFailClosed:
		return "fail_closed"
	case SecretAdmitAndLog:
		return "admit_and_log"
	default:
		return "quarantine"
	}
}

// ParseSecretPosture maps a manifest token to a posture. An empty token is the
// default (quarantine). The bool reports whether the token was recognized; an
// unknown token MUST be rejected at load, never silently coerced — the same
// fail-loud contract the tool-admission posture parse has.
func ParseSecretPosture(s string) (SecretPosture, bool) {
	switch s {
	case "", "quarantine":
		return SecretQuarantine, true
	case "fail_closed":
		return SecretFailClosed, true
	case "admit_and_log":
		return SecretAdmitAndLog, true
	default:
		return SecretQuarantine, false
	}
}

// MatchesDeclaredSecret reports whether body matches any policy-declared EXTRA
// secret pattern. It is the declared half only — the canon.SecretPatterns floor is
// applied by the gate, which UNIONS the two (extend, never replace). An empty
// declared set is false (the floor stands alone).
func (p Policy) MatchesDeclaredSecret(body []byte) bool {
	for _, re := range p.SecretPatterns {
		if re != nil && re.Match(body) {
			return true
		}
	}
	return false
}

// Verdict maps the posture to the verdict the gate returns on a discovered secret
// in tool's result:
//   - quarantine  -> hold it out of context (page-out), the default;
//   - fail_closed -> a provable Deny;
//   - admit_and_log -> admit a READ-SHAPED result with the would-deny recorded in
//     Meta; a non-read-shaped tool falls back to quarantine (fail-safe: never admit
//     a secret from a write/exec-shaped result just because the posture is lenient).
//
// Every verdict cites ReasonSecretDiscovered (the on-discovery event), keeping
// ReasonSecretExfil for the egress verdict (the #152 decoupling). The gate
// (internal/secretgate) calls this directly off the posture it reads.
func (p SecretPosture) Verdict(tool string) abi.Verdict {
	switch p {
	case SecretFailClosed:
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonSecretDiscovered, By: "secretgate"}
	case SecretAdmitAndLog:
		if lowRiskReadShaped(tool) {
			return abi.Verdict{Kind: abi.VerdictAllow, Reason: abi.ReasonSecretDiscovered, By: "secretgate",
				Meta: map[string]string{"posture": "admit_and_log", "would_deny": "RESULT_SECRET_DISCOVERED"}}
		}
		fallthrough
	default: // SecretQuarantine + the non-read-shaped admit_and_log fallback
		return abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonSecretDiscovered, By: "secretgate",
			Payload: abi.QuarantinePayload{PageOut: true}}
	}
}

// SecretVerdict is the Policy-level shorthand for p.SecretPosture.Verdict(tool).
func (p Policy) SecretVerdict(tool string) abi.Verdict { return p.SecretPosture.Verdict(tool) }

// SecretPolicy returns the active secret posture + declared patterns, read under the
// policy lock so a concurrent SetPolicy is safe. The on-discovery secret rung calls
// this at admit time; with no manifest loaded the zero posture (quarantine) + nil
// patterns reproduce today's behavior.
func (a *Adjudicator) SecretPolicy() (SecretPosture, []*regexp.Regexp) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.policy.SecretPosture, a.policy.SecretPatterns
}
