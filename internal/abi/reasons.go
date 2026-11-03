// reasons.go — the CLOSED core refusal vocabulary (the model's label space).
//
// A small, closed, additive set mirroring DOS dos_refuse_reasons: a refusal MUST
// cite one of these, never free text, and a later minor may add codes without
// breaking models trained on the set.
package abi

import "strconv"

const (
	// ReasonNone is the zero value: no refusal was issued, the call passed.
	ReasonNone ReasonCode = iota
	// ReasonDefaultDeny fires when no policy affirmatively permits the call; denial stands until one does (deny-first posture).
	ReasonDefaultDeny
	// ReasonPolicyBlock is a structural refusal: the policy manifest denies the tool or args outright, no model judgment involved.
	ReasonPolicyBlock
	// ReasonSelfModify refuses calls that would alter the agent or kernel itself; structural, never model-fixable.
	ReasonSelfModify
	// ReasonLeaseHeld means a conflicting lease still holds the tree; retry once it releases (dos arbitrate).
	ReasonLeaseHeld
	// ReasonTrustViolation flags a taint or scope breach across shared results; a structural isolation failure.
	ReasonTrustViolation
	// ReasonMalformed rejects a call that fails the grammar or arity rung; model-fixable by reshaping the arguments.
	ReasonMalformed
	// ReasonMisroute means the wrong tool or argument shape was chosen; model-fixable by picking the right tool.
	ReasonMisroute
	// ReasonRateLimited throttles the caller past its budget window; transient, retry after the quota refills.
	ReasonRateLimited
	// ReasonSecretExfil seals a result or args matching a secret pattern; the egress verdict, distinct from discovery.
	ReasonSecretExfil
	// ReasonUnwitnessed fires when the require-witness gate finds no corroborating witness; supply one to proceed.
	ReasonUnwitnessed
	// ReasonOversize rejects a payload past the context-admission budget; shrink it to fit.
	ReasonOversize
	// ReasonUnknownTool names a tool absent from the registry; model-fixable by calling a registered tool.
	ReasonUnknownTool
	// ReasonSecretDiscovered catches a tool result bearing a secret on discovery; the on-discovery twin of SECRET_EXFIL [#884].
	ReasonSecretDiscovered
	// ReasonSecretRedacted masks a credential span in a tool result in place (warn-first default); the rest stays in context, unlike the SECRET_EXFIL seal.
	ReasonSecretRedacted
	// ReasonShellDialect rejects a command in the wrong shell dialect for its tool (a PowerShell cmdlet sent to the POSIX Bash tool); model-fixable by re-routing (#3941).
	ReasonShellDialect
	// ReasonPIIRedacted masks a general-PII span (email, phone, national ID, PAN, IBAN) in a tool result in place under the warn-first default; the rest of the result stays in context; the PII twin of SECRET_REDACTED [#5378].
	ReasonPIIRedacted
	// ReasonPIIExfil seals the whole result when a general-PII pattern matches under the deny-first posture or an obfuscation-only hit; the PII twin of SECRET_EXFIL [#5378].
	ReasonPIIExfil
	// ReasonTaintEgress blocks a sensitive-sink call carrying tainted data, or a result past the taint ceiling; structural.
	ReasonTaintEgress
	// ReasonScopeCrossing fires when a call steps outside its granted scope boundary; structural, denied regardless of intent.
	ReasonScopeCrossing
	// ReasonPromptInjection flags prompt-injection markers in a tool result or payload; treated as untrusted input.
	ReasonPromptInjection
	// ReasonIntegrityRefuted means the witness resolver actively refuted the claimed effect; unlike UNWITNESSED, the proof contradicts the claim.
	ReasonIntegrityRefuted
	// ReasonSpeculationMispredict marks a speculative tool call that was mispredicted or squashed before commit.
	ReasonSpeculationMispredict
	// ReasonSpeculativeEffectLeak fires when a squashed speculative call leaked a side effect past its rollback boundary; structural.
	ReasonSpeculativeEffectLeak
	// 24..1023 reserved for additive core reasons; register out-of-tree names via RegisterReason.
	ReasonCoreMax ReasonCode = 1023
)

var coreReasonNames = map[ReasonCode]string{
	ReasonNone:                  "NONE",
	ReasonDefaultDeny:           "DEFAULT_DENY",
	ReasonPolicyBlock:           "POLICY_BLOCK",
	ReasonSelfModify:            "SELF_MODIFY",
	ReasonLeaseHeld:             "LEASE_HELD",
	ReasonTrustViolation:        "TRUST_VIOLATION",
	ReasonMalformed:             "MALFORMED",
	ReasonMisroute:              "MISROUTE",
	ReasonRateLimited:           "RATE_LIMITED",
	ReasonSecretExfil:           "SECRET_EXFIL",
	ReasonUnwitnessed:           "UNWITNESSED",
	ReasonOversize:              "OVERSIZE",
	ReasonUnknownTool:           "UNKNOWN_TOOL",
	ReasonSecretDiscovered:      "RESULT_SECRET_DISCOVERED",
	ReasonSecretRedacted:        "SECRET_REDACTED",
	ReasonShellDialect:          "SHELL_DIALECT",
	ReasonPIIRedacted:           "PII_REDACTED",
	ReasonPIIExfil:              "PII_EXFIL",
	ReasonTaintEgress:           "TAINT_EGRESS",
	ReasonScopeCrossing:         "SCOPE_CROSSING",
	ReasonPromptInjection:       "PROMPT_INJECTION",
	ReasonIntegrityRefuted:      "INTEGRITY_REFUTED",
	ReasonSpeculationMispredict: "SPECULATION_MISPREDICT",
	ReasonSpeculativeEffectLeak: "SPECULATIVE_EFFECT_LEAK",
}

// ReasonName resolves a reason code to its stable name. Unknown codes render as
// REASON_<n> rather than panicking (forward-compat).
func ReasonName(c ReasonCode) string {
	if n, ok := coreReasonNames[c]; ok {
		return n
	}
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if n, ok := reg.reasons[c]; ok {
		return n
	}
	return "REASON_" + strconv.FormatUint(uint64(c), 10)
}

// reasonCodesByName is the reverse of coreReasonNames (closed core set only; registered names resolve live in ReasonByName).
var reasonCodesByName = func() map[string]ReasonCode {
	m := make(map[string]ReasonCode, len(coreReasonNames))
	for c, n := range coreReasonNames {
		m[n] = c
	}
	return m
}()

// ReasonByName resolves a stable refusal name (e.g. "POLICY_BLOCK") to its code,
// closed core set first then registered names. Callers loading operator-authored
// refusals MUST reject an unknown name rather than coerce it, so every deny still
// cites a code from the closed vocabulary. Additive to the frozen ABI.
func ReasonByName(name string) (ReasonCode, bool) {
	if c, ok := reasonCodesByName[name]; ok {
		return c, true
	}
	reg.mu.Lock()
	defer reg.mu.Unlock()
	for c, n := range reg.reasons {
		if n == name {
			return c, true
		}
	}
	return ReasonNone, false
}

// ReasonNames returns the closed core refusal vocabulary as a sorted slice of
// names (excluding NONE) — the valid value space for a policy manifest's deny reasons.
func ReasonNames() []string {
	out := make([]string, 0, len(coreReasonNames))
	for c, n := range coreReasonNames {
		if c == ReasonNone {
			continue
		}
		out = append(out, n)
	}
	sortStrings(out)
	return out
}

// sortStrings is a tiny insertion sort that keeps this file import-free; the closed vocabulary is small enough that it is plenty.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

// CoreReasonCount is the size of the closed core vocabulary (excludes NONE); tests assert the closed reason set against it.
const CoreReasonCount = 23
