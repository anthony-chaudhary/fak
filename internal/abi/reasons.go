// reasons.go — the CLOSED core refusal vocabulary (the model's label space).
//
// This is the v0.1 realization of ReasonCode (declared in types.go): a small,
// closed, additive set mirroring DOS dos_refuse_reasons. A refusal MUST cite one
// of these — never free text — so a deny is verifiable and the deny-loopback can
// derive a disposition from the reason's category. Additive-only: a model trained
// on this set degrades gracefully when a later minor adds a code.
package abi

const (
	ReasonNone             ReasonCode = iota // not a refusal
	ReasonDefaultDeny                        // no policy affirmatively allowed it (fail-closed)
	ReasonPolicyBlock                        // an explicit policy rule denied it
	ReasonSelfModify                         // the call would modify the agent/kernel itself
	ReasonLeaseHeld                          // a file-tree lease conflict (dos arbitrate)
	ReasonTrustViolation                     // taint/scope violation (shared-result isolation)
	ReasonMalformed                          // failed a grammar / arity well-formedness rung
	ReasonMisroute                           // wrong tool or arg shape — MODEL-FIXABLE
	ReasonRateLimited                        // throttled; retry after a wait
	ReasonSecretExfil                        // result/args matched a secret pattern
	ReasonUnwitnessed                        // require-witness gate had no corroboration
	ReasonOversize                           // payload exceeded the context-admission budget
	ReasonUnknownTool                        // tool not in the registry
	ReasonSecretDiscovered                   // a tool RESULT bore a secret, caught on discovery (the on-discovery event; distinct from ReasonSecretExfil, the egress verdict) [#884]
	// 15.. reserved for additive core reasons; register out-of-tree names via
	// RegisterReason.
	ReasonCoreMax ReasonCode = 1023
)

var coreReasonNames = map[ReasonCode]string{
	ReasonNone:             "NONE",
	ReasonDefaultDeny:      "DEFAULT_DENY",
	ReasonPolicyBlock:      "POLICY_BLOCK",
	ReasonSelfModify:       "SELF_MODIFY",
	ReasonLeaseHeld:        "LEASE_HELD",
	ReasonTrustViolation:   "TRUST_VIOLATION",
	ReasonMalformed:        "MALFORMED",
	ReasonMisroute:         "MISROUTE",
	ReasonRateLimited:      "RATE_LIMITED",
	ReasonSecretExfil:      "SECRET_EXFIL",
	ReasonUnwitnessed:      "UNWITNESSED",
	ReasonOversize:         "OVERSIZE",
	ReasonUnknownTool:      "UNKNOWN_TOOL",
	ReasonSecretDiscovered: "RESULT_SECRET_DISCOVERED",
}

// ReasonName resolves a reason code to its stable name, consulting the closed
// core set first then the registered (out-of-tree) names. Unknown codes render as
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
	return "REASON_" + itoa(uint64(c))
}

// reasonCodesByName is the reverse of coreReasonNames, built once at init. It
// covers only the closed core set; registered (out-of-tree) names are resolved
// live in ReasonByName.
var reasonCodesByName = func() map[string]ReasonCode {
	m := make(map[string]ReasonCode, len(coreReasonNames))
	for c, n := range coreReasonNames {
		m[n] = c
	}
	return m
}()

// ReasonByName is the inverse of ReasonName: it resolves a stable refusal name
// (e.g. "POLICY_BLOCK") to its code, consulting the closed core set first then
// the registered names. The bool reports whether the name was known — a caller
// that loads an operator-authored refusal (a deployable policy manifest) MUST
// reject an unknown name rather than coerce it, so every deny still cites a code
// from the closed vocabulary. Additive to the frozen ABI (no existing signature
// changes).
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
// names (excluding NONE) — the valid value space for a policy manifest's deny
// reasons and the lint surface for an operator authoring one.
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

// sortStrings is a tiny insertion sort that keeps this file import-free (the
// list is the 12-name closed vocabulary; an insertion sort is plenty).
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

// CoreReasonCount is the size of the closed core vocabulary (excludes NONE) —
// referenced by tests asserting the closed reason set.
const CoreReasonCount = 13

func itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
