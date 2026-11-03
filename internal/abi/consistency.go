// consistency.go — the per-call CONSISTENCY LEVEL (#1317), a closed enum threaded through
// ToolCall.Meta: the policy input + audit field a best-effort or speculative serve mode
// keys on. You cannot honestly claim a relaxed serve mode without an EXPLICIT level on the
// call (never a hidden cache mode); ConsistencyOf defaults an unset/unknown value to the
// STRICTEST level, so the field is purely additive and the default path is byte-for-byte
// unchanged. It rides the OPEN Meta map (not a new typed field) to stay wire-transparent.

package abi

import "strings"

// ConsistencyLevel is the CLOSED relaxation ladder a call declares for its read of the
// world. STRICT is the floor (the default): the call sees the committed world and a
// relaxed mode never silently applies; the ladder loosens monotonically (more staleness).
type ConsistencyLevel uint8

const (
	// ConsistencyStrict reads the committed world: no stale data, no speculation. The
	// default for any undeclared call, and the fail-safe an unknown token resolves to.
	ConsistencyStrict ConsistencyLevel = iota
	// ConsistencyBoundedStale tolerates a bounded-age cached read (the staleness bound is a separate policy input; this level only declares the call accepts one).
	ConsistencyBoundedStale
	// ConsistencyBestEffort tolerates an arbitrarily stale read served locally if a fresh one
	// would cost a round-trip — the relaxed serve mode #1319's write barrier gates.
	ConsistencyBestEffort
	// ConsistencySpeculative permits a provisional result the caller will reconcile — the level #809's SpeculationContext / #1318's suspend-resume mechanism keys on.
	ConsistencySpeculative
)

// MetaConsistency is the ToolCall.Meta key holding a call's declared ConsistencyLevel token.
const MetaConsistency = "consistency"

// String renders the level as its closed-vocabulary token; the tokens are the audit surface (verbatim in the decision journal), so they are stable.
func (l ConsistencyLevel) String() string {
	switch l {
	case ConsistencyStrict:
		return "STRICT"
	case ConsistencyBoundedStale:
		return "BOUNDED_STALE"
	case ConsistencyBestEffort:
		return "BEST_EFFORT"
	case ConsistencySpeculative:
		return "SPECULATIVE"
	default:
		return "STRICT"
	}
}

// ParseConsistency maps a token to its level, reporting ok=false for an unrecognized
// token (case-insensitive, trimmed). An unknown token is NOT an error (ConsistencyOf
// resolves it to STRICT); ok just distinguishes "explicitly STRICT" from garbage fallback.
func ParseConsistency(s string) (ConsistencyLevel, bool) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "STRICT":
		return ConsistencyStrict, true
	case "BOUNDED_STALE":
		return ConsistencyBoundedStale, true
	case "BEST_EFFORT":
		return ConsistencyBestEffort, true
	case "SPECULATIVE":
		return ConsistencySpeculative, true
	default:
		return ConsistencyStrict, false
	}
}

// ConsistencyOf reads a call's declared consistency level from Meta["consistency"],
// defaulting an absent OR unrecognized value to ConsistencyStrict — the fail-safe strictest
// contract (nil call/Meta is STRICT). It is the single reader the kernel and any policy
// rung use, so the default can never drift between call sites.
func ConsistencyOf(c *ToolCall) ConsistencyLevel {
	if c == nil || c.Meta == nil {
		return ConsistencyStrict
	}
	lvl, _ := ParseConsistency(c.Meta[MetaConsistency])
	return lvl
}
