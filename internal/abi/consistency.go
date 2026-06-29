package abi

// consistency.go — the per-call CONSISTENCY LEVEL (#1317), a closed enum threaded
// through ToolCall.Meta. It is the policy input + audit field a best-effort or
// speculative serve mode keys on: you cannot honestly claim a relaxed serve mode
// without an EXPLICIT level on the call (never a hidden cache mode). The kernel reads
// it via ConsistencyOf, which defaults an unset/unknown value to the STRICTEST level —
// so the field is purely additive and the default path is byte-for-byte unchanged.
//
// It rides ToolCall.Meta (the OPEN string map) rather than a new typed field so it is
// wire-transparent across every existing producer/consumer: a caller that knows nothing
// about consistency emits no key and gets STRICT; a caller that sets it gets the level
// recorded verbatim in the decision journal (kernel.Decision.Consistency). #809's
// SpeculationContext composes with the SPECULATIVE level — the level is the call's
// declared world contract, the SpeculationContext is its in-flight mechanism.

import "strings"

// ConsistencyLevel is the CLOSED relaxation ladder a call declares for its read of the
// world. STRICT is the floor (the default): the call sees the committed world and a
// relaxed mode never silently applies. The ladder loosens monotonically — each level
// admits everything stricter levels admit, plus more staleness.
type ConsistencyLevel uint8

const (
	// ConsistencyStrict reads the committed world: no stale data, no speculation. The
	// default for any call that does not declare otherwise, and the fail-safe an
	// unknown token resolves to.
	ConsistencyStrict ConsistencyLevel = iota
	// ConsistencyBoundedStale tolerates a bounded-age cached read (the staleness bound
	// is a separate policy input; this level only declares the call ACCEPTS one).
	ConsistencyBoundedStale
	// ConsistencyBestEffort tolerates an arbitrarily stale read served locally if a
	// fresh one would cost a round-trip — the relaxed serve mode #1319's write barrier
	// gates.
	ConsistencyBestEffort
	// ConsistencySpeculative permits a provisional result the caller will reconcile —
	// the level #809's SpeculationContext / #1318's suspend-resume mechanism keys on.
	ConsistencySpeculative
)

// MetaConsistency is the ToolCall.Meta key carrying the consistency level token.
const MetaConsistency = "consistency"

// String renders the level as its closed-vocabulary token. The tokens are the audit
// surface (they appear verbatim in the decision journal), so they are stable.
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
// token (case-insensitive, whitespace-trimmed). An unknown token is NOT an error the
// caller must handle — ConsistencyOf resolves it to STRICT — but the ok flag lets a
// validator distinguish "explicitly STRICT" from "garbage that fell back to STRICT".
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
// defaulting an absent OR unrecognized value to ConsistencyStrict — the fail-safe
// strictest contract. A nil call or nil Meta is STRICT. This is the single reader the
// kernel and any policy rung use, so the default can never drift between call sites.
func ConsistencyOf(c *ToolCall) ConsistencyLevel {
	if c == nil || c.Meta == nil {
		return ConsistencyStrict
	}
	lvl, _ := ParseConsistency(c.Meta[MetaConsistency])
	return lvl
}
