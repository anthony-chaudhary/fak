package session

import "time"

// throughput.go — the THROUGHPUT envelope axis as live drive state (issue #2762,
// the out-of-band operator-control epic #2753). The budget envelope has parsed
// `throughput=` / `min_throughput=` since #1573, but the parsed rates were an
// inspectable contract field only: no State field carried them, so the control
// route dropped them and nothing ever enforced the floor. This file gives the
// axis a home on the drive record and a drain rule, completing the trio with
// Budget (tokens/spend) and TimeBudget (wall-clock).
//
// TWO RATES, TWO ROLES. ExpectedTokensPerSec is the SOFT pace-shaping reference
// #1585 already defined (compose.go's Throughput: a session observed below its
// expected rate sees its planner window shrink, never a stop). MinTokensPerSec is
// the HARD floor this axis adds: a session whose SUSTAINED observed rate stays
// under the floor past the grace window drains, exactly like a token/spend/wall
// exhaustion — the operator declared the run not worth continuing below that
// pace. Only the min rate ever drains; an expected rate alone never stops a run.
//
// ACCOUNTING DISCIPLINE. Like TimeBudget, no clock is read here: the caller
// reports each turn's real duration on Usage.DurationNanos and DebitUsage
// accumulates (tokens, nanos) into this record, so the observed rate is the
// sustained lineage rate, survives a JSON round-trip/process restart, and is
// deterministically testable. Observation only accumulates while a floor is
// configured, so arming a floor mid-run judges the session from that moment
// forward — never against history recorded under no contract.

// ReasonThroughputFloor marks a session Draining because its sustained observed
// throughput stayed below the operator's configured minimum rate — the
// throughput-axis peer of ReasonBudgetSpend/ReasonTimeBudgetExhausted in the
// closed stop-reason vocabulary (decide.go).
const ReasonThroughputFloor = "THROUGHPUT_BELOW_FLOOR"

// throughputFloorGraceNanos is the minimum accumulated observation window before
// the floor is judged at all: one small fast turn (or one slow first turn) must
// not drain a session off a statistically meaningless sample. 10s of observed
// turn time bounds the false-positive window while still catching a genuinely
// crawling session within its first couple of turns.
const throughputFloorGraceNanos = int64(10 * time.Second)

// ThroughputBudget is a session's throughput envelope as live drive state: the
// configured rates plus the accumulated observation window they are judged
// against. The zero value is "axis not configured" — no observation accumulates
// and BelowFloor is always false, so a pre-#2762 State behaves byte-identically.
type ThroughputBudget struct {
	// ExpectedTokensPerSec is the soft pace-shaping reference rate (#1585). It
	// never drains a session by itself.
	ExpectedTokensPerSec float64 `json:"expected_tokens_per_sec,omitempty"`
	// MinTokensPerSec is the enforced floor: sustained observed throughput below
	// it (past the grace window) drains the session. 0 = no floor configured.
	MinTokensPerSec float64 `json:"min_tokens_per_sec,omitempty"`
	// ObservedOutputTokens / ObservedNanos are the accumulated observation window
	// DebitUsage debits: total output tokens over total reported turn duration
	// since the floor was configured. Their ratio is the sustained rate the floor
	// is judged against.
	ObservedOutputTokens int64 `json:"observed_output_tokens,omitempty"`
	ObservedNanos        int64 `json:"observed_nanos,omitempty"`
}

// IsZero supports json omitzero, so a session with no throughput envelope keeps
// the pre-#2762 wire shape byte-for-byte.
func (b ThroughputBudget) IsZero() bool {
	return b.ExpectedTokensPerSec == 0 && b.MinTokensPerSec == 0 &&
		b.ObservedOutputTokens == 0 && b.ObservedNanos == 0
}

// Bounded reports whether this axis carries an enforced floor, mirroring
// TimeBudget.Bounded / Budget.spendBounded. An expected-only envelope is NOT
// bounded — the expected rate shapes pace, it never stops a run.
func (b ThroughputBudget) Bounded() bool { return b.MinTokensPerSec > 0 }

// observe folds one completed turn's (output tokens, real duration) into the
// accumulated observation window. A turn with no reported duration is ignored —
// a duration-blind caller must never skew the sustained rate.
func (b ThroughputBudget) observe(outputTokens int, durNanos int64) ThroughputBudget {
	if durNanos <= 0 {
		return b
	}
	if outputTokens > 0 {
		b.ObservedOutputTokens += int64(outputTokens)
	}
	b.ObservedNanos += durNanos
	return b
}

// ObservedTokensPerSec is the sustained observed rate over the accumulated
// window; 0 when nothing has been observed yet.
func (b ThroughputBudget) ObservedTokensPerSec() float64 {
	if b.ObservedNanos <= 0 {
		return 0
	}
	return float64(b.ObservedOutputTokens) / (float64(b.ObservedNanos) / float64(time.Second))
}

// BelowFloor reports whether the enforced floor is breached: a configured floor,
// an observation window past the grace period, and a sustained rate under the
// minimum. An unconfigured floor (or a still-in-grace window) is never a breach.
func (b ThroughputBudget) BelowFloor() bool {
	if !b.Bounded() || b.ObservedNanos < throughputFloorGraceNanos {
		return false
	}
	return b.ObservedTokensPerSec() < b.MinTokensPerSec
}
