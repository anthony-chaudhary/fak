// Package cacheprice is the ONE source of truth for the provider prompt-cache price
// multipliers — the cost of a cached-prefix READ or WRITE relative to a base (uncached)
// input token. Every layer that prices cache economics reads these constants instead of
// re-declaring the literal 0.1 / 1.25 / 2.0, so the "the shed token's marginal value must
// have one source of truth" invariant (#2798) holds by CONSTRUCTION rather than by a
// drift-pin test that mirrors a bare literal.
//
// Why a dedicated leaf. The multipliers used to live in internal/gateway (a tier-4
// integrator). The fire gate (internal/agent), the resume planner (internal/resume), the
// per-session net-true ledger (internal/sessionobs), and the cache-value report
// (internal/cachevaluereport) all price the SAME cached token, but none of them may import
// the gateway upward without breaking the layered-DAG rule (or, for agent, an import cycle).
// So each copied the literal down and pinned it back to the gateway's with a test. That is
// five copies of one economic fact held together by four "if the canonical 0.1 ever moves,
// sweep every copy" comments. This package is that canonical: a tier-1 foundation leaf that
// imports nothing internal, so gateway (4) and resume (1) now READ the same symbol and the
// compaction fire gate (agent, 4) is PINNED to it by a test. The remaining copies (the
// net-true ledger in sessionobs and the Track-2 report in cachevaluereport) fold to it as
// their files free up. A rate change is now a one-line edit here, and the compiler — not a
// reviewer chasing comments — propagates it.
//
// The values are stable PUBLISHED Anthropic economics, not a fak measurement: a cache read
// bills at 0.1× base input, a 5-minute-TTL write at 1.25×, a 1-hour-TTL write at 2.0×. The
// asymmetry (a write costs MORE than an uncached read; each subsequent read recovers 0.9×)
// is why caching is a net win only once reads accrue — a fact the pricing layers make
// mechanical rather than hiding.
package cacheprice

const (
	// ReadMultiplier is the price of a cached-prefix READ relative to base input (0.1×).
	// A compaction-shed token that landed on a WARM prefix was already a provider cache_read,
	// so dropping it saves only this marginal — not full input (1.0×). Booking warm shed at
	// 1.0× is the ~10× over-valuation #2794/#2798 correct.
	ReadMultiplier = 0.1
	// Write5mMultiplier is the price of a 5-minute-TTL cache WRITE relative to base input (1.25×).
	Write5mMultiplier = 1.25
	// Write1hMultiplier is the price of a 1-hour-TTL cache WRITE relative to base input (2.0×).
	Write1hMultiplier = 2.0
)
