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

// ShedTokenEquiv is the honest token-equivalent value of `shed` compaction-shed tokens,
// given `warmWitness` — the OBSERVED provider cache_read that witnesses how many of those
// tokens the provider was already serving cheaply from cache when they were dropped. It is
// the ONE source both the live gateway split (MechanismSavings.FakTokenEquiv) and the
// durable Track-2 report price a shed token at, so the two surfaces value fak's compaction
// identically by CONSTRUCTION rather than by two copies of the same rule kept in sync by
// comment (#2794/#2798).
//
// The value is a PROPORTIONAL blend, not a binary flip. The warm portion — min(shed,
// warmWitness) tokens the provider evidenced as cache_reads — is worth only ReadMultiplier
// (0.1×): dropping an already-cached token saves the read marginal, not fresh input.
// The remainder — shed BEYOND any witnessed warm prefix — is worth full input (1.0×),
// because with no cache_read witness those tokens would have been billed fresh.
//
// This replaces the aggregate-warm binary rule that discounted an ENTIRE session's shed to
// 0.1× the instant a single warm cache_read appeared (and, before that, booked every shed
// token at 1.0×). Both were the same defect — a step function on a continuous quantity —
// and produced the over-count (all-1.0×) then the under-count (a warm sliver collapsing a
// cold-dominant session ~10×). The blend has no cliff: it collapses to the pure cache-read
// marginal when warmWitness ≥ shed (fully warm) and to full input when warmWitness == 0
// (fully cold), and interpolates monotonically between, so the fak cache-value share stops
// swinging with the warm/cold mix of a session's fires.
func ShedTokenEquiv(shed, warmWitness uint64) float64 {
	warm := shed
	if warmWitness < warm {
		warm = warmWitness
	}
	cold := shed - warm
	return float64(warm)*ReadMultiplier + float64(cold)
}

// AdmissionTokens returns the billable prompt-token count for admitting a turn whose prefix
// is partially resident — the residency-discounted admission cost (#3893, vLLM's
// num_new_tokens = total − num_computed_tokens). residentPrefixTokens are the prompt tokens
// already held in the KV cache: served from a locally-resident prefix PLUS any span pulled
// across the fabric from the external / DISAGGREGATED KV tier (the two non-recompute buckets
// of the cacheobs provenance axis). Those tokens are already computed, so scheduling them costs
// nothing; the turn need only pay for the uncached suffix. A fully-resident prefix drives
// billable → 0; a cold prompt with no residency drives billable → promptTokens. This is the
// M2 "2×" lever expressed as the token count the cache lets a scheduler NOT admit —
// residency shrinks admission cost LINEARLY, with no cliff, so a warmer prefix is literally
// cheaper to admit than a colder one.
//
// It clamps defensively so a miscounted residency can never bill negative or discount more
// than the prompt: a non-positive promptTokens books as 0, and residentPrefixTokens is
// clamped into [0, promptTokens] before the subtraction (an over-reported residency caps at
// a full discount, never a negative bill). Pricing the billable tokens in $ stays the
// caller's job (ReadMultiplier for the resident share it displaced, base input for what is
// billed here), so this returns the raw token count both the pricing and the cacheobs
// provenance axes agree on — one source of truth for "the tokens residency saved".
func AdmissionTokens(promptTokens, residentPrefixTokens int) int {
	if promptTokens <= 0 {
		return 0
	}
	if residentPrefixTokens < 0 {
		residentPrefixTokens = 0
	}
	if residentPrefixTokens > promptTokens {
		residentPrefixTokens = promptTokens
	}
	return promptTokens - residentPrefixTokens
}
