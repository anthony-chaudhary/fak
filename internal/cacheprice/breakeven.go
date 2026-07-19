package cacheprice

// breakeven.go adds the one thing the opaque-overhead leaves cannot express: how a fetch's toll
// AMORTIZES over prefix length. DisaggregationDividend takes the transfer overhead as a single
// opaque total; but a real fabric fetch pays a FIXED per-transfer floor (connection, RPC, first-
// byte latency) plus a per-token wire cost, while the recompute it saves is purely per-token. That
// makes disaggregation a LENGTH question: short prefixes cannot amortize the fixed floor and must
// recompute; only past a break-even length does the fetch win. This is the gate that PRECEDES the
// per-turn dividend and the router — a prefix shorter than the break-even is never a disaggregation
// candidate, so the scheduler need not even price it. Pure arithmetic, import-free, tier-1.

// TransferBreakEvenLength returns the smallest prefix length (in tokens) at which fetching a prefix
// across the fabric STRICTLY beats recomputing it locally, under a linear transfer-cost model:
//
//	recompute(L) = recomputePerToken * L                          // saved by having the KV
//	transfer(L)  = fixedOverheadTokens + transferPerToken * L      // paid to move it
//	worthwhile   ⟺ recompute(L) > transfer(L) ⟺ slope*L > fixedOverheadTokens
//
// where slope = recomputePerToken − transferPerToken. The break-even is the least L clearing that
// strict inequality (matching DisaggregationWorthwhile's strict-positive convention): with a
// positive slope, breakEven = fixedOverheadTokens/slope + 1 (integer floor), and everWorthwhile is
// true — a long enough prefix always amortizes any fixed floor. When slope ≤ 0 the per-token wire
// cost meets or exceeds the per-token recompute, so NO length ever wins regardless of the floor:
// everWorthwhile is false and breakEven is returned as 0 (meaningless — callers must check the
// bool). Inputs clamp defensively to non-negative; a zero floor yields breakEven 1 (any non-empty
// prefix wins when there is no fixed cost to amortize).
func TransferBreakEvenLength(fixedOverheadTokens, recomputePerToken, transferPerToken int) (breakEven int, everWorthwhile bool) {
	if fixedOverheadTokens < 0 {
		fixedOverheadTokens = 0
	}
	if recomputePerToken < 0 {
		recomputePerToken = 0
	}
	if transferPerToken < 0 {
		transferPerToken = 0
	}
	slope := recomputePerToken - transferPerToken
	if slope <= 0 {
		return 0, false
	}
	return fixedOverheadTokens/slope + 1, true
}

// TransferWorthwhileAtLength reports whether a prefix of prefixTokens is strictly worth fetching
// across the fabric under the same linear model — the direct per-prefix gate: slope*prefixTokens >
// fixedOverheadTokens. It is the length-aware companion to DisaggregationWorthwhile (which takes an
// already-computed overhead total): true here for exactly the lengths at or above
// TransferBreakEvenLength when that break-even exists, and never true when it does not. A
// non-positive prefix is never worthwhile. Inputs clamp defensively to non-negative.
func TransferWorthwhileAtLength(prefixTokens, fixedOverheadTokens, recomputePerToken, transferPerToken int) bool {
	if prefixTokens <= 0 {
		return false
	}
	if fixedOverheadTokens < 0 {
		fixedOverheadTokens = 0
	}
	if recomputePerToken < 0 {
		recomputePerToken = 0
	}
	if transferPerToken < 0 {
		transferPerToken = 0
	}
	slope := recomputePerToken - transferPerToken
	return slope*prefixTokens > fixedOverheadTokens
}
