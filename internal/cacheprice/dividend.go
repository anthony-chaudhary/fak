package cacheprice

// dividend.go joins this package's billing view to the cacheobs PROVENANCE axis (#3896): it
// prices the external_kv_transfer bucket — the tokens a remote / disaggregated KV tier served —
// into the net compute disaggregation actually bought. AdmissionTokens answers "how many tokens
// did residency save"; this answers "of the saving that came from ACROSS the fabric, how much
// survived the transfer toll" — the number the disaggregated-serving thesis has to justify.

// DisaggregationDividend returns the NET compute a disaggregated KV tier bought for a turn, in
// recompute-token equivalents. externalTransferTokens are the prompt tokens served across the
// fabric from a remote / L3 KV tier (cacheobs SourceStats.ExternalTransferTokens): each is a
// token this box would otherwise have RE-PREFILLED locally, so its gross saving is exactly one
// recompute. transferOverheadTokens is the caller's recompute-equivalent price of moving that KV
// across the fabric — fabric bytes converted to the per-token prefill cost, or a measured
// transfer-latency-to-token conversion — the toll disaggregation pays that a single resident box
// does not:
//
//	dividend = externalTransferTokens − transferOverheadTokens        (SIGNED)
//
// The result is SIGNED on purpose. A POSITIVE dividend is the token weight disaggregation
// genuinely bought over a single box: the remote fetch beat a local recompute. A NEGATIVE
// dividend is the honest verdict that the fabric hop cost MORE than just recomputing the span
// locally — disaggregation LOST for that span, and a throughput-parity scheduler should have
// re-prefilled instead of fetching. Zero is the break-even. This is the exact inequality the
// "disaggregated serving at throughput parity" thesis must clear, expressed in the same token
// unit as AdmissionTokens so admission cost and the disaggregation dividend read on one axis.
//
// It clamps its inputs defensively (a miscounted negative count books as 0) but deliberately
// does NOT clamp the result — the sign is the load-bearing signal and must survive. Like
// AdmissionTokens, it takes the raw external-transfer count as a plain int rather than importing
// the cacheobs provenance axis, keeping cacheprice a tier-1 foundation leaf that imports nothing
// internal.
func DisaggregationDividend(externalTransferTokens, transferOverheadTokens int) int {
	if externalTransferTokens < 0 {
		externalTransferTokens = 0
	}
	if transferOverheadTokens < 0 {
		transferOverheadTokens = 0
	}
	return externalTransferTokens - transferOverheadTokens
}

// DisaggregationWorthwhile reports whether serving externalTransferTokens from the remote tier
// beat recomputing them locally — the dividend strictly positive. It is the scheduler-facing
// predicate: admit the cross-fabric fetch only when it clears the transfer toll. A tie
// (dividend == 0, exact break-even) reports false — with no net gain, prefer the simpler local
// recompute over a fabric dependency.
func DisaggregationWorthwhile(externalTransferTokens, transferOverheadTokens int) bool {
	return DisaggregationDividend(externalTransferTokens, transferOverheadTokens) > 0
}
