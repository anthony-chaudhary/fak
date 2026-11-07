package model

// expert_ring_stable.go — the scan-resistant stable set of the routed-expert ring.
//
// Why: a forward visits MoE layers in a fixed cycle, and a prefill row routes nearly every expert of
// every layer, so the per-row expert working set is the whole routed band. When the ring is smaller
// than the band, recency (LRU) and decaying-heat rankings both evict each expert before the next row
// reaches its layer again: the classic cyclic-scan pathology, cross-row hit fraction 0 regardless of
// ring size (witnessed: TestExpertRingStableSetSurvivesLayerCyclicSweep; DS4.1 on one H200 streamed
// the full ~183 GiB band per 586-token prefill row through a 32 GiB ring).
//
// What: while the stable set has room (stableCap), a cold admit is pinned. Pinned residents are never
// victims (polymodel's invariant), so they survive every sweep and serve hits on every later row and
// token; the remaining budget - stableCap is the LRU churn region every other expert (and every live
// hold span) uses. Expected sweep hit fraction ~ stableCap / band instead of 0.
//
// Off by default: stableCap == 0 is the historical ring byte-for-byte. A serve sets it only for a
// large non-token ring smaller than the band (cmd/fak serveExpertRingStableBytes).

// stableAdmit reports whether a cold admit of weightBytes joins the stable set.
func (r *pagedRing) stableAdmit(weightBytes int64) bool {
	return r.stableCap > 0 && weightBytes > 0 && r.stableBytes+weightBytes <= r.stableCap
}

// noteStableAdmit books an admitted stable-set weight.
func (r *pagedRing) noteStableAdmit(stable bool, weightBytes int64) {
	if stable {
		r.stableBytes += weightBytes
	}
}
