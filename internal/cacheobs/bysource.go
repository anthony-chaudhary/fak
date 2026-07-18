package cacheobs

// ReuseSource names WHERE a served prompt token's value came from — the PROVENANCE axis of
// cache value (#3896, vLLM's by_source counter), orthogonal to the DEPTH axis (prompt /
// cacheable / reused) the rest of this package books. The depth axis answers "how much of
// the prefix did we reuse"; this axis answers "where did that value come from" — and the
// three answers drive DIFFERENT caching decisions:
//
//   - SourceLocalCompute — the tokens were recomputed here (a miss): no cache paid for them.
//     Rising local-compute is the signal that reuse is being LEFT on the table.
//   - SourceLocalHit — served from a prefix already RESIDENT on this box (an in-session
//     prefix or a shared local KV store): near-free, no fabric hop. This is the reuse a
//     single-box cache already earns.
//   - SourceExternalTransfer — pulled ACROSS the fabric from the external / DISAGGREGATED KV
//     tier. This is the token weight disaggregation actually BOUGHT — the value a shared
//     remote pool added over what this box held locally, and the number the "disaggregated
//     serving" thesis has to justify against the transfer's latency cost. Isolating it is
//     the whole point of splitting reuse by source rather than reporting one "reused" bucket.
//
// The mapping the in-kernel taps use: session-prefix reuse ⇒ SourceLocalHit, a hit served
// from the shared tool-page / L3 store ⇒ SourceExternalTransfer, a re-prefill ⇒
// SourceLocalCompute. The vocabulary is CLOSED: an out-of-range ReuseSource is ignored,
// never bucketed to a catch-all, so a summed source snapshot can only ever under-count, never
// mis-attribute value to the wrong provenance.
type ReuseSource int

const (
	// SourceLocalCompute is a prompt token recomputed locally — a cache miss.
	SourceLocalCompute ReuseSource = iota
	// SourceLocalHit is a prompt token served from a locally-resident KV prefix.
	SourceLocalHit
	// SourceExternalTransfer is a prompt token served by a cross-instance KV transfer from
	// the external / disaggregated tier — the disaggregation dividend.
	SourceExternalTransfer
)

// String renders the Prometheus/label spelling of a source, matching vLLM's by_source label
// values so a fak exposition and a vLLM one read on the same axis. An out-of-range value
// renders "unknown" (it is never booked, so it can only appear if a caller stringifies a raw
// int) rather than panicking.
func (s ReuseSource) String() string {
	switch s {
	case SourceLocalCompute:
		return "local_compute"
	case SourceLocalHit:
		return "local_cache_hit"
	case SourceExternalTransfer:
		return "external_kv_transfer"
	default:
		return "unknown"
	}
}

// ObserveBySource books `tokens` served prompt tokens against one provenance bucket (#3896).
// It is the SOURCE-axis tap, orthogonal to Observe / ObserveSplit (the depth axis): a caller
// that knows a served span's provenance calls this once per (source, token-count) it can
// attribute, exactly as vLLM's scheduler increments its by_source counter. A non-positive
// token count is ignored (no value to attribute) and an out-of-range source is ignored (the
// vocabulary is closed), so every booked token lands in exactly one of the three buckets and
// the parts==total invariant holds by construction. Safe for concurrent use, like the rest
// of the observer.
func (o *Observer) ObserveBySource(source ReuseSource, tokens int) {
	if o == nil || tokens <= 0 {
		return
	}
	o.mu.Lock()
	switch source {
	case SourceLocalCompute:
		o.srcLocalCompute = saturatingAddU64(o.srcLocalCompute, uint64(tokens))
	case SourceLocalHit:
		o.srcLocalHit = saturatingAddU64(o.srcLocalHit, uint64(tokens))
	case SourceExternalTransfer:
		o.srcExternalTransfer = saturatingAddU64(o.srcExternalTransfer, uint64(tokens))
	}
	o.mu.Unlock()
}

// SourceStats is a point-in-time snapshot of the provenance decomposition. The load-bearing
// invariant: LocalComputeTokens + LocalHitTokens + ExternalTransferTokens == TotalTokens
// (parts==total), and ReusedTokens == LocalHitTokens + ExternalTransferTokens — the
// source-axis view of "everything not recomputed", which reconciles with the depth axis's
// ReusedTokens when a tap feeds both.
type SourceStats struct {
	LocalComputeTokens     uint64
	LocalHitTokens         uint64
	ExternalTransferTokens uint64
	// TotalTokens is the sum of the three buckets — the `total` the parts==total invariant
	// checks against. (Saturating: a real process never books near 2^64 tokens.)
	TotalTokens uint64
	// ReusedTokens is LocalHitTokens + ExternalTransferTokens — the tokens served from cache
	// regardless of provenance. It is what the depth axis calls reused, so the two axes
	// reconcile.
	ReusedTokens uint64
	// ExternalTransferRatio is ExternalTransferTokens / TotalTokens — the share of served
	// value pulled across the fabric from the disaggregated tier (the disaggregation
	// dividend). 0 when nothing has been observed (an idle process reports no phantom ratio).
	ExternalTransferRatio float64
	// LocalHitRatio is LocalHitTokens / TotalTokens — the share served from a locally-resident
	// prefix. 0 when nothing has been observed.
	LocalHitRatio float64
}

// SourceSnapshot returns the current provenance decomposition under the lock, so the derived
// totals and ratios are always consistent with the counters they are computed from. A nil
// observer returns the zero SourceStats (an idle tap never reports a phantom split).
func (o *Observer) SourceSnapshot() SourceStats {
	if o == nil {
		return SourceStats{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	s := SourceStats{
		LocalComputeTokens:     o.srcLocalCompute,
		LocalHitTokens:         o.srcLocalHit,
		ExternalTransferTokens: o.srcExternalTransfer,
	}
	s.ReusedTokens = saturatingAddU64(o.srcLocalHit, o.srcExternalTransfer)
	s.TotalTokens = saturatingAddU64(s.ReusedTokens, o.srcLocalCompute)
	if s.TotalTokens > 0 {
		s.ExternalTransferRatio = float64(s.ExternalTransferTokens) / float64(s.TotalTokens)
		s.LocalHitRatio = float64(s.LocalHitTokens) / float64(s.TotalTokens)
	}
	return s
}
