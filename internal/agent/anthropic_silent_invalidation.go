package agent

// Silent provider-side cache invalidation (#2791) — the divergence between what fak PROVED
// about the bytes it shipped and what the provider actually did with them.
//
// verifySplicedBody proves the protected prefix survived a compaction splice byte-identically,
// and compactSpliceVerdict turns any failure of that proof into an identity return labelled
// CompactReasonPrefixMismatch. So a FIRED outcome (CompactReasonNone) IS the byte-equality
// witness: fak cannot fire without having proven bytes.Equal on the protected prefix.
//
// That proof is necessary but not sufficient for a provider cache hit. The provider may still
// have dropped the prefix for reasons entirely outside the splice — a TTL expiry, a capacity
// eviction, or the client moving its breakpoint — and re-created it on the next turn. The bytes
// were preserved; the cache was not. Without a counter, that miss is invisible: it looks
// identical to a compaction bug on the shed/read ledger, and the netting misattributes a
// provider-side cache-lifecycle effect to the fire.

// SilentCacheInvalidation reports whether one post-fire turn witnessed a silent provider-side
// cache invalidation: the compaction FIRED (so the protected prefix was proven byte-identical),
// yet the provider re-created that prefix instead of reading it.
//
// out is the compaction verdict for the turn's outbound body; u is the provider's OBSERVED usage
// relayed back for that same turn.
//
// The rule is deliberately narrow, and each clause is load-bearing:
//
//   - FIRED only. An identity return proves nothing about the prefix — the splice either never
//     happened or was rejected — so only CompactReasonNone carries the bytes.Equal witness this
//     signal is defined against.
//
//   - cache_read == 0. This is what scopes the check to the PROTECTED PREFIX rather than the
//     turn's creation total, and it is what keeps this signal disjoint from the induced-creation
//     burst tracked by #2785. A byte-preserved prefix is cacheable by construction, so a live
//     provider cache MUST report a read against it. A head-anchored fire deliberately bursts the
//     recent message breakpoint's cached suffix (#1408) — but it still READS its protected head,
//     so its cache_read stays positive and it is correctly not flagged here. Only a read that
//     craters to zero says the prefix ITSELF was not served from cache.
//
//   - cache_creation > 0. The provider re-ingested the prompt rather than reading it. Paired
//     with a zero read, that is the re-creation of the very prefix fak preserved.
//
// It is conservative by design: it catches a TOTAL prefix invalidation, where the read craters to
// zero. A PARTIAL invalidation — prefix part-read, part-recreated — is not separable from the
// #2785 induced burst without per-region attribution the provider does not relay, so it is left
// uncounted rather than guessed at. This under-counts; it does not false-positive. A nonzero
// count is therefore a floor on provider-side invalidation, never an accusation against a fire.
func SilentCacheInvalidation(out CompactOutcome, u Usage) bool {
	if out.Reason != CompactReasonNone {
		return false // identity return — no byte-equality witness to diverge from
	}
	return u.CacheReadInputTokens == 0 && u.CacheCreationInputTokens > 0
}
