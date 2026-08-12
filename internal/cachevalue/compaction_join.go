package cachevalue

// Compaction fire ↔ provider-turn event join (#2788). Parent epic #2783
// (cache-value netting/attribution); sibling of compaction_economics.go (#3028, the net
// verdict this join feeds) and of internal/agent/compact_join.go (#2788's receipt-side half).
//
// ClassifyCompaction scores ONE compaction event, but its two halves arrive on two
// independently collected axes: the WITNESSED shed is known at fire time, while the re-warm
// cost (RewriteCacheCreationTokens) is a provider counter that only lands with the response to
// the turn the fire rewrote. Nothing bound them. A caller assembled a CompactionSample by
// convention — "the first settled post-rewrite turn" — with no proof the counter it pasted in
// belonged to THIS fire rather than a neighbouring turn or a re-fire of the same one. That is
// the #2788 complaint exactly: netting requires joining a fire to the specific provider usage
// block it changed, and today they are separate axes.
//
// This file supplies the shared coordinate: CompactionJoinKey, the (turn sequence, monotonic ts)
// pair a fire shares with the provider usage record for the turn it affected. The turn sequence
// says WHICH turn the fire rewrote; the monotonic reading disambiguates re-fires of the SAME
// turn (a retry compacts that turn again at a strictly later reading) and is immune to
// wall-clock steps. JoinProviderUsage correlates the two streams 1:1 after the fact.
//
// The load-bearing safety property, and the reason the join is worth more than a convention:
// once a sample opts in by carrying a key, the join is AUTHORITATIVE. A key that matched exactly
// one usage record yields RewriteKnown — a net verdict backed by a proven binding. A key that
// matched none, or that is duplicated on either side, clears RewriteKnown, so ClassifyCompaction
// falls through to ReasonCompactionEconomicsUnknown and ABSTAINS. An unproven join can therefore
// never produce a net-cheaper or cache-hostile claim; it degrades to the same honest silence the
// package already uses for a dollar-blind provider row. An UNSTAMPED sample (zero key) is passed
// through verbatim — a byte-level caller that hand-assembled its own counters is not second-
// guessed, and its absent coordinate is counted apart rather than read as a failed join.
//
// CompactionJoinKey deliberately mirrors the JSON shape of internal/agent.CompactJoinKey
// (`turn_seq` / `monotonic_ts_nano`) so a receipt emitted by the agent half and a sample folded
// here join on the same wire coordinate. It is redeclared rather than imported because
// internal/cachevalue is a tier-2 std-only leaf and internal/agent is tier 5: the import would be
// an upward cross-tier edge (ARCH_LAYER_VIOLATION). Pure, like the rest of the package: no I/O,
// no clock — callers supply the monotonic reading they already hold at emission.

// CompactionJoinKey is the event-join key one compaction fire shares with the provider usage
// record for the turn it affected (#2788). Two coordinates, both caller-stamped at emission:
//
//   - TurnSeq: the 1-based sequence of the turn (request) the fire rewrote — the counter a
//     gateway caller already threads as CompactOptions.CurrentTurn. It answers WHICH turn.
//   - MonotonicTSNano: a monotonic-clock reading (nanoseconds) taken when the fire was attempted.
//     It answers WHICH ATTEMPT when one turn is compacted more than once, and cannot be perturbed
//     by a wall-clock step. It is an ORDER anchor, not a date — readers must not render it as one.
//     Using a wall-clock stamp here would break exactly the ordering the key exists to guarantee.
type CompactionJoinKey struct {
	TurnSeq         uint64 `json:"turn_seq"`
	MonotonicTSNano int64  `json:"monotonic_ts_nano"`
}

// IsZero reports whether the key is unstamped — no turn coordinate was known at emission. An
// unstamped key is not an error: it is the honest state of a sample assembled without turn
// context, and JoinProviderUsage counts it apart instead of calling it a failed join.
func (k CompactionJoinKey) IsZero() bool {
	return k == CompactionJoinKey{}
}

// ProviderTurnUsage is the provider-side half of the join: the OBSERVED cache_read /
// cache_creation one turn's provider response reported, keyed by the same CompactionJoinKey the
// turn's fire carries. The token fields are provider-relayed, never WITNESSED by fak — the join
// copies them onto the matched sample and asserts nothing about them beyond the non-negative
// clamp every count in this package gets. They are int64 to match CompactionSample's currency
// (the agent-side receipt carries the same wire fields as uint64; both are token counts).
type ProviderTurnUsage struct {
	Key                 CompactionJoinKey `json:"key"`
	CacheReadTokens     int64             `json:"cache_read_tokens"`
	CacheCreationTokens int64             `json:"cache_creation_tokens"`
}

// CompactionJoinResult is the outcome of one JoinProviderUsage pass. Samples preserves the input
// order — matched ones carrying the OBSERVED counters, the rest returned with their netting
// abstained or untouched — so a caller can fold the result straight into ClassifyCompaction. The
// counters are the join-health verdict:
//
//   - Joined: stamped samples that resolved to exactly one provider usage record. Only these
//     carry a proven net verdict.
//   - Unstamped: samples with a zero key (no turn context). Passed through verbatim; not
//     joinable, not an error.
//   - Unmatched: stamped samples whose key found NO usage record — a fire whose turn's usage went
//     unrecorded. Abstained rather than guessed.
//   - Ambiguous: stamped samples whose key appears more than once on EITHER side. The 1:1
//     guarantee is broken for that key, so the join refuses to pick a winner and abstains.
type CompactionJoinResult struct {
	Samples   []CompactionSample
	Joined    int
	Unstamped int
	Unmatched int
	Ambiguous int
}

// Clean reports whether every stamped sample resolved 1:1 — the #2788 acceptance held over this
// pass. Unstamped samples do not spoil it: they never claimed a coordinate.
func (r CompactionJoinResult) Clean() bool {
	return r.Unmatched == 0 && r.Ambiguous == 0
}

// JoinProviderUsage correlates compaction-fire samples with per-turn provider usage records by
// their shared CompactionJoinKey, 1:1 — the #2788 resolution. For each stamped sample whose key
// matches exactly one usage record (and is itself carried by exactly one sample), the record's
// OBSERVED cache_read / cache_creation are stamped onto the returned copy and RewriteKnown is
// SET, so ClassifyCompaction can score a net backed by a proven binding. A stamped key that
// matched none, or that is duplicated on either side, has its rewrite counter CLEARED and
// RewriteKnown unset — the netting abstains instead of scoring against a turn it cannot prove.
// Unstamped samples pass through verbatim. Pure: no I/O, and neither input slice's elements are
// mutated.
func JoinProviderUsage(samples []CompactionSample, usage []ProviderTurnUsage) CompactionJoinResult {
	// Count key multiplicity on both sides FIRST: ambiguity is a property of the key, so every
	// sample carrying an over-represented key must be refused, not merely the second one seen.
	sampleKeyCount := make(map[CompactionJoinKey]int, len(samples))
	for _, s := range samples {
		if !s.JoinKey.IsZero() {
			sampleKeyCount[s.JoinKey]++
		}
	}
	usageKeyCount := make(map[CompactionJoinKey]int, len(usage))
	usageByKey := make(map[CompactionJoinKey]ProviderTurnUsage, len(usage))
	for _, u := range usage {
		if u.Key.IsZero() {
			// A zero-key usage record has no coordinate to join on, and can never match a stamped
			// sample (a stamped key is nonzero by definition), so it is simply not indexed.
			continue
		}
		usageKeyCount[u.Key]++
		usageByKey[u.Key] = u
	}

	res := CompactionJoinResult{Samples: make([]CompactionSample, 0, len(samples))}
	for _, s := range samples {
		switch {
		case s.JoinKey.IsZero():
			res.Unstamped++
		case sampleKeyCount[s.JoinKey] > 1 || usageKeyCount[s.JoinKey] > 1:
			res.Ambiguous++
			s = s.abstainNet()
		case usageKeyCount[s.JoinKey] == 0:
			res.Unmatched++
			s = s.abstainNet()
		default:
			u := usageByKey[s.JoinKey]
			res.Joined++
			// Clamp like ClassifyCompaction does for the WITNESSED shed split: a
			// defensively negative provider counter must not flip the net verdict's sign.
			s.ObservedCacheReadTokens = nonNegInt64(u.CacheReadTokens)
			s.RewriteCacheCreationTokens = nonNegInt64(u.CacheCreationTokens)
			s.RewriteKnown = true
		}
		res.Samples = append(res.Samples, s)
	}
	return res
}

// abstainNet returns the sample with its provider-side counters withdrawn, so a stamped sample
// whose join could not be proven scores ReasonCompactionEconomicsUnknown rather than a net the
// evidence does not support. The WITNESSED shed fields are untouched — the "tokens shed" story
// survives an unproven join; only the "net cheaper" claim is withheld.
func (s CompactionSample) abstainNet() CompactionSample {
	s.ObservedCacheReadTokens = 0
	s.RewriteCacheCreationTokens = 0
	s.RewriteKnown = false
	return s
}

// nonNegInt64 floors a provider-relayed count at zero — the same discipline ClassifyCompaction
// applies to the WITNESSED warm/cold split — so a negative counter can never subtract from the
// re-warm cost and fake a net win.
func nonNegInt64(n int64) int64 {
	if n < 0 {
		return 0
	}
	return n
}
